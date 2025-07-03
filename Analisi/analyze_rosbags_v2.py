#!/usr/bin/env python3
"""
analyze_rosbags_v2.py
------------------
Batch‑analyse ROS 2 Jazzy rosbags to compute the RMSE between:
  • Vicon reference pose  :  child = vicon/magnetic_drone/magnetic_drone, parent = vicon/world
  • EKF_CF estimator       :  child = EKF_CF,                    parent = vicon/world

The script:
  1. Reads an Excel metadata file that lists all tests and associated rosbag folders.
  2. Iterates (multiprocess) over each bag and extracts only the `/tf` topic to minimise I/O.
  3. Synchronises the two pose streams (nearest‑neighbour within --sync-epsilon s).
  4. Computes RMSE & MAE per axis and overall, stores them in CSV.
  5. Generates publication‑ready plots (IEEE TRO compliant) in PNG & PDF.
  6. Creates 3D trajectory plots with anchor positions.
  7. Visualizes pointwise errors along the trajectory.
  8. Computes standard deviation metrics.
  9. Generates a summary figure with best scores per category.

Note: This version analyzes only EKF_CF vs Vicon (no cf21 or Opt transformations).

Requirements:
  sudo apt install python3-rosbag2-py python3-rclpy python3-numpy python3-pandas python3-matplotlib
  pip  install  tqdm

Usage example
-------------

# da dentro la cartella con le cartelle dei rosbags
python analyze_rosbags.py --excel ETH.xlsx --root . --out results --workers 4

# per mandare i risultati sulla cartel OneDrive condivisa
python analyze_rosbags.py --excel '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/ROSBAGS/ETH.xlsx' --root '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/ROSBAGS' --out '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/risultati' --workers 4


Author: ChatGPT (o3) – 2025‑06‑30
"""

import argparse, os, pathlib, sys, math, multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from pathlib import Path
import yaml

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

# -------------------------  CONFIGURATION  -------------------------------- #
TF_TOPIC            = '/tf'
CH_REF_CHILD        = 'vicon/magnetic_drone/magnetic_drone'
CH_REF_PARENT       = 'vicon/world'
CH_EKF_CF_CHILD     = 'EKF_CF'
CH_EKF_CF_PARENT    = 'vicon/world'

DOG_CHILD           = 'vicon/robodog_magnetic/robodog_magnetic'
DOG_PARENT          = 'vicon/world'

# Anchor frame names
ANCHOR_FRAMES       = ['Nero', 'Giallo', 'Grigio', 'Rosso']
ANCHOR_COLORS       = ['black', 'gold', 'gray', 'red']
ANCHOR_PARENT       = 'world'  # Parent frame for anchor transformations

SYNC_EPSILON        = 0.01        # [s] max gap between reference & estimates
PLOT_FONT_SIZE      = 9
DPI                 = 300

# IEEE Publication Style Colors
VICON_COLOR         = '#1f77b4'   # Professional blue
EKF_CF_COLOR        = '#d62728'   # Professional red
ERROR_YLIM_MAX      = 1.0         # Maximum Y axis for error plots [m]
# ------------------------------------------------------------------------- #

def read_excel_list(excel_path: str):
    """Return dataframe with columns category, label, rosbags(list)."""
    df = pd.read_excel(excel_path, sheet_name=0)
    header_row = df.iloc[0, 2:].reset_index(drop=True)
    data       = df.iloc[1:].reset_index(drop=True)

    categories = data.iloc[:, 0].ffill()
    labels     = data.iloc[:, 1]
    bag_cols   = data.iloc[:, 2:]

    bag_lists  = []
    for _, row in bag_cols.iterrows():
        lst = []
        for cell in row:
            if isinstance(cell, str) and 'rosbag2' in cell:
                name = cell.strip()
                # Append '_processed' suffix to rosbag folder names
                lst.append(f"{name}_processed")
        bag_lists.append(lst)

    out = pd.DataFrame({'category': categories,
                        'label'    : labels,
                        'rosbags'  : bag_lists})
    return out

# ------------------------------------------------------------------------- #

def detect_storage_id(bag_dir: Path) -> str:
    meta = bag_dir / 'metadata.yaml'
    if meta.exists():
        with open(meta, 'r') as f:
            return yaml.safe_load(f).get('storage_identifier', '')
    return ''       

def rosbag_iter_tf(bag_path: pathlib.Path):
    """Yield (time_sec, transform_msg) only for TF_TOPIC."""
    reader = SequentialReader()
     # vuoto = autodetect

    sid = detect_storage_id(bag_path)

    storage_options   = StorageOptions(uri=str(bag_path), storage_id=sid)
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    tf_type     = None
    for t in topic_types:
        if t.name == TF_TOPIC:
            tf_type = t.type
            break
    if tf_type is None:
        raise RuntimeError(f"{bag_path}: no {TF_TOPIC} found")

    msg_type = get_message(tf_type)

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic != TF_TOPIC:  # fast path
            continue
        msg = deserialize_message(data, msg_type)
        stamp_sec = t / 1e9
        for tf in msg.transforms:
            yield stamp_sec, tf

# ------------------------------------------------------------------------- #
def extract_streams(bag_dir: pathlib.Path):
    """Return dict frame_name -> np.ndarray[[t,x,y,z]]."""
    streams = { 'ref': [], 'ekf_cf': [], 'dog': [] }
    anchors = {name: [] for name in ANCHOR_FRAMES}
    
    for t, tf in rosbag_iter_tf(bag_dir):
        child  = tf.child_frame_id
        parent = tf.header.frame_id
        
        if child == CH_REF_CHILD   and parent == CH_REF_PARENT:
            streams['ref'].append((t,
                                    tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z))
        elif child == CH_EKF_CF_CHILD and parent == CH_EKF_CF_PARENT:
            streams['ekf_cf'].append((t,
                                    tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z))
        elif child == DOG_CHILD    and parent == DOG_PARENT:
            streams['dog'].append((t,
                                    tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z))
        # Capture anchor positions - verifica anche il parent frame corretto
        elif child in ANCHOR_FRAMES and parent == ANCHOR_PARENT:
            anchors[child].append((t,
                                   tf.transform.translation.x,
                                   tf.transform.translation.y,
                                   tf.transform.translation.z))

    for k in streams:
        if streams[k]:
            streams[k] = np.asarray(streams[k])
        else:
            streams[k] = np.empty((0,4))
    
    # Convert anchor lists to arrays
    for k in anchors:
        if anchors[k]:
            anchors[k] = np.asarray(anchors[k])
        else:
            anchors[k] = np.empty((0,4))
    
    return streams, anchors

# ------------------------------------------------------------------------- #
def apply_static_transform(streams, anchors):
    """Non applica più trasformazioni statiche dato che tutti i frame sono già in vicon/world."""
    # print("INFO: Trasformazione statica disabilitata - tutti i frame sono già in vicon/world")
    
    # # Debug info per verificare i dati
    # if streams['ref'].size > 0:
    #     print(f"DEBUG: Ref (vicon/world) - primo punto: {streams['ref'][0, 1:4]}")
    # if streams['ekf_cf'].size > 0:
    #     print(f"DEBUG: EKF_CF (vicon/world) - primo punto: {streams['ekf_cf'][0, 1:4]}")
    # if streams['dog'].size > 0:
    #     print(f"DEBUG: Dog (vicon/world) - primo punto: {streams['dog'][0, 1:4]}")
    
    return streams, anchors

# ------------------------------------------------------------------------- #
def align_streams(streams):
    """Return times, ref_xyz, ekf_cf_xyz aligned arrays."""
    if streams['ekf_cf'].size == 0 or streams['ref'].size == 0:
        raise RuntimeError("Missing one or more streams")
    
    base_t = streams['ekf_cf'][:,0]   # timestamps of EKF_CF
    ref_xyz  = np.zeros((len(base_t), 3))
    ref_ok   = np.zeros(len(base_t), dtype=bool)

    # index pointers
    idx_ref = 0
    ref_arr = streams['ref']

    for i, t in enumerate(base_t):
        # REF
        while idx_ref+1 < len(ref_arr) and ref_arr[idx_ref+1,0] <= t:
            idx_ref += 1
        if abs(ref_arr[idx_ref,0] - t) <= SYNC_EPSILON:
            ref_xyz[i] = ref_arr[idx_ref,1:4]
            ref_ok[i] = True

    mask = ref_ok
    t_out = base_t[mask]
    ekf_cf_xyz = streams['ekf_cf'][mask][:,1:4]
    ref_xyz = ref_xyz[mask]
    return t_out, ref_xyz, ekf_cf_xyz

# ------------------------------------------------------------------------- #
def compute_rmse(ref, est):
    err = est - ref
    rmse = np.sqrt(np.mean(err**2, axis=0))
    rmse_total = np.sqrt(np.mean(err**2))
    return rmse, rmse_total, err

# ------------------------------------------------------------------------- #
def plot_3d_trajectory(t, ref, ekf_cf, anchors, output_path_3d, output_path_2d, 
                       output_path_yz=None, output_path_xz=None):
    """Create 3D trajectory plot in vicon/world frame and 2D projection plots."""
    
    # Calculate axis limits for uniform scaling
    all_points = np.vstack([ref, ekf_cf])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    # Add margin (10% of range)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    margin = 0.1
    
    x_lim = [x_min - margin * x_range, x_max + margin * x_range]
    y_lim = [y_min - margin * y_range, y_max + margin * y_range]
    z_lim = [z_min - margin * z_range, z_max + margin * z_range]
    
    # 3D World frame plot
    fig = plt.figure(figsize=(8, 6), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
            label='Vicon', alpha=0.9)
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], color=EKF_CF_COLOR, 
            linewidth=2.0, linestyle='--', label='EKF_CF', alpha=0.9)
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=120, label=name, edgecolors='k', linewidth=1)
    
    # Mark start and end points
    ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color='green', marker='o', 
               s=100, label='Start', edgecolors='k', linewidth=1.5, zorder=10)
    ax.scatter(ref[-1, 0], ref[-1, 1], ref[-1, 2], color='orange', marker='s', 
               s=100, label='End', edgecolors='k', linewidth=1.5, zorder=10)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE)
    
    # Position legend to avoid trajectory overlap
    ax.legend(fontsize=PLOT_FONT_SIZE-1, loc='upper left', 
              bbox_to_anchor=(0.02, 0.98), framealpha=0.9, fancybox=True, shadow=True)
    ax.set_title('3D Trajectory in Vicon/World Frame', fontsize=PLOT_FONT_SIZE+1, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path_3d, bbox_inches='tight')
    # Also save as PNG
    plt.savefig(str(output_path_3d).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2D XY trajectory plot
    fig = plt.figure(figsize=(8, 6), dpi=DPI)
    ax = fig.add_subplot(111)
    
    ax.plot(ref[:, 0], ref[:, 1], color=VICON_COLOR, linewidth=2.5, 
            label='Vicon', alpha=0.9)
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], color=EKF_CF_COLOR, linewidth=2.0, 
            linestyle='--', label='EKF_CF', alpha=0.9)
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=120, label=name, edgecolors='k', linewidth=1)
    
    # Mark start and end points
    ax.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=100, 
               label='Start', edgecolors='k', linewidth=1.5, zorder=10)
    ax.scatter(ref[-1, 0], ref[-1, 1], color='orange', marker='s', s=100, 
               label='End', edgecolors='k', linewidth=1.5, zorder=10)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    
    # Create title with colored text using matplotlib's text formatting
    title_text = 'XY Trajectory in Vicon/World Frame'
    # Simple clear legend without symbols that might overlap
    legend_text = r'Vicon (solid) — EKF_CF (dashed) — Start (●) — End (■)'
    
    ax.set_title(f'{title_text}\n{legend_text}', 
                 fontsize=PLOT_FONT_SIZE+1, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path_2d, bbox_inches='tight')
    # Also save as PNG
    plt.savefig(str(output_path_2d).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2D YZ trajectory plot
    if output_path_yz:
        fig = plt.figure(figsize=(8, 6), dpi=DPI)
        ax = fig.add_subplot(111)
        
        ax.plot(ref[:, 1], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
                label='Vicon', alpha=0.9)
        ax.plot(ekf_cf[:, 1], ekf_cf[:, 2], color=EKF_CF_COLOR, linewidth=2.0, 
                linestyle='--', label='EKF_CF', alpha=0.9)
        
        # Plot anchors if available (YZ projection)
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
                if pos is not None:
                    ax.scatter(pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                               s=120, label=name, edgecolors='k', linewidth=1)
        
        # Mark start and end points
        ax.scatter(ref[0, 1], ref[0, 2], color='green', marker='o', s=100, 
                   label='Start', edgecolors='k', linewidth=1.5, zorder=10)
        ax.scatter(ref[-1, 1], ref[-1, 2], color='orange', marker='s', s=100, 
                   label='End', edgecolors='k', linewidth=1.5, zorder=10)
        
        ax.set_xlim(y_lim)
        ax.set_ylim(z_lim)
        ax.set_xlabel('Y [m]', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('Z [m]', fontsize=PLOT_FONT_SIZE)
        # Create title with colored text using matplotlib's text formatting
        title_text = 'YZ Trajectory in Vicon/World Frame'
        # Simple clear legend without symbols that might overlap
        legend_text = r'Vicon (solid) — EKF_CF (dashed) — Start (●) — End (■)'
        
        ax.set_title(f'{title_text}\n{legend_text}', 
                     fontsize=PLOT_FONT_SIZE+1, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(output_path_yz, bbox_inches='tight')
        # Also save as PNG
        plt.savefig(str(output_path_yz).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2D XZ trajectory plot
    if output_path_xz:
        fig = plt.figure(figsize=(8, 6), dpi=DPI)
        ax = fig.add_subplot(111)
        
        ax.plot(ref[:, 0], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
                label='Vicon', alpha=0.9)
        ax.plot(ekf_cf[:, 0], ekf_cf[:, 2], color=EKF_CF_COLOR, linewidth=2.0, 
                linestyle='--', label='EKF_CF', alpha=0.9)
        
        # Plot anchors if available (XZ projection)
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
                if pos is not None:
                    ax.scatter(pos[0], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                               s=120, label=name, edgecolors='k', linewidth=1)
        
        # Mark start and end points
        ax.scatter(ref[0, 0], ref[0, 2], color='green', marker='o', s=100, 
                   label='Start', edgecolors='k', linewidth=1.5, zorder=10)
        ax.scatter(ref[-1, 0], ref[-1, 2], color='orange', marker='s', s=100, 
                   label='End', edgecolors='k', linewidth=1.5, zorder=10)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(z_lim)
        ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('Z [m]', fontsize=PLOT_FONT_SIZE)
        
        # Create title with colored text using matplotlib's text formatting
        title_text = 'XZ Trajectory in Vicon/World Frame'
        # Simple clear legend without symbols that might overlap
        legend_text = r'Vicon (solid) — EKF_CF (dashed) — Start (●) — End (■)'
        
        ax.set_title(f'{title_text}\n{legend_text}', 
                     fontsize=PLOT_FONT_SIZE+1, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(output_path_xz, bbox_inches='tight')
        # Also save as PNG
        plt.savefig(str(output_path_xz).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------------- #
def plot_errors(t, err_ekf_cf, output_path):
    """Create plots of pointwise errors along the trajectory."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), dpi=DPI, sharex=True)
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z']
    
    # Plot errors per axis
    for i in range(3):
        axes[i].plot(t, err_ekf_cf[:, i], color=EKF_CF_COLOR, linewidth=1.5, 
                     alpha=0.8, label='EKF_CF Error')
        axes[i].set_ylabel(f'{labels[i]} Error [m]', fontsize=PLOT_FONT_SIZE)
        axes[i].grid(True, linestyle=':', alpha=0.7)
        axes[i].set_ylim([-ERROR_YLIM_MAX, ERROR_YLIM_MAX])
        if i == 0:
            axes[i].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
        axes[i].tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-1)
    
    # Plot total error
    total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
    axes[3].plot(t, total_err_ekf_cf, color=EKF_CF_COLOR, linewidth=2.0, 
                 label='EKF_CF Total Error')
    axes[3].set_ylabel('Total Error [m]', fontsize=PLOT_FONT_SIZE)
    axes[3].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    axes[3].grid(True, linestyle=':', alpha=0.7)
    axes[3].set_ylim([0, ERROR_YLIM_MAX])
    axes[3].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    axes[3].tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-1)
    
    plt.suptitle('Position Errors vs Reference', fontsize=PLOT_FONT_SIZE+1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    # Also save as PNG
    plt.savefig(str(output_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------- #
def analyse_bag(bag_dir, out_dir):
    bag_dir = pathlib.Path(bag_dir)
    
    # Create subfolder for this rosbag
    bag_out_dir = out_dir / bag_dir.name
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    
    streams, anchors = extract_streams(bag_dir)
    
    # Apply static transform to align frames correctly
    streams, anchors = apply_static_transform(streams, anchors)
    
    t, ref, ekf_cf = align_streams(streams)
    
    # Error calculations
    rmse_ekf_cf_axis, rmse_ekf_cf_tot, err_ekf_cf = compute_rmse(ref, ekf_cf)

    # Save CSV summary
    csv_path = bag_out_dir / f"{bag_dir.name}_metrics.csv"
    pd.DataFrame({
        'metric' : ['RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_TOT'],
        'ekf_cf': np.concatenate([rmse_ekf_cf_axis, [rmse_ekf_cf_tot]])
    }).to_csv(csv_path, index=False)

    # Create traditional plots (as before)
    plot_path = bag_out_dir / f"{bag_dir.name}_traj.pdf"
    plt.figure(figsize=(10, 8), dpi=DPI)
    axx = [plt.subplot(3,1,i+1) for i in range(3)]
    labels = ['x','y','z']
    for i, ax in enumerate(axx):
        ax.plot(t, ref[:,i], color=VICON_COLOR, linewidth=2.0, label='Vicon', alpha=0.9)
        ax.plot(t, ekf_cf[:,i], color=EKF_CF_COLOR, linewidth=1.8, linestyle='--', 
                label='EKF_CF', alpha=0.9)
        ax.set_ylabel(labels[i]+' [m]', fontsize=PLOT_FONT_SIZE)
        ax.grid(True, which='both', linestyle=':', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-1)
        if i==0:
            ax.legend(fontsize=PLOT_FONT_SIZE, ncol=2, loc='upper center')
    axx[-1].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    plt.suptitle('Position Trajectories vs Time', fontsize=PLOT_FONT_SIZE+1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    # Also save as PNG
    plt.savefig(str(plot_path).replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create 3D and 2D trajectory plots
    traj3d_path = bag_out_dir / f"{bag_dir.name}_traj3d.pdf"
    traj2d_path = bag_out_dir / f"{bag_dir.name}_traj2d.pdf"
    trajyz_path = bag_out_dir / f"{bag_dir.name}_trajyz.pdf"
    trajxz_path = bag_out_dir / f"{bag_dir.name}_trajxz.pdf"
    plot_3d_trajectory(t, ref, ekf_cf, anchors, traj3d_path, traj2d_path, trajyz_path, trajxz_path)
    
    # Create error plots
    error_path = bag_out_dir / f"{bag_dir.name}_errors.pdf"
    plot_errors(t, err_ekf_cf, error_path)
    
    return csv_path, plot_path, rmse_ekf_cf_tot, bag_out_dir

def worker(bag_dir, out_root):
    try:
        csv, plot, r_ekf_cf, bag_out_dir = analyse_bag(bag_dir, out_root)
        return (bag_dir.name, r_ekf_cf, str(csv), str(plot), str(bag_out_dir))
    except Exception as e:
        return (bag_dir.name, 'ERROR', str(e), '', '')

# ------------------------------------------------------------------------- #
def create_category_summary_figure(table, results_df, out_dir):
    """Create comprehensive summary figure with 3D trajectories and 2D views per category."""
    # Convert string errors to NaN
    results_df['RMSE_ekf_cf'] = pd.to_numeric(results_df['RMSE_ekf_cf'], errors='coerce')
    
    # Merge with metadata table
    metadata_dict = {}
    for _, row in table.iterrows():
        for bag in row['rosbags']:
            bag_name = Path(bag).name
            metadata_dict[bag_name] = {'category': row['category'], 'label': row['label']}
    
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')
    metadata_df.index.name = 'bag'
    metadata_df.reset_index(inplace=True)
    
    combined_df = results_df.merge(metadata_df, on='bag', how='left')
    # Remove error entries and ensure we have valid categories with numeric data
    combined_df = combined_df.dropna(subset=['RMSE_ekf_cf', 'category'])
    
    categories = combined_df['category'].unique()
    categories = [cat for cat in categories if pd.notna(cat)]
    
    # Filter out categories that don't have any valid numeric data
    valid_categories = []
    for cat in categories:
        cat_data = combined_df[combined_df['category'] == cat]
        if len(cat_data) > 0 and not cat_data['RMSE_ekf_cf'].isna().all():
            valid_categories.append(cat)
    
    categories = valid_categories
    
    if len(categories) == 0:
        print("No valid categories found for summary figure")
        return None
    
    # Calculate global axis limits for all 3D plots
    global_x_min, global_x_max = float('inf'), float('-inf')
    global_y_min, global_y_max = float('inf'), float('-inf')
    global_z_min, global_z_max = float('inf'), float('-inf')
    
    # First pass: collect all trajectory data to determine global limits
    trajectory_data = {}
    for cat_idx, category in enumerate(categories):
        cat_data = combined_df[combined_df['category'] == category]
        if len(cat_data) == 0:
            continue
        best_idx = cat_data['RMSE_ekf_cf'].idxmin()
        best_bag = cat_data.loc[best_idx]
        best_bag_name = best_bag['bag']
        best_bag_dir = out_dir / best_bag_name
        
        try:
            streams, anchors = extract_streams(best_bag_dir.parent.parent / best_bag_name)
            streams, anchors = apply_static_transform(streams, anchors)
            t, ref, ekf_cf = align_streams(streams)
            trajectory_data[category] = (t, ref, ekf_cf, anchors)
            
            # Update global limits
            all_points = np.vstack([ref, ekf_cf])
            global_x_min = min(global_x_min, all_points[:, 0].min())
            global_x_max = max(global_x_max, all_points[:, 0].max())
            global_y_min = min(global_y_min, all_points[:, 1].min())
            global_y_max = max(global_y_max, all_points[:, 1].max())
            global_z_min = min(global_z_min, all_points[:, 2].min())
            global_z_max = max(global_z_max, all_points[:, 2].max())
        except Exception as e:
            print(f"Error loading data for {best_bag_name}: {e}")
            continue
    
    # Add margin to global limits
    x_range = global_x_max - global_x_min
    y_range = global_y_max - global_y_min
    z_range = global_z_max - global_z_min
    margin = 0.1
    
    global_x_lim = [global_x_min - margin * x_range, global_x_max + margin * x_range]
    global_y_lim = [global_y_min - margin * y_range, global_y_max + margin * y_range]
    global_z_lim = [global_z_min - margin * z_range, global_z_max + margin * z_range]
    
    # Create figure with subplots: 8 columns (3D traj, XY, YZ, XZ, Error X, Error Y, Error Z, Error Total) x n_categories rows
    fig = plt.figure(figsize=(32, 5*len(categories)), dpi=DPI)
    
    for cat_idx, category in enumerate(categories):
        cat_data = combined_df[combined_df['category'] == category]
        
        if len(cat_data) == 0 or category not in trajectory_data:
            continue
            
        # Get pre-computed trajectory data
        t, ref, ekf_cf, anchors = trajectory_data[category]
        _, rmse_total, err_ekf_cf = compute_rmse(ref, ekf_cf)
        
        # Find the best bag info for this category
        best_rmse_idx = cat_data['RMSE_ekf_cf'].idxmin()
        best_bag_label = cat_data.loc[best_rmse_idx, 'label']
        best_bag_name = cat_data.loc[best_rmse_idx, 'bag']
        
        # 3D Trajectory plot
        ax1 = fig.add_subplot(len(categories), 8, cat_idx*8 + 1, projection='3d')
        ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
                 label='Vicon', alpha=0.9)
        ax1.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], color=EKF_CF_COLOR, 
                 linewidth=2.0, linestyle='--', label='EKF_CF', alpha=0.9)
        
        # Plot anchors if available
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4]
                ax1.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=60, label=name, edgecolors='k', alpha=0.8)
        
        # Apply global axis limits
        ax1.set_xlim(global_x_lim)
        ax1.set_ylim(global_y_lim)
        ax1.set_zlim(global_z_lim)
        
        ax1.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-2)
        ax1.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE-2)
        ax1.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE-2)
        ax1.set_title(f'{category} - 3D Trajectory\nRMSE: {rmse_total:.4f}m ({best_bag_name})', 
                      fontsize=PLOT_FONT_SIZE-2)
        
        # Position legend to avoid trajectory overlap
        if cat_idx == 0:
            ax1.legend(fontsize=PLOT_FONT_SIZE-3, loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # XY trajectory plot
        ax2 = fig.add_subplot(len(categories), 8, cat_idx*8 + 2)
        ax2.plot(ref[:, 0], ref[:, 1], color=VICON_COLOR, linewidth=2.5, 
                 label='Vicon', alpha=0.9)
        ax2.plot(ekf_cf[:, 0], ekf_cf[:, 1], color=EKF_CF_COLOR, linewidth=2.0, 
                 linestyle='--', label='EKF_CF', alpha=0.9)
        ax2.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=60, zorder=10)
        ax2.scatter(ref[-1, 0], ref[-1, 1], color='orange', marker='s', s=60, zorder=10)
        ax2.set_xlim(global_x_lim)
        ax2.set_ylim(global_y_lim)
        ax2.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-2)
        ax2.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE-2)
        ax2.set_title(f'XY View', fontsize=PLOT_FONT_SIZE-2)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # YZ trajectory plot
        ax3 = fig.add_subplot(len(categories), 8, cat_idx*8 + 3)
        ax3.plot(ref[:, 1], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
                 label='Vicon', alpha=0.9)
        ax3.plot(ekf_cf[:, 1], ekf_cf[:, 2], color=EKF_CF_COLOR, linewidth=2.0, 
                 linestyle='--', label='EKF_CF', alpha=0.9)
        ax3.scatter(ref[0, 1], ref[0, 2], color='green', marker='o', s=60, zorder=10)
        ax3.scatter(ref[-1, 1], ref[-1, 2], color='orange', marker='s', s=60, zorder=10)
        ax3.set_xlim(global_y_lim)
        ax3.set_ylim(global_z_lim)
        ax3.set_xlabel('Y [m]', fontsize=PLOT_FONT_SIZE-2)
        ax3.set_ylabel('Z [m]', fontsize=PLOT_FONT_SIZE-2)
        ax3.set_title(f'YZ View', fontsize=PLOT_FONT_SIZE-2)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        
        # XZ trajectory plot
        ax4 = fig.add_subplot(len(categories), 8, cat_idx*8 + 4)
        ax4.plot(ref[:, 0], ref[:, 2], color=VICON_COLOR, linewidth=2.5, 
                 label='Vicon', alpha=0.9)
        ax4.plot(ekf_cf[:, 0], ekf_cf[:, 2], color=EKF_CF_COLOR, linewidth=2.0, 
                 linestyle='--', label='EKF_CF', alpha=0.9)
        ax4.scatter(ref[0, 0], ref[0, 2], color='green', marker='o', s=60, zorder=10)
        ax4.scatter(ref[-1, 0], ref[-1, 2], color='orange', marker='s', s=60, zorder=10)
        ax4.set_xlim(global_x_lim)
        ax4.set_ylim(global_z_lim)
        ax4.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-2)
        ax4.set_ylabel('Z [m]', fontsize=PLOT_FONT_SIZE-2)
        ax4.set_title(f'XZ View', fontsize=PLOT_FONT_SIZE-2)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        # Error plots
        labels = ['X', 'Y', 'Z']
        
        # X Error plot
        ax5 = fig.add_subplot(len(categories), 8, cat_idx*8 + 5)
        ax5.plot(t, err_ekf_cf[:, 0], color=EKF_CF_COLOR, linewidth=1.5, alpha=0.8)
        ax5.set_ylabel('X Error [m]', fontsize=PLOT_FONT_SIZE-2)
        ax5.set_title('X Error', fontsize=PLOT_FONT_SIZE-2)
        ax5.grid(True, linestyle=':', alpha=0.7)
        ax5.set_ylim([-ERROR_YLIM_MAX, ERROR_YLIM_MAX])
        ax5.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-3)
        
        # Y Error plot
        ax6 = fig.add_subplot(len(categories), 8, cat_idx*8 + 6)
        ax6.plot(t, err_ekf_cf[:, 1], color=EKF_CF_COLOR, linewidth=1.5, alpha=0.8)
        ax6.set_ylabel('Y Error [m]', fontsize=PLOT_FONT_SIZE-2)
        ax6.set_title('Y Error', fontsize=PLOT_FONT_SIZE-2)
        ax6.grid(True, linestyle=':', alpha=0.7)
        ax6.set_ylim([-ERROR_YLIM_MAX, ERROR_YLIM_MAX])
        ax6.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-3)
        
        # Z Error plot
        ax7 = fig.add_subplot(len(categories), 8, cat_idx*8 + 7)
        ax7.plot(t, err_ekf_cf[:, 2], color=EKF_CF_COLOR, linewidth=1.5, alpha=0.8)
        ax7.set_ylabel('Z Error [m]', fontsize=PLOT_FONT_SIZE-2)
        ax7.set_title('Z Error', fontsize=PLOT_FONT_SIZE-2)
        ax7.grid(True, linestyle=':', alpha=0.7)
        ax7.set_ylim([-ERROR_YLIM_MAX, ERROR_YLIM_MAX])
        ax7.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-3)
        
        # Total Error plot
        ax8 = fig.add_subplot(len(categories), 8, cat_idx*8 + 8)
        total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
        ax8.plot(t, total_err_ekf_cf, color=EKF_CF_COLOR, linewidth=2.0)
        ax8.set_ylabel('Total Error [m]', fontsize=PLOT_FONT_SIZE-2)
        ax8.set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE-2)
        ax8.set_title('Total Error', fontsize=PLOT_FONT_SIZE-2)
        ax8.grid(True, linestyle=':', alpha=0.7)
        ax8.set_ylim([0, ERROR_YLIM_MAX])
        ax8.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-3)
    
    plt.tight_layout()
    summary_path = out_dir / "category_summary_comprehensive.pdf"
    plt.savefig(summary_path, bbox_inches='tight')
    
    # Also save as PNG for easy viewing
    summary_png_path = out_dir / "category_summary_comprehensive.png"
    plt.savefig(summary_png_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    return summary_path

# ------------------------------------------------------------------------- #
def create_summary_figure(table, results_df, out_dir):
    """Create summary figure with best scores per category."""
    # Convert string errors to NaN
    results_df['RMSE_ekf_cf'] = pd.to_numeric(results_df['RMSE_ekf_cf'], errors='coerce')
    
    # Merge with metadata table
    metadata_dict = {}
    for _, row in table.iterrows():
        for bag in row['rosbags']:
            bag_name = Path(bag).name
            metadata_dict[bag_name] = {'category': row['category'], 'label': row['label']}
    
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')
    metadata_df.index.name = 'bag'
    metadata_df.reset_index(inplace=True)
    
    combined_df = results_df.merge(metadata_df, on='bag', how='left')
    
    # Group by category and find best performers
    best_performers = []
    for cat, group in combined_df.groupby('category'):
        # Skip categories with no valid data
        if pd.isna(cat):
            continue
            
        # Filter out rows with NaN values for RMSE_ekf_cf
        valid_rmse_group = group.dropna(subset=['RMSE_ekf_cf'])
        if len(valid_rmse_group) > 0:
            # Best RMSE for EKF_CF estimator
            best_rmse_idx = valid_rmse_group['RMSE_ekf_cf'].idxmin()
            best_rmse_ekf_cf = valid_rmse_group.loc[best_rmse_idx]
            best_performers.append({
                'category': cat,
                'label': best_rmse_ekf_cf['label'],
                'bag': best_rmse_ekf_cf['bag'],
                'metric': 'RMSE',
                'estimator': 'EKF_CF',
                'value': best_rmse_ekf_cf['RMSE_ekf_cf']
            })
    
    best_df = pd.DataFrame(best_performers)
    
    # Check if we have any valid data
    if len(best_df) == 0:
        print("No valid data found for summary figure")
        return None
    
    # Create the summary figure
    fig, ax = plt.figure(figsize=(12, 8), dpi=DPI), plt.subplot(111)
    
    categories = best_df['category'].unique()
    
    x = np.arange(len(categories))
    width = 0.6
    
    # Only RMSE now
    data = best_df[best_df['metric'] == 'RMSE']
    values = [data[data['category'] == cat]['value'].values[0] if len(data[data['category'] == cat]) > 0 else np.nan 
             for cat in categories]
    
    bars = ax.bar(x, values, width, label='RMSE - EKF_CF', 
                color=EKF_CF_COLOR, alpha=0.8)
    
    # Add value labels inside bars
    for bar_idx, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f"{height:.3f}", ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    ax.set_ylabel('RMSE [m]', fontsize=PLOT_FONT_SIZE+1)
    ax.set_title('Migliori risultati RMSE per categoria (EKF_CF vs Vicon)', 
                 fontsize=PLOT_FONT_SIZE+2, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE-1)
    
    plt.tight_layout()
    summary_path = out_dir / "best_performers_summary.pdf"
    plt.savefig(summary_path, bbox_inches='tight')
    
    # Also save as PNG for easy viewing
    summary_png_path = out_dir / "best_performers_summary.png"
    plt.savefig(summary_png_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    # Save the best performers to CSV as well
    best_df.to_csv(out_dir / "best_performers.csv", index=False)
    
    return summary_path

# ------------------------------------------------------------------------- #
def create_rmse_table(table, results_df, out_dir):
    """Create a table similar to the original but with RMSE values instead of rosbag names."""
    # Convert string errors to NaN
    results_df['RMSE_ekf_cf'] = pd.to_numeric(results_df['RMSE_ekf_cf'], errors='coerce')
    
    # Create mapping from rosbag name to RMSE (convert to cm and mark failures)
    rmse_map = {}
    for _, row in results_df.iterrows():
        bag_name = row['bag']
        rmse_value = row['RMSE_ekf_cf']
        if not pd.isna(rmse_value):
            # Convert to cm
            rmse_cm = rmse_value * 100
            # Mark as FAIL if > 50 cm
            if rmse_cm > 50:
                rmse_formatted = 'FAIL'
            else:
                rmse_formatted = f"{rmse_cm:.2f}"
        else:
            rmse_formatted = 'ERROR'
            
        # Remove _processed suffix for matching
        original_name = bag_name.replace('_processed', '')
        rmse_map[original_name] = rmse_formatted
        # Also map with _processed suffix
        rmse_map[bag_name] = rmse_formatted
    
    print(f"DEBUG: Created RMSE map with {len(rmse_map)} entries")
    print(f"DEBUG: First 5 entries: {dict(list(rmse_map.items())[:5])}")
    
    # Create new table with RMSE values
    rmse_table_data = []
    
    for _, row in table.iterrows():
        category = row['category']
        label = row['label']
        rosbags_list = row['rosbags']
        
        # Process each rosbag in the list
        rmse_values = []
        numeric_values = []  # For calculating mean
        
        for bag_name in rosbags_list:
            if ',' in bag_name:  # Handle multiple bags in one cell
                bag_parts = [b.strip() for b in bag_name.split(',')]
                rmse_parts = []
                for part in bag_parts:
                    if part in rmse_map:
                        rmse_val = rmse_map[part]
                        rmse_parts.append(str(rmse_val))
                        # Add to numeric values for mean calculation if it's a valid number
                        if rmse_val not in ['FAIL', 'ERROR', 'N/A']:
                            try:
                                numeric_values.append(float(rmse_val))
                            except:
                                pass
                    else:
                        print(f"DEBUG: No match for '{part}'")
                        rmse_parts.append('N/A')
                rmse_values.append(', '.join(rmse_parts))
            else:
                if bag_name in rmse_map:
                    rmse_val = rmse_map[bag_name]
                    rmse_values.append(str(rmse_val))
                    # Add to numeric values for mean calculation if it's a valid number
                    if rmse_val not in ['FAIL', 'ERROR', 'N/A']:
                        try:
                            numeric_values.append(float(rmse_val))
                        except:
                            pass
                else:
                    print(f"DEBUG: No match for '{bag_name}'")
                    rmse_values.append('N/A')
        
        # Calculate mean (excluding FAIL, ERROR, N/A)
        if numeric_values:
            mean_rmse = f"{np.mean(numeric_values):.2f}"
        else:
            mean_rmse = 'N/A'
        
        # Create row data
        row_data = {
            'Category': category,
            'Label': label
        }
        
        # Add test columns (up to 10 tests)
        for i in range(10):
            if i < len(rmse_values):
                row_data[f'Test_{i+1}'] = rmse_values[i]
            else:
                row_data[f'Test_{i+1}'] = ''
        
        # Add mean column
        row_data['Mean_RMSE_cm'] = mean_rmse
        
        rmse_table_data.append(row_data)
    
    # Save as CSV
    rmse_csv_path = out_dir / 'rmse_table.csv'
    rmse_df = pd.DataFrame(rmse_table_data)
    rmse_df.to_csv(rmse_csv_path, index=False)
    
    # Save as Markdown table
    md_path = out_dir / 'rmse_table.md'
    with open(md_path, 'w') as f:
        f.write("# RMSE Results Table\n\n")
        f.write("**Note:** Values are in centimeters (cm). FAIL indicates RMSE > 50 cm.\n\n")
        f.write("| Category | Label |")
        
        for i in range(10):
            f.write(f" Test {i+1} |")
        f.write(" Mean RMSE |\n")
        
        # Header separator
        f.write("|----------|-------|")
        for i in range(10):
            f.write("--------|")
        f.write("----------|\n")
        
        # Data rows
        for row_data in rmse_table_data:
            f.write(f"| {row_data['Category']} | {row_data['Label']} |")
            for i in range(10):
                f.write(f" {row_data[f'Test_{i+1}']} |")
            f.write(f" {row_data['Mean_RMSE_cm']} |\n")
    
    return rmse_csv_path, md_path

# ------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Batch ROS2 bag analyser')
    parser.add_argument('--excel', required=True, help='Excel list of tests')
    parser.add_argument('--root',  required=True, help='Root folder containing rosbag dirs')
    parser.add_argument('--out',   required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count()-1))
    args = parser.parse_args()

    excel_path = pathlib.Path(args.excel).expanduser()
    root_dir   = pathlib.Path(args.root).expanduser()
    out_dir    = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    table = read_excel_list(excel_path)

    bag_dirs = []
    for bags in table['rosbags']:
        for b in bags:
            bag_path = root_dir / b
            if bag_path.exists():
                bag_dirs.append(bag_path)
            else:
                print(f'WARNING: {bag_path} not found')

    print(f"Found {len(bag_dirs)} rosbags, analysing with {args.workers} workers …")

    with mp.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(partial(worker, out_root=out_dir), bag_dirs),
                            total=len(bag_dirs)))

    # Global summary
    summary = pd.DataFrame(results,
                           columns=['bag','RMSE_ekf_cf','csv','plot','bag_dir'])
    summary.to_csv(out_dir / 'summary.csv', index=False)
    
    # Create comprehensive category summary figure
    category_summary_path = create_category_summary_figure(table, summary, out_dir)
    if category_summary_path:
        print(f"Category summary figure saved to {category_summary_path}")
    
    # Create summary figure with best performers per category
    summary_fig_path = create_summary_figure(table, summary, out_dir)
    if summary_fig_path:
        print(f"Summary figure saved to {summary_fig_path}")
    else:
        print("No summary figure created due to lack of valid data")
    
    # Create RMSE table
    rmse_table_path, rmse_md_path = create_rmse_table(table, summary, out_dir)
    print(f"RMSE table saved to {rmse_table_path} and {rmse_md_path}")
    
    print("Results saved to", out_dir)
    print("\nGenerated files:")
    print("- Individual rosbag plots (PDF + PNG)")
    print("- Category summary comprehensive figure")
    print("- Best performers summary figure") 
    print("- RMSE table (CSV + Markdown)")

if __name__ == '__main__':
    main()
