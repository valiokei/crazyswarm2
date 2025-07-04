#!/usr/bin/env python3
"""
analyze_rosbags_v2.py
------------------
Batch‑analyse ROS 2 Jazzy rosbags to compute the RMSE between:
  • Vicon reference pose  :  child = vicon/magnetic_drone/magnetic_drone, parent = vicon/world
  • EKF_CF estimator       :  child = EKF_CF,                    parent = vicon/world

The script:
  1. Reads an Excel metadata file that lists all tests and associated rosbag folders.
  2. Iterates (multiprocess) over each bag and extracts the `/tf` and `/cf21/VoltagesCalibrated` topics.
  3. Synchronises the two pose streams (nearest‑neighbour within --sync-epsilon s).
  4. Computes RMSE & MAE per axis and overall, stores them in CSV.
  5. Generates publication‑ready plots (IEEE TRO compliant) in PNG & PDF.
  6. Creates 3D trajectory plots with anchor positions.
  7. Visualizes pointwise errors along the trajectory.
  8. Creates voltage plots for all 4 anchors over time.
  9. Computes standard deviation metrics.
  10. Generates a summary figure with best scores per category including voltage plots.

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
VOLTAGE_TOPIC       = '/cf21/VoltagesCalibrated'
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

# Plot axis limits for uniformity
ERROR_PLOT_Y_LIM    = (-0.7, 0.7)    # Y-axis limits for error plots per axis
ERROR_TOTAL_Y_LIM   = (0.0, 1.0)     # Y-axis limits for total error plot
TRAJ_3D_X_LIM       = (-3.0, 3.0)    # X-axis limits for 3D trajectory plots
TRAJ_3D_Y_LIM       = (-3.0, 3.0)    # Y-axis limits for 3D trajectory plots
TRAJ_3D_Z_LIM       = (0.0, 2.5)     # Z-axis limits for 3D trajectory plots
# ------------------------------------------------------------------------- #

def read_excel_list(excel_path: str):
    """Return dataframe with columns category, label, rosbags(list)."""
    df = pd.read_excel(excel_path, sheet_name=0)
    header_row = df.iloc[0, 2:].reset_index(drop=True)
    data       = df.iloc[1:].reset_index(drop=True)

    categories = data.iloc[:, 0].fillna(method='ffill')
    labels     = data.iloc[:, 1]
    bag_cols   = data.iloc[:, 2:]

    bag_lists  = []
    for _, row in bag_cols.iterrows():
        lst = []
        for cell in row:
            if isinstance(cell, str) and 'rosbag2' in cell:
                # Handle multiple rosbags separated by commas in the same cell
                bag_names = [name.strip() for name in cell.split(',')]
                for name in bag_names:
                    if 'rosbag2' in name:
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

def rosbag_iter_topics(bag_path: pathlib.Path, topics):
    """Yield (topic, time_sec, msg) for specified topics."""
    reader = SequentialReader()
    
    sid = detect_storage_id(bag_path)
    
    storage_options   = StorageOptions(uri=str(bag_path), storage_id=sid)
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    available_topics = {t.name: t.type for t in topic_types}
    
    # Check which topics are available
    topics_to_read = []
    for topic in topics:
        if topic in available_topics:
            topics_to_read.append(topic)
        else:
            print(f"Warning: Topic {topic} not found in bag {bag_path}")
    
    if not topics_to_read:
        return

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic not in topics_to_read:
            continue
        
        msg_type = get_message(available_topics[topic])
        msg = deserialize_message(data, msg_type)
        stamp_sec = t / 1e9
        yield topic, stamp_sec, msg

# ------------------------------------------------------------------------- #
def rosbag_iter_tf(bag_path: pathlib.Path):
    """Yield (time_sec, transform_msg) only for TF_TOPIC."""
    for topic, stamp_sec, msg in rosbag_iter_topics(bag_path, [TF_TOPIC]):
        if topic == TF_TOPIC:
            for tf in msg.transforms:
                yield stamp_sec, tf

# ------------------------------------------------------------------------- #
def extract_streams(bag_dir: pathlib.Path):
    """Return dict frame_name -> np.ndarray[[t,x,y,z]], anchors dict, and voltage data."""
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
    
    # Extract voltage data
    voltage_data = extract_voltage_data(bag_dir)
    
    return streams, anchors, voltage_data

# ------------------------------------------------------------------------- #
def apply_static_transform(streams, anchors, voltage_data):
    """Non applica più trasformazioni statiche dato che tutti i frame sono già in vicon/world."""
    # print("INFO: Trasformazione statica disabilitata - tutti i frame sono già in vicon/world")
    
    # # Debug info per verificare i dati
    # if streams['ref'].size > 0:
    #     print(f"DEBUG: Ref (vicon/world) - primo punto: {streams['ref'][0, 1:4]}")
    # if streams['ekf_cf'].size > 0:
    #     print(f"DEBUG: EKF_CF (vicon/world) - primo punto: {streams['ekf_cf'][0, 1:4]}")
    # if streams['dog'].size > 0:
    #     print(f"DEBUG: Dog (vicon/world) - primo punto: {streams['dog'][0, 1:4]}")
    
    return streams, anchors, voltage_data

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
def compute_stddev(err):
    """Compute standard deviation of the error."""
    stddev_axis = np.std(err, axis=0)
    stddev_total = np.std(np.sqrt(np.sum(err**2, axis=1)))
    return stddev_axis, stddev_total

# ------------------------------------------------------------------------- #
def plot_3d_trajectory(t, ref, ekf_cf, anchors, output_path_3d, output_path_2d):
    """Create 3D trajectory plot in vicon/world frame and 2D XY trajectory plot."""
    # 3D World frame plot
    fig = plt.figure(figsize=(6, 6), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'b-', linewidth=2, label='Vicon')
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], 'r--', linewidth=1.5, label='EKF_CF')
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            # Use the most recent position (or the median for stability)
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE)
    
    # Set uniform axis limits for all 3D plots
    ax.set_xlim(TRAJ_3D_X_LIM)
    ax.set_ylim(TRAJ_3D_Y_LIM)
    ax.set_zlim(TRAJ_3D_Z_LIM)
    
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title('Trajectory in Vicon/World Frame (3D)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path_3d)
    plt.close()
    
    # 2D XY trajectory plot
    fig = plt.figure(figsize=(6, 6), dpi=DPI)
    ax = fig.add_subplot(111)
    
    ax.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=2, label='Vicon')
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], 'r--', linewidth=1.5, label='EKF_CF')
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    # Mark start and end points
    if len(ref) > 0:
        ax.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=80, 
                   label='Start', edgecolors='k', zorder=10)
        ax.scatter(ref[-1, 0], ref[-1, 1], color='red', marker='s', s=80, 
                   label='End', edgecolors='k', zorder=10)
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title('Trajectory in Vicon/World Frame (XY)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path_2d)
    plt.close()

# ------------------------------------------------------------------------- #
def plot_errors(t, err_ekf_cf, output_path):
    """Create plots of pointwise errors along the trajectory."""
    fig, axes = plt.subplots(4, 1, figsize=(6.3, 8), dpi=DPI, sharex=True)
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z']
    
    # Plot errors per axis with uniform Y limits
    for i in range(3):
        axes[i].plot(t, err_ekf_cf[:, i], 'r-', linewidth=1, alpha=0.8, label='EKF_CF Error')
        axes[i].set_ylabel(f'{labels[i]} Error [m]', fontsize=PLOT_FONT_SIZE)
        axes[i].set_ylim(ERROR_PLOT_Y_LIM)  # Set uniform Y-axis limits (-0.7, 0.7)
        axes[i].grid(True, linestyle=':')
        if i == 0:
            axes[i].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    
    # Plot total error with uniform Y limit
    total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
    axes[3].plot(t, total_err_ekf_cf, 'r-', linewidth=1.5, label='EKF_CF Total Error')
    axes[3].set_ylabel('Total Error [m]', fontsize=PLOT_FONT_SIZE)
    axes[3].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    axes[3].set_ylim(ERROR_TOTAL_Y_LIM)  # Set uniform Y-axis limit for total error
    axes[3].grid(True, linestyle=':')
    axes[3].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------------------------------------------------- #
def plot_voltage_data(voltage_data, output_path):
    """Create voltage plot over time for all 4 anchors."""
    if voltage_data.size == 0:
        print("Warning: No voltage data available for plotting")
        return
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=DPI)
    
    times = voltage_data[:, 0]
    
    # Normalize time to start from 0
    times = times - times[0] if len(times) > 0 else times
    
    # Colors for the 4 anchors matching ANCHOR_COLORS
    colors = ['black', 'gold', 'gray', 'red']  # Nero, Giallo, Grigio, Rosso
    anchor_names = ['Nero', 'Giallo', 'Grigio', 'Rosso']
    
    # Plot each anchor's voltage
    for i in range(4):
        if voltage_data.shape[1] > i + 1:  # Check if we have voltage data for this anchor
            voltages = voltage_data[:, i + 1]  # i+1 because column 0 is time
            ax.plot(times, voltages, color=colors[i], linewidth=2, 
                   label=f'Anchor {anchor_names[i]}', alpha=0.8, marker='o', markersize=2)
    
    ax.set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Voltage [V]', fontsize=PLOT_FONT_SIZE)
    ax.set_title('Anchor Voltages Over Time', fontsize=PLOT_FONT_SIZE)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=PLOT_FONT_SIZE-1, loc='best')
    
    # Add statistics as text for each anchor
    if voltage_data.shape[1] > 1:
        stats_text = "Voltage Statistics:\n"
        for i in range(4):
            if voltage_data.shape[1] > i + 1:
                voltages = voltage_data[:, i + 1]
                if len(voltages) > 0:
                    mean_v = np.mean(voltages)
                    std_v = np.std(voltages)
                    min_v = np.min(voltages)
                    max_v = np.max(voltages)
                    stats_text += f"{anchor_names[i]}: {mean_v:.3f}V (±{std_v:.3f})\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------------------------------------------------- #
def extract_voltage_data(bag_dir: pathlib.Path):
    """Extract voltage data from the VOLTAGE_TOPIC with 4 anchor values."""
    voltage_data = []
    
    for topic, stamp_sec, msg in rosbag_iter_topics(bag_dir, [VOLTAGE_TOPIC]):
        if topic == VOLTAGE_TOPIC:
            try:
                voltages = None
                
                # Try to extract the 4 voltage values for the anchors
                if hasattr(msg, 'data') and hasattr(msg.data, '__len__') and len(msg.data) == 4:
                    # If it's an array with 4 values
                    voltages = list(msg.data)
                elif hasattr(msg, 'voltages') and hasattr(msg.voltages, '__len__') and len(msg.voltages) == 4:
                    # If it's explicitly named 'voltages'
                    voltages = list(msg.voltages)
                elif hasattr(msg, 'voltage') and hasattr(msg.voltage, '__len__') and len(msg.voltage) == 4:
                    # If it's named 'voltage' but is an array
                    voltages = list(msg.voltage)
                else:
                    # Try to find any array field with 4 elements
                    for field_name in dir(msg):
                        if not field_name.startswith('_'):
                            field_value = getattr(msg, field_name)
                            if hasattr(field_value, '__len__') and len(field_value) == 4:
                                try:
                                    # Check if all elements are numeric
                                    voltages = [float(v) for v in field_value]
                                    break
                                except (ValueError, TypeError):
                                    continue
                
                if voltages is not None and len(voltages) == 4:
                    # Store timestamp and 4 voltage values: [time, v1, v2, v3, v4]
                    voltage_data.append([stamp_sec] + voltages)
                
            except Exception as e:
                print(f"Warning: Could not extract voltage data from message: {e}")
                continue
    
    if voltage_data:
        return np.array(voltage_data)  # Shape: (n_samples, 5) = [time, v1, v2, v3, v4]
    else:
        return np.empty((0, 5))  # 5 columns: time + 4 voltages

# ------------------------------------------------------------------------- #
def analyse_bag(bag_dir, out_dir):
    bag_dir = pathlib.Path(bag_dir)
    
    # Create subfolder for this rosbag
    bag_out_dir = out_dir / bag_dir.name
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    
    streams, anchors, voltage_data = extract_streams(bag_dir)
    
    # Apply static transform to align frames correctly
    streams, anchors, voltage_data = apply_static_transform(streams, anchors, voltage_data)
    
    t, ref, ekf_cf = align_streams(streams)
    
    # Error calculations
    rmse_ekf_cf_axis, rmse_ekf_cf_tot, err_ekf_cf = compute_rmse(ref, ekf_cf)
    
    # Standard deviation calculations
    stddev_ekf_cf_axis, stddev_ekf_cf_tot = compute_stddev(err_ekf_cf)

    # Save CSV summary with added stddev metrics
    csv_path = bag_out_dir / f"{bag_dir.name}_metrics.csv"
    pd.DataFrame({
        'metric' : ['RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_TOT', 
                   'StdDev_X', 'StdDev_Y', 'StdDev_Z', 'StdDev_TOT'],
        'ekf_cf': np.concatenate([rmse_ekf_cf_axis, [rmse_ekf_cf_tot], 
                                 stddev_ekf_cf_axis, [stddev_ekf_cf_tot]])
    }).to_csv(csv_path, index=False)

    # Create traditional plots (as before)
    plot_path = bag_out_dir / f"{bag_dir.name}_traj.pdf"
    plt.figure(figsize=(6.3, 2.8*3), dpi=DPI)
    axx = [plt.subplot(3,1,i+1) for i in range(3)]
    labels = ['x','y','z']
    for i, ax in enumerate(axx):
        ax.plot(t, ref[:,i], label='Vicon', linewidth=1.0)
        ax.plot(t, ekf_cf[:,i], label='EKF_CF', linewidth=0.8, linestyle='--')
        ax.set_ylabel(labels[i]+' [m]', fontsize=PLOT_FONT_SIZE)
        ax.grid(True, which='both', linestyle=':')
        if i==0:
            ax.legend(fontsize=PLOT_FONT_SIZE, ncol=2, loc='upper center')
    axx[-1].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Create 3D and 2D trajectory plots
    traj3d_path = bag_out_dir / f"{bag_dir.name}_traj3d.pdf"
    traj2d_path = bag_out_dir / f"{bag_dir.name}_traj2d.pdf"
    plot_3d_trajectory(t, ref, ekf_cf, anchors, traj3d_path, traj2d_path)
    
    # Create error plots
    error_path = bag_out_dir / f"{bag_dir.name}_errors.pdf"
    plot_errors(t, err_ekf_cf, error_path)
    
    # Create voltage plot
    voltage_path = bag_out_dir / f"{bag_dir.name}_voltage.pdf"
    plot_voltage_data(voltage_data, voltage_path)
    
    return csv_path, plot_path, rmse_ekf_cf_tot, stddev_ekf_cf_tot, bag_out_dir

def worker(bag_dir, out_root):
    try:
        csv, plot, r_ekf_cf, std_ekf_cf, bag_out_dir = analyse_bag(bag_dir, out_root)
        return (bag_dir.name, r_ekf_cf, std_ekf_cf, str(csv), str(plot), str(bag_out_dir))
    except Exception as e:
        return (bag_dir.name, 'ERROR', 'ERROR', str(e), '', '')

# ------------------------------------------------------------------------- #
def create_category_summary_figure(table, results_df, out_dir):
    """Create comprehensive summary figure with 3D trajectories and error plots per category."""
    # Convert string errors to NaN
    for col in ['RMSE_ekf_cf', 'StdDev_ekf_cf']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
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
    combined_df = combined_df.dropna(subset=['RMSE_ekf_cf'])  # Remove error entries
    
    categories = combined_df['category'].unique()
    categories = [cat for cat in categories if pd.notna(cat)]
    
    if len(categories) == 0:
        print("No valid categories found for summary figure")
        return None
    
    # Create figure with subplots: 6 columns (3D traj, XY traj, XZ traj, error total, error xyz, voltage) x n_categories rows
    fig = plt.figure(figsize=(36, 6*len(categories)), dpi=DPI)
    
    for cat_idx, category in enumerate(categories):
        cat_data = combined_df[combined_df['category'] == category]
        
        if len(cat_data) == 0:
            continue
            
        # Find best performer for this category (lowest RMSE total)
        best_idx = cat_data['RMSE_ekf_cf'].idxmin()
        best_bag = cat_data.loc[best_idx]
        best_bag_name = best_bag['bag']
        best_bag_dir = out_dir / best_bag_name
        
        # Get RMSE value for the title
        best_rmse_cm = best_bag['RMSE_ekf_cf'] * 100  # Convert to cm
        
        # Load data for best bag to create plots
        try:
            # Re-analyze the best bag to get trajectory data and voltage data
            streams, anchors, voltage_data = extract_streams(best_bag_dir.parent.parent / best_bag_name)
            streams, anchors, voltage_data = apply_static_transform(streams, anchors, voltage_data)
            t, ref, ekf_cf = align_streams(streams)
            _, _, err_ekf_cf = compute_rmse(ref, ekf_cf)
        except Exception as e:
            print(f"Error loading data for {best_bag_name}: {e}")
            continue
        
        # 3D Trajectory plot with RMSE in title
        ax1 = fig.add_subplot(len(categories), 6, cat_idx*6 + 1, projection='3d')
        ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'b-', linewidth=2, label='Vicon', alpha=0.8)
        ax1.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], 'r--', linewidth=1.5, label='EKF_CF', alpha=0.8)
        
        # Plot anchors if available
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4]
                ax1.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=60, label=name, edgecolors='k', alpha=0.8)
        
        ax1.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-1)
        ax1.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE-1)
        ax1.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE-1)
        
        # Set uniform axis limits for all 3D plots
        ax1.set_xlim(TRAJ_3D_X_LIM)
        ax1.set_ylim(TRAJ_3D_Y_LIM)
        ax1.set_zlim(TRAJ_3D_Z_LIM)
        
        ax1.set_title(f'{category} - 3D\nRMSE: {best_rmse_cm:.2f} cm\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        if cat_idx == 0:
            ax1.legend(fontsize=PLOT_FONT_SIZE-2, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # XY Trajectory plot
        ax2 = fig.add_subplot(len(categories), 6, cat_idx*6 + 2)
        ax2.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=2, label='Vicon', alpha=0.8)
        ax2.plot(ekf_cf[:, 0], ekf_cf[:, 1], 'r--', linewidth=1.5, label='EKF_CF', alpha=0.8)
        
        # Plot anchors if available (XY projection)
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4]
                ax2.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=60, label=name, edgecolors='k', alpha=0.8)
        
        ax2.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-1)
        ax2.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE-1)
        ax2.set_title(f'{category} - XY\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        ax2.grid(True, alpha=0.7)
        ax2.set_aspect('equal', adjustable='box')
        
        # XZ Trajectory plot
        ax3 = fig.add_subplot(len(categories), 6, cat_idx*6 + 3)
        ax3.plot(ref[:, 0], ref[:, 2], 'b-', linewidth=2, label='Vicon', alpha=0.8)
        ax3.plot(ekf_cf[:, 0], ekf_cf[:, 2], 'r--', linewidth=1.5, label='EKF_CF', alpha=0.8)
        
        # Plot anchors if available (XZ projection)
        for i, name in enumerate(ANCHOR_FRAMES):
            if anchors[name].size > 0:
                pos = anchors[name][-1, 1:4]
                ax3.scatter(pos[0], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=60, label=name, edgecolors='k', alpha=0.8)
        
        ax3.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-1)
        ax3.set_ylabel('Z [m]', fontsize=PLOT_FONT_SIZE-1)
        ax3.set_title(f'{category} - XZ\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        ax3.grid(True, alpha=0.7)
        
        # Total Error plot
        ax4 = fig.add_subplot(len(categories), 6, cat_idx*6 + 4)
        total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
        ax4.plot(t, total_err_ekf_cf, 'r-', linewidth=1.5, label='EKF_CF Total Error', alpha=0.8)
        ax4.set_ylabel('Errore totale [m]', fontsize=PLOT_FONT_SIZE-1)
        ax4.set_xlabel('Tempo [s]', fontsize=PLOT_FONT_SIZE-1)
        ax4.set_ylim(ERROR_TOTAL_Y_LIM)  # Set uniform Y-axis limit for total error
        ax4.set_title(f'{category} - Errore Totale\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        ax4.grid(True, linestyle=':', alpha=0.7)
        if cat_idx == 0:
            ax4.legend(fontsize=PLOT_FONT_SIZE-2)
        
        # XYZ Error components plot
        ax5 = fig.add_subplot(len(categories), 6, cat_idx*6 + 5)
        labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        for i in range(3):
            ax5.plot(t, err_ekf_cf[:, i], color=colors[i], linewidth=1.2, 
                    label=f'Errore {labels[i]}', alpha=0.8)
        ax5.set_ylabel('Errore [m]', fontsize=PLOT_FONT_SIZE-1)
        ax5.set_xlabel('Tempo [s]', fontsize=PLOT_FONT_SIZE-1)
        ax5.set_ylim(ERROR_PLOT_Y_LIM)  # Set uniform Y-axis limits (-0.7, 0.7)
        ax5.set_title(f'{category} - Errori XYZ\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        ax5.grid(True, linestyle=':', alpha=0.7)
        if cat_idx == 0:
            ax5.legend(fontsize=PLOT_FONT_SIZE-2)
        
        # Voltage plot
        ax6 = fig.add_subplot(len(categories), 6, cat_idx*6 + 6)
        if voltage_data.size > 0:
            v_times = voltage_data[:, 0]
            # Normalize time to start from 0
            v_times = v_times - v_times[0] if len(v_times) > 0 else v_times
            
            # Colors for the 4 anchors
            colors_v = ['black', 'gold', 'gray', 'red']
            anchor_names = ['Nero', 'Giallo', 'Grigio', 'Rosso']
            
            # Plot each anchor's voltage
            for i in range(4):
                if voltage_data.shape[1] > i + 1:
                    v_voltages = voltage_data[:, i + 1]
                    ax6.plot(v_times, v_voltages, color=colors_v[i], linewidth=1.2, 
                            label=anchor_names[i], alpha=0.8)
            
            # Show mean voltage for all anchors
            if voltage_data.shape[1] > 1:
                all_voltages = voltage_data[:, 1:].flatten()
                mean_v = np.mean(all_voltages)
                ax6.text(0.02, 0.98, f'Media: {mean_v:.3f}V', transform=ax6.transAxes, 
                        fontsize=7, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, 'Dati tensione\nnon disponibili', transform=ax6.transAxes, 
                    fontsize=10, ha='center', va='center')
        
        ax6.set_ylabel('Tensione [V]', fontsize=PLOT_FONT_SIZE-1)
        ax6.set_xlabel('Tempo [s]', fontsize=PLOT_FONT_SIZE-1)
        ax6.set_title(f'{category} - Tensioni\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
        ax6.grid(True, linestyle=':', alpha=0.7)
        if cat_idx == 0 and voltage_data.size > 0:
            ax6.legend(fontsize=PLOT_FONT_SIZE-2, loc='lower right')
    
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
    for col in ['RMSE_ekf_cf', 'StdDev_ekf_cf']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
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
        # Best RMSE for EKF_CF estimator
        best_rmse_ekf_cf = group.loc[group['RMSE_ekf_cf'].idxmin()]
        best_performers.append({
            'category': cat,
            'label': best_rmse_ekf_cf['label'],
            'bag': best_rmse_ekf_cf['bag'],
            'metric': 'RMSE',
            'estimator': 'EKF_CF',
            'value': best_rmse_ekf_cf['RMSE_ekf_cf']
        })
        
        # Best StdDev for EKF_CF
        best_std_ekf_cf = group.loc[group['StdDev_ekf_cf'].idxmin()]
        best_performers.append({
            'category': cat,
            'label': best_std_ekf_cf['label'],
            'bag': best_std_ekf_cf['bag'],
            'metric': 'StdDev',
            'estimator': 'EKF_CF',
            'value': best_std_ekf_cf['StdDev_ekf_cf']
        })
    
    best_df = pd.DataFrame(best_performers)
    
    # Create the summary figure
    fig, ax = plt.figure(figsize=(10, 8), dpi=DPI), plt.subplot(111)
    
    categories = best_df['category'].unique()
    metrics = ['RMSE', 'StdDev']
    estimators = ['EKF_CF']
    
    x = np.arange(len(categories))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        data = best_df[best_df['metric'] == metric]
        values = [data[data['category'] == cat]['value'].values[0] if len(data[data['category'] == cat]) > 0 else np.nan 
                 for cat in categories]
        
        bars = ax.bar(x + (i-0.5)*width, values, width, 
                    label=f"{metric} - EKF_CF", 
                    color=f"C{i}", alpha=0.8)
        
        # Add value labels on top of bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f"{height:.3f}", ha='center', va='bottom',
                       fontsize=8, rotation=90)
    
    ax.set_ylabel('Valore metrico [m]', fontsize=PLOT_FONT_SIZE+1)
    ax.set_title('Migliori risultati per categoria (EKF_CF vs Vicon)', fontsize=PLOT_FONT_SIZE+2)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    summary_path = out_dir / "best_performers_summary.pdf"
    plt.savefig(summary_path)
    
    # Also save as PNG for easy viewing
    summary_png_path = out_dir / "best_performers_summary.png"
    plt.savefig(summary_png_path, dpi=150)
    
    plt.close()
    
    # Save the best performers to CSV as well
    best_df.to_csv(out_dir / "best_performers.csv", index=False)
    
    return summary_path

# ------------------------------------------------------------------------- #
def create_rmse_markdown_table(table, results_df, out_dir):
    """Create a Markdown table with RMSE values for each rosbag, structured like the original Excel table."""
    
    # Convert RMSE values to numeric, handle errors
    results_df['RMSE_ekf_cf'] = pd.to_numeric(results_df['RMSE_ekf_cf'], errors='coerce')
    
    # Create a mapping from bag name to RMSE value
    rmse_dict = {}
    for _, row in results_df.iterrows():
        bag_name = row['bag']
        rmse_val = row['RMSE_ekf_cf']
        if pd.isna(rmse_val):
            rmse_dict[bag_name] = 'FAIL'
        else:
            # Convert to cm and check if > 50 cm
            rmse_cm = rmse_val * 100  # Convert from meters to cm
            if rmse_cm > 50:
                rmse_dict[bag_name] = f'{rmse_cm:.2f} FAIL'
            else:
                rmse_dict[bag_name] = f'{rmse_cm:.2f}'
    
    # Start building the markdown table
    md_lines = []
    md_lines.append("# RMSE Results Table")
    md_lines.append("")
    md_lines.append("**Note:** Values are in centimeters (cm). FAIL indicates RMSE > 50 cm.")
    md_lines.append("")
    
    # Create header row
    header = "| Category | Label |"
    separator = "|----------|-------|"
    
    # Find the maximum number of test columns by looking at all rosbag lists
    max_tests = 0
    for _, row in table.iterrows():
        if len(row['rosbags']) > max_tests:
            max_tests = len(row['rosbags'])
    
    # Add test columns to header
    for i in range(1, max_tests + 1):
        header += f" Test {i} |"
        separator += "-------|"
    
    # Add Mean RMSE column
    header += " Mean RMSE |"
    separator += "-----------|"
    
    md_lines.append(header)
    md_lines.append(separator)
    
    # Process each row in the table
    for _, row in table.iterrows():
        category = row['category']
        label = row['label']
        rosbags = row['rosbags']
        
        # Start building the row
        md_row = f"| {category} | {label} |"
        
        # Collect RMSE values for this row
        row_rmse_values = []
        
        # Add test results
        for i in range(max_tests):
            if i < len(rosbags):
                bag_name = rosbags[i]
                # Remove '_processed' suffix if present for lookup
                lookup_name = bag_name.replace('_processed', '')
                
                # Try both with and without _processed suffix
                rmse_val = rmse_dict.get(bag_name, rmse_dict.get(lookup_name, 'N/A'))
                md_row += f" {rmse_val} |"
                
                # Collect numeric values for mean calculation
                if rmse_val != 'N/A' and 'FAIL' not in rmse_val:
                    try:
                        row_rmse_values.append(float(rmse_val))
                    except ValueError:
                        pass
            else:
                md_row += " |"
        
        # Calculate mean RMSE for this row
        if row_rmse_values:
            mean_rmse = np.mean(row_rmse_values)
            md_row += f" {mean_rmse:.2f} |"
        else:
            md_row += " N/A |"
        
        md_lines.append(md_row)
    
    # Save the markdown file
    md_file_path = out_dir / "rmse_results_table.md"
    with open(md_file_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"RMSE Markdown table saved to {md_file_path}")
    return md_file_path

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
                           columns=['bag','RMSE_ekf_cf','StdDev_ekf_cf','csv','plot','bag_dir'])
    summary.to_csv(out_dir / 'summary.csv', index=False)
    
    # Create RMSE Markdown table
    rmse_md_path = create_rmse_markdown_table(table, summary, out_dir)
    print(f"RMSE Markdown table saved to {rmse_md_path}")
    
    # Create comprehensive category summary figure
    category_summary_path = create_category_summary_figure(table, summary, out_dir)
    if category_summary_path:
        print(f"Category summary figure saved to {category_summary_path}")
    
    # Create summary figure with best performers per category
    summary_fig_path = create_summary_figure(table, summary, out_dir)
    print(f"Summary figure saved to {summary_fig_path}")
    
    print("Results saved to", out_dir)

if __name__ == '__main__':
    main()
