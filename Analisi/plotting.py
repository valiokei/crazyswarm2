#!/usr/bin/env python3
"""
plotting.py
----------
Plotting functions for trajectory visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from .config import (
        PLOT_FONT_SIZE, DPI, ERROR_PLOT_Y_LIM, ERROR_TOTAL_Y_LIM,
        TRAJ_3D_X_LIM, TRAJ_3D_Y_LIM, TRAJ_3D_Z_LIM,
        DOG_TRAJ_3D_X_LIM, DOG_TRAJ_3D_Y_LIM, DOG_TRAJ_3D_Z_LIM,
        ANCHOR_FRAMES, ANCHOR_COLORS,
        ROBODOG_LENGTH, ROBODOG_WIDTH, ROBODOG_HEIGHT, ROBODOG_FRAME_HEIGHT,
        CRAZYFLIE_DIAMETER, CRAZYFLIE_HEIGHT, SNAPSHOT_POSITIONS
    )
except ImportError:
    from config import (
        PLOT_FONT_SIZE, DPI, ERROR_PLOT_Y_LIM, ERROR_TOTAL_Y_LIM,
        TRAJ_3D_X_LIM, TRAJ_3D_Y_LIM, TRAJ_3D_Z_LIM,
        DOG_TRAJ_3D_X_LIM, DOG_TRAJ_3D_Y_LIM, DOG_TRAJ_3D_Z_LIM,
        ANCHOR_FRAMES, ANCHOR_COLORS,
        ROBODOG_LENGTH, ROBODOG_WIDTH, ROBODOG_HEIGHT, ROBODOG_FRAME_HEIGHT,
        CRAZYFLIE_DIAMETER, CRAZYFLIE_HEIGHT, SNAPSHOT_POSITIONS
    )


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
    plt.show()
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


def plot_errors(t, err_ekf_cf, output_path):
    """Create plots of pointwise errors along the trajectory."""
    fig, axes = plt.subplots(4, 1, figsize=(6.3, 8), dpi=DPI, sharex=True)
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z']
    
    # Plot errors per axis with uniform Y limits
    for i in range(3):
        axes[i].plot(t, err_ekf_cf[:, i], 'r-', linewidth=1, alpha=0.8, label='EKF_CF Error')
        axes[i].set_ylabel(f'{labels[i]} Error [m]', fontsize=PLOT_FONT_SIZE)
        axes[i].set_ylim(ERROR_PLOT_Y_LIM)
        axes[i].grid(True, linestyle=':')
        if i == 0:
            axes[i].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    
    # Plot total error with uniform Y limit
    total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
    axes[3].plot(t, total_err_ekf_cf, 'r-', linewidth=1.5, label='EKF_CF Total Error')
    axes[3].set_ylabel('Total Error [m]', fontsize=PLOT_FONT_SIZE)
    axes[3].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    axes[3].set_ylim(ERROR_TOTAL_Y_LIM)
    axes[3].grid(True, linestyle=':')
    axes[3].legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_voltage_data(voltage_data, output_path):
    """Create voltage plot over time for all 4 anchors."""
    if voltage_data.size == 0:
        print("Warning: No voltage data available for plotting")
        return
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=DPI)
    
    times = voltage_data[:, 0]
    times = times - times[0] if len(times) > 0 else times
    
    # Colors for the 4 anchors matching ANCHOR_COLORS
    colors = ['black', 'gold', 'gray', 'red']
    anchor_names = ['Nero', 'Giallo', 'Grigio', 'Rosso']
    
    # Plot each anchor's voltage
    for i in range(4):
        if voltage_data.shape[1] > i + 1:
            voltages = voltage_data[:, i + 1]
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
                    stats_text += f"{anchor_names[i]}: {mean_v:.3f}V (±{std_v:.3f})\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_error_colored_trajectory(t, ref, ekf_cf, err, anchors, dog_data, output_path, show_plot=False):
    """Create a figure with EKF_CF trajectory colored by error magnitude (3D and XY views)."""
    # Calculate total error for colormap
    total_error = np.sqrt(np.sum(err**2, axis=1))
    
    fig = plt.figure(figsize=(16, 8), dpi=DPI)
    
    # 3D plot with tighter axis limits
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    
    # Plot Vicon reference trajectory in gray
    ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'gray', linewidth=3, alpha=0.7, label='Vicon (Reference)')
    
    # Plot RoboDog trajectory FIRST (so it's underneath) with fixed green color
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        ax1.plot(dog_xyz[:, 0], dog_xyz[:, 1], dog_xyz[:, 2], 'g-', linewidth=2, alpha=0.8, label='RoboDog')
    
    # Create continuous colored line for EKF_CF trajectory (3D)
    points = ekf_cf.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create 3D line collection with colors
    lc = Line3DCollection(segments, cmap='plasma', alpha=0.8, linewidths=2)
    lc.set_array(total_error[:-1])
    line = ax1.add_collection3d(lc, zs=None, zdir='z')
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax1.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=200, label=name, edgecolors='k', alpha=0.9)
    
    # Add start and end points to 3D plot
    if len(ref) > 0:
        ax1.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color='green', marker='o', s=120, 
                   label='Start', edgecolors='k', zorder=10)
        ax1.scatter(ref[-1, 0], ref[-1, 1], ref[-1, 2], color='red', marker='s', s=120, 
                   label='End', edgecolors='k', zorder=10)
    
    # Set labels and fixed limits for 3D plot
    ax1.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE+1)
    ax1.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE+1)
    ax1.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE+1)
    
    # Set fixed axis limits centered on data
    x_center = (np.max(ekf_cf[:, 0]) + np.min(ekf_cf[:, 0])) / 2
    y_center = (np.max(ekf_cf[:, 1]) + np.min(ekf_cf[:, 1])) / 2
    z_center = (np.max(ekf_cf[:, 2]) + np.min(ekf_cf[:, 2])) / 2
    
    ax1.set_xlim(x_center - 0.75, x_center + 0.75)
    ax1.set_ylim(y_center - 0.75, y_center + 0.75)
    ax1.set_zlim(z_center - 0.75, z_center + 0.75)
    ax1.grid(True, alpha=0.3)
    
    # XY plot
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot Vicon reference trajectory in gray
    ax2.plot(ref[:, 0], ref[:, 1], 'gray', linewidth=3, alpha=0.7, label='Vicon (Reference)')
    
    # Plot RoboDog trajectory FIRST (so it's underneath) with fixed green color
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        ax2.plot(dog_xyz[:, 0], dog_xyz[:, 1], 'g-', linewidth=2, alpha=0.8, label='RoboDog')
    
    # Create continuous colored line for EKF_CF trajectory (2D)
    points_2d = ekf_cf[:, :2].reshape(-1, 1, 2)
    segments_2d = np.concatenate([points_2d[:-1], points_2d[1:]], axis=1)
    
    # Create 2D line collection with colors
    lc2 = LineCollection(segments_2d, cmap='plasma', alpha=0.8, linewidths=3)
    lc2.set_array(total_error[:-1])
    line2 = ax2.add_collection(lc2)
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax2.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=200, label=name, edgecolors='k', alpha=0.9)
    
    # Mark start and end points
    if len(ref) > 0:
        ax2.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=120, 
                   label='Start', edgecolors='k', zorder=10)
        ax2.scatter(ref[-1, 0], ref[-1, 1], color='red', marker='s', s=120, 
                   label='End', edgecolors='k', zorder=10)
    
    # Set labels and properties
    ax2.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE+1)
    ax2.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE+1)
    ax2.grid(True, alpha=0.7)
    ax2.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    cbar = plt.colorbar(line2, ax=ax2, shrink=0.8, aspect=20)
    cbar.set_label('Error Magnitude [m]', fontsize=PLOT_FONT_SIZE)
    
    # Create unified legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=3, alpha=0.7, label='Vicon (Reference)'),
        Patch(facecolor='purple', alpha=0.8, label='EKF_CF (Error Colored)'),
        Line2D([0], [0], marker='o', color='green', linewidth=0, markersize=8, 
               markeredgecolor='k', label='Start'),
        Line2D([0], [0], marker='s', color='red', linewidth=0, markersize=8, 
               markeredgecolor='k', label='End')
    ]
    
    # Add RoboDog to legend if it exists
    if dog_data.size > 0:
        legend_elements.insert(1, Line2D([0], [0], color='green', linewidth=2, alpha=0.8, label='RoboDog'))
    
    # Add anchors to legend if they exist
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            legend_elements.append(
                Line2D([0], [0], marker='*', color=ANCHOR_COLORS[i], linewidth=0, markersize=10,
                       markeredgecolor='k', label=name)
            )
    
    fig.legend(handles=legend_elements, fontsize=PLOT_FONT_SIZE, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=len(legend_elements))
    
    # Calculate error statistics
    max_error = np.max(total_error)
    mean_error = np.mean(total_error)
    std_error = np.std(total_error)
    min_error = np.min(total_error)
    
    # Add statistics text box
    stats_text = f"Error Statistics:\nMin: {min_error:.3f} m\nMax: {max_error:.3f} m\nMean: {mean_error:.3f} m\nStd: {std_error:.3f} m"
    fig.text(0.5, 0.12, stats_text, fontsize=PLOT_FONT_SIZE, 
            verticalalignment='bottom', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, bottom=0.25)
    plt.savefig(output_path, bbox_inches='tight')
    
    return fig


def plot_robodog_trajectory(dog_data, anchors, output_path_3d, output_path_2d):
    """Create 3D and 2D trajectory plots for RoboDog in vicon/world frame."""
    if dog_data.size == 0:
        print("Warning: No RoboDog data available for plotting")
        return
    
    # Extract xyz positions
    dog_xyz = dog_data[:, 1:4]
    
    # 3D World frame plot
    fig = plt.figure(figsize=(8, 8), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(dog_xyz[:, 0], dog_xyz[:, 1], dog_xyz[:, 2], 'g-', linewidth=2, label='RoboDog')
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    # Mark start and end points
    if len(dog_xyz) > 0:
        ax.scatter(dog_xyz[0, 0], dog_xyz[0, 1], dog_xyz[0, 2], color='blue', marker='o', s=120, 
                   label='Start', edgecolors='k', zorder=10)
        ax.scatter(dog_xyz[-1, 0], dog_xyz[-1, 1], dog_xyz[-1, 2], color='red', marker='s', s=120, 
                   label='End', edgecolors='k', zorder=10)
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE)
    
    # Set axis limits for RoboDog
    ax.set_xlim(DOG_TRAJ_3D_X_LIM)
    ax.set_ylim(DOG_TRAJ_3D_Y_LIM)
    ax.set_zlim(DOG_TRAJ_3D_Z_LIM)
    
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title('RoboDog Trajectory in Vicon/World Frame (3D)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path_3d)
    plt.close()
    
    # 2D XY trajectory plot
    fig = plt.figure(figsize=(8, 8), dpi=DPI)
    ax = fig.add_subplot(111)
    
    ax.plot(dog_xyz[:, 0], dog_xyz[:, 1], 'g-', linewidth=2, label='RoboDog')
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    # Mark start and end points
    if len(dog_xyz) > 0:
        ax.scatter(dog_xyz[0, 0], dog_xyz[0, 1], color='blue', marker='o', s=120, 
                   label='Start', edgecolors='k', zorder=10)
        ax.scatter(dog_xyz[-1, 0], dog_xyz[-1, 1], color='red', marker='s', s=120, 
                   label='End', edgecolors='k', zorder=10)
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_FONT_SIZE, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title('RoboDog Trajectory in Vicon/World Frame (XY)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path_2d)
    plt.close()


def plot_combined_trajectories(t, ref, ekf_cf, dog_data, anchors, output_path_3d, output_path_2d):
    """Create combined 3D and 2D trajectory plots showing drone and RoboDog together."""
    # 3D World frame plot
    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot RoboDog trajectory FIRST (so it's underneath)
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        ax.plot(dog_xyz[:, 0], dog_xyz[:, 1], dog_xyz[:, 2], 'g-', linewidth=2, label='RoboDog')
    
    # Plot drone trajectories on top
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'b-', linewidth=2, label='Drone Vicon')
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], 'r--', linewidth=1.5, label='Drone EKF_CF')
    
    # Mark RoboDog start and end if available
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        if len(dog_xyz) > 0:
            ax.scatter(dog_xyz[0, 0], dog_xyz[0, 1], dog_xyz[0, 2], color='darkgreen', marker='o', s=100, 
                       label='RoboDog Start', edgecolors='k', zorder=10)
            ax.scatter(dog_xyz[-1, 0], dog_xyz[-1, 1], dog_xyz[-1, 2], color='darkred', marker='s', s=100, 
                       label='RoboDog End', edgecolors='k', zorder=10)
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    # Mark drone start and end points
    if len(ref) > 0:
        ax.scatter(ref[0, 0], ref[0, 1], ref[0, 2], color='blue', marker='o', s=100, 
                   label='Drone Start', edgecolors='k', zorder=10)
        ax.scatter(ref[-1, 0], ref[-1, 1], ref[-1, 2], color='red', marker='s', s=100, 
                   label='Drone End', edgecolors='k', zorder=10)
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE)
    
    # Use larger limits to accommodate both drone and dog
    ax.set_xlim(DOG_TRAJ_3D_X_LIM)
    ax.set_ylim(DOG_TRAJ_3D_Y_LIM)
    ax.set_zlim(DOG_TRAJ_3D_Z_LIM)
    
    ax.legend(fontsize=PLOT_FONT_SIZE-1, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    ax.set_title('Combined Trajectories in Vicon/World Frame (3D)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path_3d)
    plt.close()
    
    # 2D XY trajectory plot
    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    ax = fig.add_subplot(111)
    
    # Plot RoboDog trajectory FIRST (so it's underneath)
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        ax.plot(dog_xyz[:, 0], dog_xyz[:, 1], 'g-', linewidth=2, label='RoboDog')
    
    # Plot drone trajectories on top
    ax.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=2, label='Drone Vicon')
    ax.plot(ekf_cf[:, 0], ekf_cf[:, 1], 'r--', linewidth=1.5, label='Drone EKF_CF')
    
    # Mark RoboDog start and end if available
    if dog_data.size > 0:
        dog_xyz = dog_data[:, 1:4]
        if len(dog_xyz) > 0:
            ax.scatter(dog_xyz[0, 0], dog_xyz[0, 1], color='darkgreen', marker='o', s=100, 
                       label='RoboDog Start', edgecolors='k', zorder=10)
            ax.scatter(dog_xyz[-1, 0], dog_xyz[-1, 1], color='darkred', marker='s', s=100, 
                       label='RoboDog End', edgecolors='k', zorder=10)
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=100, label=name, edgecolors='k')
    
    # Mark drone start and end points
    if len(ref) > 0:
        ax.scatter(ref[0, 0], ref[0, 1], color='blue', marker='o', s=100, 
                   label='Drone Start', edgecolors='k', zorder=10)
        ax.scatter(ref[-1, 0], ref[-1, 1], color='red', marker='s', s=100, 
                   label='Drone End', edgecolors='k', zorder=10)
    
    ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
    ax.legend(fontsize=PLOT_FONT_SIZE-1, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
    ax.set_title('Combined Trajectories in Vicon/World Frame (XY)', fontsize=PLOT_FONT_SIZE)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path_2d)
    plt.close()


def create_3d_robot_snapshots(t, ref, ekf_cf, anchors, dog_data, output_path):
    """
    Create 3D snapshots showing robot shapes, coordinate frames and distance lines at 3 trajectory positions.
    
    Args:
        t: Time array
        ref: Reference trajectory (CH_REF_CHILD) - can be [N,3] or [N,8] with quaternions
        ekf_cf: EKF estimate trajectory (CH_EKF_CF_CHILD) - can be [N,3] or [N,8] with quaternions
        anchors: Anchor positions
        dog_data: RoboDog trajectory data - can be [N,4] or [N,8] with quaternions
        output_path: Path to save the figure
    """
    
    # Calculate snapshot indices: start, middle, end
    n_points = len(t)
    if n_points < 3:
        print("Warning: Not enough trajectory points for snapshots")
        return
        
    indices = [0, n_points // 2, n_points - 1]
    snapshot_labels = ['Inizio Traiettoria', 'Metà Traiettoria', 'Fine Traiettoria']
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6), dpi=DPI)
    
    for plot_idx, (idx, label) in enumerate(zip(indices, snapshot_labels)):
        ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')
        
        # Get positions at this time point
        ref_pos = ref[idx, :3] if ref.shape[1] >= 3 else ref[idx]
        ekf_pos = ekf_cf[idx, :3] if ekf_cf.shape[1] >= 3 else ekf_cf[idx]
        
        # Get orientations if available (quaternions [x, y, z, w])
        ref_quat = ref[idx, 3:7] if ref.shape[1] >= 7 else None
        ekf_quat = ekf_cf[idx, 3:7] if ekf_cf.shape[1] >= 7 else None
        
        # Get dog position and orientation if available
        dog_pos = None
        dog_quat = None
        if dog_data.size > 0 and idx < len(dog_data):
            if dog_data.shape[1] >= 8:  # New format with quaternion [time, x, y, z, qx, qy, qz, qw]
                dog_pos = dog_data[idx, 1:4]  # Position
                dog_quat = dog_data[idx, 4:8]  # Quaternion [x, y, z, w]
            elif dog_data.shape[1] >= 4:  # Old format [time, x, y, z]
                dog_pos = dog_data[idx, 1:4]  # Position only
                dog_quat = None
        
        # Draw RoboDog as a parallelepiped with dynamic orientation
        if dog_pos is not None:
            _draw_robodog_shape(ax, dog_pos, dog_quat)
        
        # Draw Crazyflie reference position as a sphere
        _draw_crazyflie_sphere(ax, ref_pos, color='blue', alpha=0.7, label='Vicon Reference')
        
        # Draw Crazyflie EKF estimate as a sphere  
        _draw_crazyflie_sphere(ax, ekf_pos, color='red', alpha=0.7, label='EKF Estimate')
        
        # Draw distance line between reference and estimate
        ax.plot([ref_pos[0], ekf_pos[0]], 
                [ref_pos[1], ekf_pos[1]], 
                [ref_pos[2], ekf_pos[2]], 
                'k--', linewidth=2, alpha=0.8, label=f'Distanza: {np.linalg.norm(ref_pos - ekf_pos):.3f} m')
        
        # Draw coordinate frames
        frame_scale = 0.15  # 15cm frame axes
        
        # World frame at origin
        ax.scatter(0, 0, 0, color='black', marker='o', s=100, alpha=0.8)
        _draw_coordinate_frame(ax, [0, 0, 0], None, frame_scale, 'World', alpha=0.6)
        
        # Reference frame (Vicon)
        _draw_coordinate_frame(ax, ref_pos, ref_quat, frame_scale, 'Vicon Ref', alpha=0.8)
        
        # EKF estimate frame
        _draw_coordinate_frame(ax, ekf_pos, ekf_quat, frame_scale, 'EKF Est', alpha=0.8)
        
        # Dog frame (if available)
        if dog_pos is not None:
            _draw_coordinate_frame(ax, dog_pos, dog_quat, frame_scale * 1.5, 'RoboDog', alpha=0.8)
        
        # Set labels and limits
        ax.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE)
        ax.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE)
        
        # Set appropriate axis limits based on data
        all_positions = [ref_pos, ekf_pos]
        if dog_pos is not None:
            all_positions.append(dog_pos)
            
        if len(all_positions) > 0:
            all_pos = np.array(all_positions)
            margin = 1.0  # 1 meter margin
            ax.set_xlim([np.min(all_pos[:, 0]) - margin, np.max(all_pos[:, 0]) + margin])
            ax.set_ylim([np.min(all_pos[:, 1]) - margin, np.max(all_pos[:, 1]) + margin])
            ax.set_zlim([max(0, np.min(all_pos[:, 2]) - margin), np.max(all_pos[:, 2]) + margin])
        
        ax.set_title(f'{label}\nTempo: {t[idx]:.2f} s', fontsize=PLOT_FONT_SIZE + 1, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first plot to avoid clutter
        if plot_idx == 0:
            ax.legend(fontsize=PLOT_FONT_SIZE - 2, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def create_3d_robot_snapshots_with_textures(ref_pos, ekf_pos, dog_pos, ref_quat=None, ekf_quat=None, dog_quat=None, 
                                           timestamps=None, title="3D Robot Trajectory Snapshots (Textured)"):
    """
    Create textured 3D snapshots of robot trajectories at start, middle, and end points.
    This version uses procedural textures and enhanced robot models.
    
    Args:
        ref_pos: Reference positions array (N, 3)
        ekf_pos: EKF positions array (N, 3) 
        dog_pos: Dog positions array (N, 3)
        ref_quat: Reference quaternions array (N, 4), optional
        ekf_quat: EKF quaternions array (N, 4), optional
        dog_quat: Dog quaternions array (N, 4), optional
        timestamps: Timestamps for the data, optional
        title: Plot title
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Snapshot indices: start, middle, end
    n_points = min(len(ref_pos), len(ekf_pos), len(dog_pos))
    if n_points < 3:
        print("Warning: Not enough data points for snapshots")
        return fig
    
    snapshot_indices = [0, n_points//2, n_points-1]
    snapshot_labels = ['Start', 'Middle', 'End']
    
    for i, (idx, label) in enumerate(zip(snapshot_indices, snapshot_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Get positions at this time point
        ref_point = ref_pos[idx]
        ekf_point = ekf_pos[idx] 
        dog_point = dog_pos[idx]
        
        # Get quaternions if available
        ref_q = ref_quat[idx] if ref_quat is not None else None
        ekf_q = ekf_quat[idx] if ekf_quat is not None else None
        dog_q = dog_quat[idx] if dog_quat is not None else None
        
        # Draw robots with textures
        try:
            _draw_robodog_with_texture(ax, dog_point, dog_q)
        except Exception as e:
            print(f"Warning: Could not draw textured RoboDog: {e}")
            # Fallback to basic visualization
            l, w, h = ROBODOG_LENGTH/2, ROBODOG_WIDTH/2, ROBODOG_HEIGHT/2
            ax.scatter(*dog_point, c='orange', s=100, label='RoboDog', alpha=0.7)
        
        try:
            _draw_crazyflie_with_texture(ax, ref_point, ref_q, 'blue', 'Reference')
            _draw_crazyflie_with_texture(ax, ekf_point, ekf_q, 'red', 'EKF_CF')
        except Exception as e:
            print(f"Warning: Could not draw textured Crazyflies: {e}")
            # Fallback to basic visualization
            ax.scatter(*ref_point, c='blue', s=60, label='Reference (Vicon)', alpha=0.7)
            ax.scatter(*ekf_point, c='red', s=60, label='EKF_CF Estimate', alpha=0.7)
        
        # Draw distance line
        ax.plot([ref_point[0], ekf_point[0]], 
                [ref_point[1], ekf_point[1]], 
                [ref_point[2], ekf_point[2]], 
                'purple', linewidth=3, alpha=0.8, label='Distance')
        
        # Calculate and display distance
        distance = np.linalg.norm(np.array(ref_point) - np.array(ekf_point))
        mid_point = (np.array(ref_point) + np.array(ekf_point)) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
                f'{distance:.3f}m', fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7, edgecolor='white'))
        
        # Draw coordinate frames with enhanced visualization
        try:
            # World frame (at origin)
            _draw_coordinate_frame(ax, [0, 0, 0], None, scale=0.2, label='World', alpha=0.6)
            
            # RoboDog frame
            _draw_coordinate_frame(ax, dog_point, dog_q, scale=0.15, label='RoboDog', alpha=0.8)
            
            # Reference frame  
            _draw_coordinate_frame(ax, ref_point, ref_q, scale=0.1, label='Vicon Ref', alpha=0.8)
            
            # EKF_CF frame
            _draw_coordinate_frame(ax, ekf_point, ekf_q, scale=0.1, label='EKF_CF', alpha=0.8)
            
        except Exception as e:
            print(f"Warning: Could not draw coordinate frames: {e}")
        
        # Enhanced axis settings
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10) 
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'{label} - t={timestamps[idx]:.2f}s' if timestamps is not None else label,
                    fontsize=12, fontweight='bold')
        
        # Set equal aspect ratio and reasonable limits
        max_range = 1.5
        center_x, center_y, center_z = dog_point
        ax.set_xlim([center_x - max_range, center_x + max_range])
        ax.set_ylim([center_y - max_range, center_y + max_range])
        ax.set_zlim([0, center_z + max_range])
        
        # Enhanced grid and background
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges more transparent
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        
        # Add legend for the first subplot only
        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_3d_robot_snapshots_enhanced(ref_pos, ekf_pos, dog_pos, ref_quat=None, ekf_quat=None, dog_quat=None, 
                                      timestamps=None, title="3D Robot Trajectory Snapshots (Enhanced)", 
                                      use_textures=True):
    """
    Create enhanced 3D snapshots with an option to use textures or geometric shapes.
    This provides both basic geometric and advanced textured visualization.
    
    Args:
        ref_pos: Reference positions array (N, 3)
        ekf_pos: EKF positions array (N, 3) 
        dog_pos: Dog positions array (N, 3)
        ref_quat: Reference quaternions array (N, 4), optional
        ekf_quat: EKF quaternions array (N, 4), optional
        dog_quat: Dog quaternions array (N, 4), optional
        timestamps: Timestamps for the data, optional
        title: Plot title
        use_textures: If True, use textured models; if False, use geometric shapes
    
    Returns:
        fig: matplotlib figure object
    """
    if use_textures:
        return create_3d_robot_snapshots_with_textures(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, timestamps, title)
    else:
        return create_3d_robot_snapshots(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, timestamps, title)
    


def _draw_robodog_shape(ax, position, quaternion=None):
    """Draw a parallelepiped representing the RoboDog at the given position with optional orientation."""
    import scipy.spatial.transform as sp_transform
    
    # The position represents the frame on the dog's back, so we need to offset to ground level
    x, y, z = position
    
    # Calculate the actual ground position (frame is on the back, offset down)
    ground_z = max(0, z - ROBODOG_FRAME_HEIGHT)  # Ensure not below ground
    
    # Use the dynamic height based on the actual frame position
    actual_height = z - ground_z if z > ground_z else ROBODOG_HEIGHT
    
    l, w, h = ROBODOG_LENGTH / 2, ROBODOG_WIDTH / 2, actual_height / 2
    
    # Center the box between ground and frame position
    center_z = ground_z + actual_height / 2
    
    # Define the 8 vertices of the parallelepiped centered at (x, y, center_z)
    vertices_local = np.array([
        [-l, -w, -h],  # 0: back-left-bottom
        [+l, -w, -h],  # 1: front-left-bottom  
        [+l, +w, -h],  # 2: front-right-bottom
        [-l, +w, -h],  # 3: back-right-bottom
        [-l, -w, +h],  # 4: back-left-top
        [+l, -w, +h],  # 5: front-left-top
        [+l, +w, +h],  # 6: front-right-top
        [-l, +w, +h],  # 7: back-right-top
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            # Convert quaternion [x, y, z, w] to rotation matrix
            rotation = sp_transform.Rotation.from_quat(quaternion)
            vertices_local = rotation.apply(vertices_local)
        except Exception as e:
            print(f"Warning: Could not apply rotation: {e}")
    
    # Translate to world position
    vertices = vertices_local + np.array([x, y, center_z])
    
    # Define the 6 faces of the parallelepiped
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # left
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # right  
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # front
        [vertices[4], vertices[7], vertices[3], vertices[0]],  # back
    ]
    
    # Create and add the 3D polygon collection
    collection = Poly3DCollection(faces, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=1)
    ax.add_collection3d(collection)


def _draw_crazyflie_sphere(ax, position, color='blue', alpha=0.7, label=None):
    """Draw a sphere representing the Crazyflie at the given position."""
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    radius = CRAZYFLIE_DIAMETER / 2
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + position[0]
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + position[1]
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + position[2]
    
    # Plot sphere surface
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=alpha, color=color, label=label)


def _draw_coordinate_frame(ax, position, quaternion=None, scale=0.1, label=None, alpha=0.8):
    """
    Draw a coordinate frame (X-red, Y-green, Z-blue) at the given position and orientation.
    
    Args:
        ax: matplotlib 3D axis
        position: [x, y, z] position of the frame origin
        quaternion: [x, y, z, w] quaternion for frame orientation (None for world frame)
        scale: length of the frame axes in meters
        label: optional label for the frame
        alpha: transparency of the frame
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Define the three axes in local coordinates
    axes_local = np.array([
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green) 
        [0, 0, scale]   # Z-axis (blue)
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
            axes_world = rotation.apply(axes_local)
        except Exception as e:
            print(f"Warning: Could not apply rotation to frame: {e}")
            axes_world = axes_local
    else:
        axes_world = axes_local
    
    # Colors for X, Y, Z axes
    colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']
    
    # Draw each axis
    for i, (axis, color, axis_label) in enumerate(zip(axes_world, colors, axis_labels)):
        end_point = np.array([x, y, z]) + axis
        
        # Draw the axis line
        ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], 
                color=color, linewidth=3, alpha=alpha)
        
        # Add arrowhead (simple approach with a point)
        ax.scatter(end_point[0], end_point[1], end_point[2], 
                  color=color, s=50, marker='>', alpha=alpha)
        
        # Add axis label at the end
        ax.text(end_point[0], end_point[1], end_point[2], f'  {axis_label}', 
                color=color, fontsize=8, alpha=alpha)
    
    # Add frame label if provided
    if label:
        ax.text(x, y, z + scale * 1.2, label, fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def _draw_robodog_with_texture(ax, position, quaternion=None):
    """Draw a textured RoboDog model at the given position with optional orientation."""
    import scipy.spatial.transform as sp_transform
    from matplotlib import cm
    
    # The position represents the frame on the dog's back
    x, y, z = position
    
    # Calculate the actual ground position
    ground_z = max(0, z - ROBODOG_FRAME_HEIGHT)
    actual_height = z - ground_z if z > ground_z else ROBODOG_HEIGHT
    
    l, w, h = ROBODOG_LENGTH / 2, ROBODOG_WIDTH / 2, actual_height / 2
    center_z = ground_z + actual_height / 2
    
    # Create a simplified dog mesh using multiple surfaces
    # Main body
    x_body = np.array([[x-l, x+l], [x-l, x+l]])
    y_body = np.array([[y-w, y-w], [y+w, y+w]])
    z_body = center_z + h * np.ones((2, 2))
    
    # Bottom
    z_bottom = center_z - h * np.ones((2, 2))
    
    # Try to load and apply a dog texture pattern
    try:
        # Create a procedural texture for the dog (simulating Unitree Go2 colors)
        dog_texture = _create_dog_texture()
        
        # Main body top (yellow/black pattern like Unitree Go2)
        ax.plot_surface(x_body, y_body, z_body, facecolors=dog_texture, alpha=0.8, shade=False)
        
        # Bottom (darker)
        bottom_texture = _create_dog_texture(variant='bottom')
        ax.plot_surface(x_body, y_body, z_bottom, facecolors=bottom_texture, alpha=0.6, shade=False)
        
        # Side panels
        # Left side
        x_side = np.array([[x-l, x+l], [x-l, x+l]])
        y_side = y-w * np.ones((2, 2))
        z_side = np.array([[center_z-h, center_z-h], [center_z+h, center_z+h]])
        
        side_texture = _create_dog_texture(variant='side')
        ax.plot_surface(x_side, y_side, z_side, facecolors=side_texture, alpha=0.7, shade=False)
        
        # Right side
        y_side_r = y+w * np.ones((2, 2))
        ax.plot_surface(x_side, y_side_r, z_side, facecolors=side_texture, alpha=0.7, shade=False)
        
    except Exception as e:
        print(f"Warning: Could not apply dog texture: {e}")
        # Fallback to solid color
        ax.plot_surface(x_body, y_body, z_body, color='goldenrod', alpha=0.7)
        ax.plot_surface(x_body, y_body, z_bottom, color='darkgoldenrod', alpha=0.5)


def _draw_crazyflie_with_texture(ax, position, quaternion=None, color_variant='blue', label=None):
    """Draw a textured Crazyflie at the given position with optional orientation."""
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    radius = CRAZYFLIE_DIAMETER / 2
    
    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    
    try:
        # Create Crazyflie texture
        cf_texture = _create_crazyflie_texture(color_variant)
        ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=cf_texture, alpha=0.8, shade=False)
        
        # Add propeller discs for realism
        _draw_crazyflie_propellers(ax, position, quaternion, color_variant)
        
    except Exception as e:
        print(f"Warning: Could not apply Crazyflie texture: {e}")
        # Fallback to solid color
        color = 'blue' if color_variant == 'blue' else 'red'
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.7)


def _draw_coordinate_frame(ax, position, quaternion=None, scale=0.1, label=None, alpha=0.8):
    """
    Draw a coordinate frame (X-red, Y-green, Z-blue) at the given position and orientation.
    
    Args:
        ax: matplotlib 3D axis
        position: [x, y, z] position of the frame origin
        quaternion: [x, y, z, w] quaternion for frame orientation (None for world frame)
        scale: length of the frame axes in meters
        label: optional label for the frame
        alpha: transparency of the frame
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Define the three axes in local coordinates
    axes_local = np.array([
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green) 
        [0, 0, scale]   # Z-axis (blue)
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
            axes_world = rotation.apply(axes_local)
        except Exception as e:
            print(f"Warning: Could not apply rotation to frame: {e}")
            axes_world = axes_local
    else:
        axes_world = axes_local
    
    # Colors for X, Y, Z axes
    colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']
    
    # Draw each axis
    for i, (axis, color, axis_label) in enumerate(zip(axes_world, colors, axis_labels)):
        end_point = np.array([x, y, z]) + axis
        
        # Draw the axis line
        ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], 
                color=color, linewidth=3, alpha=alpha)
        
        # Add arrowhead (simple approach with a point)
        ax.scatter(end_point[0], end_point[1], end_point[2], 
                  color=color, s=50, marker='>', alpha=alpha)
        
        # Add axis label at the end
        ax.text(end_point[0], end_point[1], end_point[2], f'  {axis_label}', 
                color=color, fontsize=8, alpha=alpha)
    
    # Add frame label if provided
    if label:
        ax.text(x, y, z + scale * 1.2, label, fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def _draw_robodog_with_texture(ax, position, quaternion=None):
    """Draw a textured RoboDog model at the given position with optional orientation."""
    import scipy.spatial.transform as sp_transform
    from matplotlib import cm
    
    # The position represents the frame on the dog's back
    x, y, z = position
    
    # Calculate the actual ground position
    ground_z = max(0, z - ROBODOG_FRAME_HEIGHT)
    actual_height = z - ground_z if z > ground_z else ROBODOG_HEIGHT
    
    l, w, h = ROBODOG_LENGTH / 2, ROBODOG_WIDTH / 2, actual_height / 2
    center_z = ground_z + actual_height / 2
    
    # Create a simplified dog mesh using multiple surfaces
    # Main body
    x_body = np.array([[x-l, x+l], [x-l, x+l]])
    y_body = np.array([[y-w, y-w], [y+w, y+w]])
    z_body = center_z + h * np.ones((2, 2))
    
    # Bottom
    z_bottom = center_z - h * np.ones((2, 2))
    
    # Try to load and apply a dog texture pattern
    try:
        # Create a procedural texture for the dog (simulating Unitree Go2 colors)
        dog_texture = _create_dog_texture()
        
        # Main body top (yellow/black pattern like Unitree Go2)
        ax.plot_surface(x_body, y_body, z_body, facecolors=dog_texture, alpha=0.8, shade=False)
        
        # Bottom (darker)
        bottom_texture = _create_dog_texture(variant='bottom')
        ax.plot_surface(x_body, y_body, z_bottom, facecolors=bottom_texture, alpha=0.6, shade=False)
        
        # Side panels
        # Left side
        x_side = np.array([[x-l, x+l], [x-l, x+l]])
        y_side = y-w * np.ones((2, 2))
        z_side = np.array([[center_z-h, center_z-h], [center_z+h, center_z+h]])
        
        side_texture = _create_dog_texture(variant='side')
        ax.plot_surface(x_side, y_side, z_side, facecolors=side_texture, alpha=0.7, shade=False)
        
        # Right side
        y_side_r = y+w * np.ones((2, 2))
        ax.plot_surface(x_side, y_side_r, z_side, facecolors=side_texture, alpha=0.7, shade=False)
        
    except Exception as e:
        print(f"Warning: Could not apply dog texture: {e}")
        # Fallback to solid color
        ax.plot_surface(x_body, y_body, z_body, color='goldenrod', alpha=0.7)
        ax.plot_surface(x_body, y_body, z_bottom, color='darkgoldenrod', alpha=0.5)


def _draw_crazyflie_with_texture(ax, position, quaternion=None, color_variant='blue', label=None):
    """Draw a textured Crazyflie at the given position with optional orientation."""
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    radius = CRAZYFLIE_DIAMETER / 2
    
    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    
    try:
        # Create Crazyflie texture
        cf_texture = _create_crazyflie_texture(color_variant)
        ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=cf_texture, alpha=0.8, shade=False)
        
        # Add propeller discs for realism
        _draw_crazyflie_propellers(ax, position, quaternion, color_variant)
        
    except Exception as e:
        print(f"Warning: Could not apply Crazyflie texture: {e}")
        # Fallback to solid color
        color = 'blue' if color_variant == 'blue' else 'red'
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.7)


def _draw_coordinate_frame(ax, position, quaternion=None, scale=0.1, label=None, alpha=0.8):
    """
    Draw a coordinate frame (X-red, Y-green, Z-blue) at the given position and orientation.
    
    Args:
        ax: matplotlib 3D axis
        position: [x, y, z] position of the frame origin
        quaternion: [x, y, z, w] quaternion for frame orientation (None for world frame)
        scale: length of the frame axes in meters
        label: optional label for the frame
        alpha: transparency of the frame
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Define the three axes in local coordinates
    axes_local = np.array([
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green) 
        [0, 0, scale]   # Z-axis (blue)
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
            axes_world = rotation.apply(axes_local)
        except Exception as e:
            print(f"Warning: Could not apply rotation to frame: {e}")
            axes_world = axes_local
    else:
        axes_world = axes_local
    
    # Colors for X, Y, Z axes
    colors = ['red', 'green', 'blue']
    axis_labels = ['X', 'Y', 'Z']
    
    # Draw each axis
    for i, (axis, color, axis_label) in enumerate(zip(axes_world, colors, axis_labels)):
        end_point = np.array([x, y, z]) + axis
        
        # Draw the axis line
        ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], 
                color=color, linewidth=3, alpha=alpha)
        
        # Add arrowhead (simple approach with a point)
        ax.scatter(end_point[0], end_point[1], end_point[2], 
                  color=color, s=50, marker='>', alpha=alpha)
        
        # Add axis label at the end
        ax.text(end_point[0], end_point[1], end_point[2], f'  {axis_label}', 
                color=color, fontsize=8, alpha=alpha)
    
    # Add frame label if provided
    if label:
        ax.text(x, y, z + scale * 1.2, label, fontsize=9, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


# ============================================================================
# REALISTIC MESH GENERATION FUNCTIONS
# ============================================================================

def _draw_unitree_go2_realistic_mesh(ax, position, quaternion=None):
    """
    Draw a highly realistic mesh representation of the Unitree Go2 robot.
    This uses detailed mesh generation with proper proportions and colors.
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
        except:
            rotation = None
    else:
        rotation = None
    
    def apply_rotation_and_translation(points, rotation_obj, translation):
        if rotation_obj is not None:
            rotated_points = rotation_obj.apply(points)
        else:
            rotated_points = points.copy()
        return rotated_points + np.array(translation)
    
    # Unitree Go2 dimensions (realistic proportions)
    body_length = ROBODOG_LENGTH
    body_width = ROBODOG_WIDTH
    body_height = 0.15  # Body thickness
    leg_length = 0.3    # Total leg length
    leg_thickness = 0.03
    
    # Dynamic height adjustment
    if z < 0.2:  # Sitting/lying
        actual_height = 0.15
        leg_bend = 0.8
    elif z < 0.35:  # Low stance
        actual_height = z
        leg_bend = 0.6
    else:  # Standing/jumping
        actual_height = z
        leg_bend = 0.3
    
    # === MAIN BODY ===
    body_vertices = np.array([
        # Bottom face
        [-body_length/2, -body_width/2, -body_height/2],
        [body_length/2, -body_width/2, -body_height/2],
        [body_length/2, body_width/2, -body_height/2],
        [-body_length/2, body_width/2, -body_height/2],
        # Top face
        [-body_length/2, -body_width/2, body_height/2],
        [body_length/2, -body_width/2, body_height/2],
        [body_length/2, body_width/2, body_height/2],
        [-body_length/2, body_width/2, body_height/2]
    ])
    
    body_vertices = apply_rotation_and_translation(body_vertices, rotation, [x, y, z])
    
    # Draw body as triangulated surfaces
    body_faces = [
        [0, 1, 2, 3],  # Bottom
        [4, 7, 6, 5],  # Top
        [0, 4, 5, 1],  # Front
        [2, 6, 7, 3],  # Back
        [0, 3, 7, 4],  # Left
        [1, 5, 6, 2]   # Right
    ]
    
    for face in body_faces:
        quad_points = body_vertices[face]
        # Split each quad into two triangles
        triangle1 = quad_points[[0, 1, 2]]
        triangle2 = quad_points[[0, 2, 3]]
        ax.plot_trisurf(*triangle1.T, color='black', alpha=0.9, shade=True)
        ax.plot_trisurf(*triangle2.T, color='black', alpha=0.9, shade=True)
    
    # === LEGS ===
    leg_positions = [
        [body_length/3, body_width/2 + 0.05, 0],    # Front right
        [body_length/3, -body_width/2 - 0.05, 0],   # Front left
        [-body_length/3, body_width/2 + 0.05, 0],   # Back right
        [-body_length/3, -body_width/2 - 0.05, 0]   # Back left
    ]
    
    for leg_pos in leg_positions:
        # Upper leg
        upper_leg_start = np.array(leg_pos) + [x, y, z]
        upper_leg_end = upper_leg_start + [0, 0, -leg_length/2 * (1 + leg_bend)]
        
        # Lower leg
        lower_leg_start = upper_leg_end
        lower_leg_end = lower_leg_start + [0, 0, -leg_length/2 * (1 + leg_bend)]
        
        # Draw leg segments as cylinders (approximated with lines and points)
        ax.plot([upper_leg_start[0], upper_leg_end[0]], 
                [upper_leg_start[1], upper_leg_end[1]], 
                [upper_leg_start[2], upper_leg_end[2]], 
                color='darkgray', linewidth=6, alpha=0.9)
        
        ax.plot([lower_leg_start[0], lower_leg_end[0]], 
                [lower_leg_start[1], lower_leg_end[1]], 
                [lower_leg_start[2], lower_leg_end[2]], 
                color='darkgray', linewidth=4, alpha=0.9)
        
        # Joints
        ax.scatter(*upper_leg_end, color='silver', s=30, alpha=0.9)
        ax.scatter(*lower_leg_end, color='black', s=40, alpha=0.9)  # Feet
    
    # === HEAD ===
    head_length = 0.15
    head_width = 0.12
    head_height = 0.08
    head_offset = [body_length/2 + head_length/2, 0, body_height/4]
    head_center = apply_rotation_and_translation(np.array([head_offset]), rotation, [x, y, z])[0]
    
    head_vertices = np.array([
        [-head_length/2, -head_width/2, -head_height/2],
        [head_length/2, -head_width/2, -head_height/2],
        [head_length/2, head_width/2, -head_height/2],
        [-head_length/2, head_width/2, -head_height/2],
        [-head_length/2, -head_width/2, head_height/2],
        [head_length/2, -head_width/2, head_height/2],
        [head_length/2, head_width/2, head_height/2],
        [-head_length/2, head_width/2, head_height/2]
    ])
    
    head_vertices = apply_rotation_and_translation(head_vertices, rotation, head_center)
    
    # Draw head
    for face in body_faces:  # Same face structure as body
        quad_points = head_vertices[face]
        triangle1 = quad_points[[0, 1, 2]]
        triangle2 = quad_points[[0, 2, 3]]
        ax.plot_trisurf(*triangle1.T, color='darkblue', alpha=0.9, shade=True)
        ax.plot_trisurf(*triangle2.T, color='darkblue', alpha=0.9, shade=True)
    
    
    # === SENSORS AND DETAILS ===
    # Camera/LIDAR on top
    sensor_pos = apply_rotation_and_translation(
        np.array([[0, 0, body_height/2 + 0.05]]), rotation, [x, y, z])[0]
    ax.scatter(*sensor_pos, color='red', s=80, alpha=0.9, marker='s')
    
    # Status lights
    light_positions = [
        [body_length/4, body_width/2, body_height/2 + 0.01],
        [body_length/4, -body_width/2, body_height/2 + 0.01]
    ]
    for light_pos in light_positions:
        light_world = apply_rotation_and_translation(
            np.array([light_pos]), rotation, [x, y, z])[0]
        ax.scatter(*light_world, color='lime', s=20, alpha=0.9)
    
    # Add label
    label_pos = [x, y, z + 0.3]
    ax.text(*label_pos, 'Unitree Go2', fontsize=10, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8))


def _draw_crazyflie_realistic_mesh(ax, position, quaternion=None, color='blue', label='Crazyflie'):
    """
    Draw a highly realistic mesh representation of the Crazyflie 2.1 drone.
    This includes detailed body, motor arms, propellers, and electronic components.
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
        except:
            rotation = None
    else:
        rotation = None
    
    def apply_rotation_and_translation(points, rotation_obj, translation):
        if rotation_obj is not None:
            rotated_points = rotation_obj.apply(points)
        else:
            rotated_points = points.copy()
        return rotated_points + np.array(translation)
    
    # Crazyflie 2.1 realistic dimensions
    arm_length = 0.046  # 46mm arm to arm
    body_size = 0.027   # 27mm body diameter
    body_height = 0.015 # 15mm height
    prop_radius = 0.023 # 23mm propeller radius
    
    # === MAIN BODY (PCB) ===
    # Create octagonal body shape for more realism
    angles = np.linspace(0, 2*np.pi, 9)
    body_top = np.array([[body_size/2 * np.cos(angle), body_size/2 * np.sin(angle), body_height/2] 
                        for angle in angles])
    body_bottom = np.array([[body_size/2 * np.cos(angle), body_size/2 * np.sin(angle), -body_height/2] 
                           for angle in angles])
    
    body_top = apply_rotation_and_translation(body_top, rotation, [x, y, z])
    body_bottom = apply_rotation_and_translation(body_bottom, rotation, [x, y, z])
    
    # Draw body sides
    for i in range(len(angles)-1):
        quad_points = np.array([body_bottom[i], body_bottom[i+1], body_top[i+1], body_top[i]])
        triangle1 = quad_points[[0, 1, 2]]
        triangle2 = quad_points[[0, 2, 3]]
        ax.plot_trisurf(*triangle1.T, color=color, alpha=0.9, shade=True)
        ax.plot_trisurf(*triangle2.T, color=color, alpha=0.9, shade=True)
    
    # Draw top and bottom
    center_top = np.mean(body_top[:-1], axis=0)
    center_bottom = np.mean(body_bottom[:-1], axis=0)
    
    for i in range(len(angles)-1):
        # Top face
        triangle_top = np.array([center_top, body_top[i], body_top[i+1]])
        ax.plot_trisurf(*triangle_top.T, color=color, alpha=0.9, shade=True)
        
        # Bottom face
        triangle_bottom = np.array([center_bottom, body_bottom[i+1], body_bottom[i]])
        ax.plot_trisurf(*triangle_bottom.T, color='darkgreen', alpha=0.9, shade=True)
    
    # === MOTOR ARMS ===
    arm_positions = [
        [arm_length/2, arm_length/2, 0],    # Front right
        [-arm_length/2, arm_length/2, 0],   # Front left  
        [-arm_length/2, -arm_length/2, 0],  # Back left
        [arm_length/2, -arm_length/2, 0]    # Back right
    ]
    
    arm_thickness = 0.003
    motor_size = 0.007
    
    for i, arm_pos in enumerate(arm_positions):
        # Arm center line
        arm_center = apply_rotation_and_translation(np.array([arm_pos]), rotation, [x, y, z])[0]
        body_center = np.array([x, y, z])
        
        # Draw arm as thick line
        ax.plot([body_center[0], arm_center[0]], 
                [body_center[1], arm_center[1]], 
                [body_center[2], arm_center[2]], 
                color='black', linewidth=8, alpha=0.9)
        
        # === MOTORS ===
        motor_height = 0.015
        motor_vertices = np.array([
            [-motor_size/2, -motor_size/2, -motor_height/2],
            [motor_size/2, -motor_size/2, -motor_height/2],
            [motor_size/2, motor_size/2, -motor_height/2],
            [-motor_size/2, motor_size/2, -motor_height/2],
            [-motor_size/2, -motor_size/2, motor_height/2],
            [motor_size/2, -motor_size/2, motor_height/2],
            [motor_size/2, motor_size/2, motor_height/2],
            [-motor_size/2, motor_size/2, motor_height/2]
        ])
        
        motor_vertices = apply_rotation_and_translation(motor_vertices, rotation, arm_center)
        
        # Draw motor housing
        motor_faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 7, 6, 5],  # Top
            [0, 4, 5, 1],  # Front
            [2, 6, 7, 3],  # Back
            [0, 3, 7, 4],  # Left
            [1, 5, 6, 2]   # Right
        ]
        
        for face in motor_faces:
            quad_points = motor_vertices[face]
            triangle1 = quad_points[[0, 1, 2]]
            triangle2 = quad_points[[0, 2, 3]]
            ax.plot_trisurf(*triangle1.T, color='silver', alpha=0.9, shade=True)
            ax.plot_trisurf(*triangle2.T, color='silver', alpha=0.9, shade=True)
        
        # === PROPELLERS ===
        # Create propeller blades
        blade_angles = [0, 90]  # Two blades per propeller
        prop_center = arm_center + apply_rotation_and_translation(
            np.array([[0, 0, motor_height/2 + 0.002]]), rotation, [0, 0, 0])[0]
        
        for blade_angle in blade_angles:
            # Blade shape (simplified)
            blade_length = prop_radius * 0.8
            blade_width = prop_radius * 0.1
            
            blade_points = np.array([
                [0, 0, 0],
                [blade_length * np.cos(np.radians(blade_angle)), 
                 blade_length * np.sin(np.radians(blade_angle)), 0],
                [blade_length * np.cos(np.radians(blade_angle)) + blade_width * np.cos(np.radians(blade_angle + 90)),
                 blade_length * np.sin(np.radians(blade_angle)) + blade_width * np.sin(np.radians(blade_angle + 90)), 0],
                [blade_width * np.cos(np.radians(blade_angle + 90)),
                 blade_width * np.sin(np.radians(blade_angle + 90)), 0]
            ])
            
            blade_points = apply_rotation_and_translation(blade_points, rotation, prop_center)
            
            # Draw blade as filled triangle
            triangle_blade = blade_points[[0, 1, 2]]
            ax.plot_trisurf(*triangle_blade.T, color='gray', alpha=0.7, shade=True)
            triangle_blade2 = blade_points[[0, 2, 3]]
            ax.plot_trisurf(*triangle_blade2.T, color='gray', alpha=0.7, shade=True)
    
    # === ELECTRONIC COMPONENTS ===
    # Battery (underneath)
    battery_pos = apply_rotation_and_translation(
        np.array([[0, 0, -body_height/2 - 0.005]]), rotation, [x, y, z])[0]
    ax.scatter(*battery_pos, color='red', s=60, alpha=0.9, marker='s')
    
    # LED lights
    led_positions = [
        [body_size/3, 0, body_height/2 + 0.001],
        [-body_size/3, 0, body_height/2 + 0.001],
        [0, body_size/3, body_height/2 + 0.001],
        [0, -body_size/3, body_height/2 + 0.001]
    ]
    
    led_colors = ['red', 'red', 'blue', 'blue']
    for led_pos, led_color in zip(led_positions, led_colors):
        led_world = apply_rotation_and_translation(
            np.array([led_pos]), rotation, [x, y, z])[0]
        ax.scatter(*led_world, color=led_color, s=15, alpha=1.0)
    
    # Add label
    label_pos = [x, y, z + 0.08]
    ax.text(*label_pos, label, fontsize=9, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))


def _draw_coordinate_frame_enhanced(ax, position, quaternion=None, scale=0.1, label=None, alpha=0.8):
    """
    Draw an enhanced coordinate frame with better visualization and labels.
    """
    import scipy.spatial.transform as sp_transform
    
    x, y, z = position
    
    # Define the three axes in local coordinates with arrow shapes
    axes_local = np.array([
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green) 
        [0, 0, scale]   # Z-axis (blue)
    ])
    
    # Apply rotation if quaternion is provided
    if quaternion is not None and len(quaternion) == 4:
        try:
            rotation = sp_transform.Rotation.from_quat(quaternion)
            axes_world = rotation.apply(axes_local)
        except Exception as e:
            print(f"Warning: Could not apply rotation to frame: {e}")
            axes_world = axes_local
    else:
        axes_world = axes_local
    
    # Enhanced colors and styling for axes
    colors = ['crimson', 'forestgreen', 'royalblue']
    axis_labels = ['X', 'Y', 'Z']
    
    # Draw each axis with enhanced styling
    for i, (axis, color, axis_label) in enumerate(zip(axes_world, colors, axis_labels)):
        end_point = np.array([x, y, z]) + axis
        
        # Draw the axis line with gradient effect (thicker at base)
        ax.plot([x, end_point[0]], [y, end_point[1]], [z, end_point[2]], 
                color=color, linewidth=5, alpha=alpha, solid_capstyle='round')
        
        # Add arrowhead with enhanced styling
        arrow_size = scale * 0.15
        arrow_base = end_point - axis * 0.2
        
        # Create arrowhead cone
        cone_angles = np.linspace(0, 2*np.pi, 8)
        cone_radius = arrow_size
        
        for j in range(len(cone_angles)-1):
            angle1, angle2 = cone_angles[j], cone_angles[j+1]
            
            # Create arrowhead triangle
            if i == 0:  # X-axis
                p1 = arrow_base + np.array([0, cone_radius*np.cos(angle1), cone_radius*np.sin(angle1)])
                p2 = arrow_base + np.array([0, cone_radius*np.cos(angle2), cone_radius*np.sin(angle2)])
            elif i == 1:  # Y-axis
                p1 = arrow_base + np.array([cone_radius*np.cos(angle1), 0, cone_radius*np.sin(angle1)])
                p2 = arrow_base + np.array([cone_radius*np.cos(angle2), 0, cone_radius*np.sin(angle2)])
            else:  # Z-axis
                p1 = arrow_base + np.array([cone_radius*np.cos(angle1), cone_radius*np.sin(angle1), 0])
                p2 = arrow_base + np.array([cone_radius*np.cos(angle2), cone_radius*np.sin(angle2), 0])
            
            triangle = np.array([end_point, p1, p2])
            ax.plot_trisurf(*triangle.T, color=color, alpha=alpha*0.8, shade=True)
        
        # Add enhanced axis label
        label_offset = axis * 1.3
        label_pos = np.array([x, y, z]) + label_offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], f'{axis_label}', 
                color=color, fontsize=10, weight='bold', alpha=alpha,
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', 
                         edgecolor=color, alpha=0.8))
    
    # Add frame label if provided
    if label:
        ax.text(x, y, z + scale * 1.5, label, fontsize=11, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                         edgecolor='black', alpha=0.9, linewidth=1))
