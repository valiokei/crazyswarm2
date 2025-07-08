#!/usr/bin/env python3
"""
analysis.py
----------
Core analysis functions for processing single rosbags.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .rosbag_reader import extract_streams
    from .data_processing import apply_static_transform, align_streams, compute_rmse, compute_stddev, align_streams_with_orientation
    from .plotting import (plot_3d_trajectory, plot_errors, plot_voltage_data, plot_error_colored_trajectory,
                          plot_robodog_trajectory, plot_combined_trajectories, create_3d_robot_snapshots)
    from .config import PLOT_FONT_SIZE, DPI, ANCHOR_FRAMES, ANCHOR_COLORS
except ImportError:
    from rosbag_reader import extract_streams
    from data_processing import apply_static_transform, align_streams, compute_rmse, compute_stddev, align_streams_with_orientation
    from plotting import (plot_3d_trajectory, plot_errors, plot_voltage_data, plot_error_colored_trajectory,
                          plot_robodog_trajectory, plot_combined_trajectories, create_3d_robot_snapshots)
    from config import PLOT_FONT_SIZE, DPI, ANCHOR_FRAMES, ANCHOR_COLORS


def analyse_bag(bag_dir, out_dir):
    """
    Analyze a single rosbag and generate all plots and metrics.
    
    Args:
        bag_dir (pathlib.Path): Path to the rosbag directory
        out_dir (pathlib.Path): Output directory for results
        
    Returns:
        tuple: (csv_path, plot_path, rmse_total, stddev_total, bag_out_dir)
    """
    bag_dir = pathlib.Path(bag_dir)
    
    # Create subfolder for this rosbag
    bag_out_dir = out_dir / bag_dir.name
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data from rosbag
    streams, anchors, voltage_data = extract_streams(bag_dir)
    streams, anchors, voltage_data = apply_static_transform(streams, anchors, voltage_data)
    
    # Align streams and compute errors
    t, ref, ekf_cf = align_streams(streams)
    rmse_ekf_cf_axis, rmse_ekf_cf_tot, err_ekf_cf = compute_rmse(ref, ekf_cf)
    stddev_ekf_cf_axis, stddev_ekf_cf_tot = compute_stddev(err_ekf_cf)

    # Save CSV summary with metrics
    csv_path = bag_out_dir / f"{bag_dir.name}_metrics.csv"
    pd.DataFrame({
        'metric': ['RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_TOT', 
                   'StdDev_X', 'StdDev_Y', 'StdDev_Z', 'StdDev_TOT'],
        'ekf_cf': np.concatenate([rmse_ekf_cf_axis, [rmse_ekf_cf_tot], 
                                 stddev_ekf_cf_axis, [stddev_ekf_cf_tot]])
    }).to_csv(csv_path, index=False)

    # Create traditional trajectory plots
    plot_path = bag_out_dir / f"{bag_dir.name}_traj.pdf"
    plt.figure(figsize=(6.3, 2.8*3), dpi=DPI)
    axx = [plt.subplot(3, 1, i+1) for i in range(3)]
    labels = ['x', 'y', 'z']
    for i, ax in enumerate(axx):
        ax.plot(t, ref[:, i], label='Vicon', linewidth=1.0)
        ax.plot(t, ekf_cf[:, i], label='EKF_CF', linewidth=0.8, linestyle='--')
        ax.set_ylabel(labels[i]+' [m]', fontsize=PLOT_FONT_SIZE)
        ax.grid(True, which='both', linestyle=':')
        if i == 0:
            ax.legend(fontsize=PLOT_FONT_SIZE, ncol=2, loc='upper center')
    axx[-1].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Create individual plots
    traj3d_path = bag_out_dir / f"{bag_dir.name}_traj3d.pdf"
    traj2d_path = bag_out_dir / f"{bag_dir.name}_traj2d.pdf"
    plot_3d_trajectory(t, ref, ekf_cf, anchors, traj3d_path, traj2d_path)
    
    error_path = bag_out_dir / f"{bag_dir.name}_errors.pdf"
    plot_errors(t, err_ekf_cf, error_path)
    
    voltage_path = bag_out_dir / f"{bag_dir.name}_voltage.pdf"
    plot_voltage_data(voltage_data, voltage_path)
    
    # Create comprehensive plot
    _create_comprehensive_plot(bag_dir, t, ref, ekf_cf, err_ekf_cf, anchors, voltage_data,
                              rmse_ekf_cf_axis, rmse_ekf_cf_tot, stddev_ekf_cf_axis, 
                              stddev_ekf_cf_tot, bag_out_dir)
    
    # Create error-colored trajectory plot
    error_colored_path = bag_out_dir / f"{bag_dir.name}_error_colored_trajectory.pdf"
    error_fig = plot_error_colored_trajectory(t, ref, ekf_cf, err_ekf_cf, anchors, streams['dog'],
                                            error_colored_path, show_plot=False)
    plt.close(error_fig)
    
    # Create RoboDog trajectory plots
    if streams['dog'].size > 0:
        robodog_3d_path = bag_out_dir / f"{bag_dir.name}_robodog_traj3d.pdf"
        robodog_2d_path = bag_out_dir / f"{bag_dir.name}_robodog_traj2d.pdf"
        plot_robodog_trajectory(streams['dog'], anchors, robodog_3d_path, robodog_2d_path)
        
        # Create combined trajectory plots (drone + robodog)
        combined_3d_path = bag_out_dir / f"{bag_dir.name}_combined_traj3d.pdf"
        combined_2d_path = bag_out_dir / f"{bag_dir.name}_combined_traj2d.pdf"
        plot_combined_trajectories(t, ref, ekf_cf, streams['dog'], anchors, combined_3d_path, combined_2d_path)
    else:
        print(f"Warning: No RoboDog data found in {bag_dir.name}")
        
    # Create 3D robot snapshots visualization
    snapshots_path = bag_out_dir / f"{bag_dir.name}_3d_snapshots.pdf"
    try:
        # Get aligned data with orientation for snapshots
        t_orient, ref_orient, ekf_cf_orient = align_streams_with_orientation(streams)
        create_3d_robot_snapshots(t_orient, ref_orient, ekf_cf_orient, anchors, streams['dog'], snapshots_path)
    except Exception as e:
        print(f"Warning: Could not create 3D snapshots: {e}")
        # Fallback to position-only data
        try:
            create_3d_robot_snapshots(t, ref, ekf_cf, anchors, streams['dog'], snapshots_path)
        except Exception as e2:
            print(f"Warning: Could not create 3D snapshots (fallback): {e2}")
    
    return csv_path, plot_path, rmse_ekf_cf_tot, stddev_ekf_cf_tot, bag_out_dir


def analyse_bag_with_display(bag_dir, out_dir):
    """
    Analyze a single bag and display all plots in addition to saving them.
    This is the interactive version for single bag analysis.
    """
    bag_dir = pathlib.Path(bag_dir)
    
    # Create subfolder for this rosbag
    bag_out_dir = out_dir / bag_dir.name
    bag_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and process data
    streams, anchors, voltage_data = extract_streams(bag_dir)
    streams, anchors, voltage_data = apply_static_transform(streams, anchors, voltage_data)
    
    t, ref, ekf_cf = align_streams(streams)
    rmse_ekf_cf_axis, rmse_ekf_cf_tot, err_ekf_cf = compute_rmse(ref, ekf_cf)
    stddev_ekf_cf_axis, stddev_ekf_cf_tot = compute_stddev(err_ekf_cf)

    # Save CSV summary
    csv_path = bag_out_dir / f"{bag_dir.name}_metrics.csv"
    pd.DataFrame({
        'metric': ['RMSE_X', 'RMSE_Y', 'RMSE_Z', 'RMSE_TOT', 
                   'StdDev_X', 'StdDev_Y', 'StdDev_Z', 'StdDev_TOT'],
        'ekf_cf': np.concatenate([rmse_ekf_cf_axis, [rmse_ekf_cf_tot], 
                                 stddev_ekf_cf_axis, [stddev_ekf_cf_tot]])
    }).to_csv(csv_path, index=False)

    # Create and display comprehensive plot
    _create_comprehensive_plot(bag_dir, t, ref, ekf_cf, err_ekf_cf, anchors, voltage_data,
                              rmse_ekf_cf_axis, rmse_ekf_cf_tot, stddev_ekf_cf_axis, 
                              stddev_ekf_cf_tot, bag_out_dir)
    
    # Create individual plots for compatibility
    plot_path = bag_out_dir / f"{bag_dir.name}_traj.pdf"
    plt.figure(figsize=(6.3, 2.8*3), dpi=DPI)
    axx = [plt.subplot(3, 1, i+1) for i in range(3)]
    labels = ['x', 'y', 'z']
    for i, ax in enumerate(axx):
        ax.plot(t, ref[:, i], label='Vicon', linewidth=1.0)
        ax.plot(t, ekf_cf[:, i], label='EKF_CF', linewidth=0.8, linestyle='--')
        ax.set_ylabel(labels[i]+' [m]', fontsize=PLOT_FONT_SIZE)
        ax.grid(True, which='both', linestyle=':')
        if i == 0:
            ax.legend(fontsize=PLOT_FONT_SIZE, ncol=2, loc='upper center')
    axx[-1].set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Create other individual plots
    traj3d_path = bag_out_dir / f"{bag_dir.name}_traj3d.pdf"
    traj2d_path = bag_out_dir / f"{bag_dir.name}_traj2d.pdf"
    plot_3d_trajectory(t, ref, ekf_cf, anchors, traj3d_path, traj2d_path)
    
    error_path = bag_out_dir / f"{bag_dir.name}_errors.pdf"
    plot_errors(t, err_ekf_cf, error_path)
    
    voltage_path = bag_out_dir / f"{bag_dir.name}_voltage.pdf"
    plot_voltage_data(voltage_data, voltage_path)
    
    # Create error-colored trajectory plot with display
    error_colored_path = bag_out_dir / f"{bag_dir.name}_error_colored_trajectory.pdf"
    error_fig = plot_error_colored_trajectory(t, ref, ekf_cf, err_ekf_cf, anchors, streams['dog'],
                                            error_colored_path, show_plot=True)
    
    # Create RoboDog trajectory plots with display
    if streams['dog'].size > 0:
        robodog_3d_path = bag_out_dir / f"{bag_dir.name}_robodog_traj3d.pdf"
        robodog_2d_path = bag_out_dir / f"{bag_dir.name}_robodog_traj2d.pdf"
        plot_robodog_trajectory(streams['dog'], anchors, robodog_3d_path, robodog_2d_path)
        
        # Create combined trajectory plots (drone + robodog) with display
        combined_3d_path = bag_out_dir / f"{bag_dir.name}_combined_traj3d.pdf"
        combined_2d_path = bag_out_dir / f"{bag_dir.name}_combined_traj2d.pdf"
        plot_combined_trajectories(t, ref, ekf_cf, streams['dog'], anchors, combined_3d_path, combined_2d_path)
    else:
        print(f"Warning: No RoboDog data found in {bag_dir.name}")
    
    # Create 3D robot snapshots visualization (both geometric and textured)
    snapshots_path = bag_out_dir / f"{bag_dir.name}_3d_snapshots.pdf"
    snapshots_textured_path = bag_out_dir / f"{bag_dir.name}_3d_snapshots_textured.pdf"
    
    try:
        # Get aligned data with orientation for snapshots
        aligned_data = align_streams_with_orientation(streams)
        
        if len(aligned_data) >= 6:  # New format with orientation
            t_align, ref_pos, ekf_pos, ref_quat, ekf_quat, dog_data = aligned_data
            dog_pos = dog_data[:, :3] if dog_data.shape[1] >= 3 else dog_data
            dog_quat = dog_data[:, 3:7] if dog_data.shape[1] >= 7 else None
            
            # Create geometric snapshots
            from .plotting import create_3d_robot_snapshots
            fig_geom = create_3d_robot_snapshots(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t_align)
            fig_geom.savefig(snapshots_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig_geom)
            
            # Create textured snapshots
            from .plotting import create_3d_robot_snapshots_with_textures
            fig_textured = create_3d_robot_snapshots_with_textures(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t_align)
            fig_textured.savefig(snapshots_textured_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig_textured)
            
            print(f"3D robot snapshots (geometric) saved to: {snapshots_path}")
            print(f"3D robot snapshots (textured) saved to: {snapshots_textured_path}")
            
        else:  # Fallback for position-only data
            t_align, ref_pos, ekf_pos = aligned_data[:3]
            dog_pos = streams['dog'][:, :3] if streams['dog'].shape[1] >= 3 else streams['dog']
            
            from .plotting import create_3d_robot_snapshots
            fig_geom = create_3d_robot_snapshots(ref_pos, ekf_pos, dog_pos, timestamps=t_align)
            fig_geom.savefig(snapshots_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig_geom)
            
            print(f"3D robot snapshots (position-only) saved to: {snapshots_path}")
            
    except Exception as e:
        print(f"Warning: Could not create 3D snapshots: {e}")
        # Final fallback to basic position data
        try:
            dog_pos = streams['dog'][:, :3] if streams['dog'].shape[1] >= 3 else streams['dog']
            from .plotting import create_3d_robot_snapshots
            fig_fallback = create_3d_robot_snapshots(ref, ekf_cf, dog_pos, timestamps=t)
            fig_fallback.savefig(snapshots_path, dpi=DPI, bbox_inches='tight')
            plt.close(fig_fallback)
            print(f"3D robot snapshots (fallback) saved to: {snapshots_path}")
        except Exception as e2:
            print(f"Warning: Could not create 3D snapshots (fallback): {e2}")
    
    # Show both figures together
    plt.show()
    plt.close('all')

    return csv_path, plot_path, rmse_ekf_cf_tot, stddev_ekf_cf_tot, bag_out_dir


def _create_comprehensive_plot(bag_dir, t, ref, ekf_cf, err_ekf_cf, anchors, voltage_data,
                              rmse_ekf_cf_axis, rmse_ekf_cf_tot, stddev_ekf_cf_axis, 
                              stddev_ekf_cf_tot, bag_out_dir):
    """Create the comprehensive plot with all analysis in a single figure."""
    fig = plt.figure(figsize=(18, 12), dpi=DPI)
    fig.suptitle(f'Complete Analysis: {bag_dir.name}', fontsize=16, fontweight='bold', y=0.96)
    
    # 1. Traditional trajectory plot (top left)
    ax1 = plt.subplot(2, 3, 1)
    labels = ['x', 'y', 'z']
    colors = ['red', 'green', 'blue']
    for i in range(3):
        ax1.plot(t, ref[:, i], label=f'Vicon {labels[i]}', linewidth=1.2, color=colors[i])
        ax1.plot(t, ekf_cf[:, i], label=f'EKF_CF {labels[i]}', linewidth=1.0, 
                linestyle='--', color=colors[i], alpha=0.7)
    ax1.set_ylabel('Position [m]', fontsize=PLOT_FONT_SIZE+1)
    ax1.set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE+1)
    ax1.set_title('Trajectories Over Time', fontsize=PLOT_FONT_SIZE+2, pad=12)
    ax1.grid(True, which='both', linestyle=':')
    ax1.legend(fontsize=PLOT_FONT_SIZE-1, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    
    # 2. 3D Trajectory plot (top center)
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    ax2.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'b-', linewidth=2.5, label='Vicon', alpha=0.8)
    ax2.plot(ekf_cf[:, 0], ekf_cf[:, 1], ekf_cf[:, 2], 'r--', linewidth=2, label='EKF_CF', alpha=0.8)
    
    # Plot anchors if available
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax2.scatter(pos[0], pos[1], pos[2], color=ANCHOR_COLORS[i], marker='*', 
                           s=150, label=name, edgecolors='k')
    
    ax2.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE, labelpad=8)
    ax2.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE, labelpad=8)
    ax2.set_zlabel('Z [m]', fontsize=PLOT_FONT_SIZE, labelpad=8)
    ax2.set_title('3D Trajectory', fontsize=PLOT_FONT_SIZE+2, pad=20)
    ax2.legend(fontsize=PLOT_FONT_SIZE-2, loc='upper left', bbox_to_anchor=(0, 1))
    ax2.grid(True, alpha=0.3)
    
    # 3. XY Trajectory plot (top right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=2.5, label='Vicon', alpha=0.8)
    ax3.plot(ekf_cf[:, 0], ekf_cf[:, 1], 'r--', linewidth=2, label='EKF_CF', alpha=0.8)
    
    # Plot anchors if available (XY projection)
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4] if len(anchors[name]) > 0 else None
            if pos is not None:
                ax3.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                           s=150, label=name, edgecolors='k')
    
    # Mark start and end points
    if len(ref) > 0:
        ax3.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=100, 
                   label='Start', edgecolors='k', zorder=10)
        ax3.scatter(ref[-1, 0], ref[-1, 1], color='red', marker='s', s=100, 
                   label='End', edgecolors='k', zorder=10)
    
    ax3.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE+1)
    ax3.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE+1)
    ax3.set_title('XY Trajectory', fontsize=PLOT_FONT_SIZE+2, pad=12)
    ax3.grid(True)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend(fontsize=PLOT_FONT_SIZE-2, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    
    # 4. Error plots (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    error_labels = ['X Error', 'Y Error', 'Z Error']
    error_colors = ['red', 'green', 'blue']
    
    # Plot individual axis errors
    for i in range(3):
        ax4.plot(t, err_ekf_cf[:, i], color=error_colors[i], linewidth=1.5, 
                alpha=0.8, label=f'{error_labels[i]}')
    
    # Plot total error with thicker line
    total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
    ax4.plot(t, total_err_ekf_cf, 'k-', linewidth=2, label='Total Error')
    
    ax4.set_ylabel('Error [m]', fontsize=PLOT_FONT_SIZE+1)
    ax4.set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE+1)
    ax4.set_title('All Errors', fontsize=PLOT_FONT_SIZE+2, pad=12)
    ax4.grid(True, linestyle=':')
    ax4.legend(fontsize=PLOT_FONT_SIZE-1, ncol=2)
    
    # Set Y limits to accommodate both individual and total errors
    max_individual_error = np.max(np.abs(err_ekf_cf))
    max_total_error = np.max(total_err_ekf_cf)
    y_limit = max(max_individual_error, max_total_error) * 1.1
    ax4.set_ylim(-y_limit, y_limit)
    
    # 5. Voltage plot or Metrics (bottom center and right)
    if voltage_data.size > 0:
        ax5 = plt.subplot(2, 3, (5, 6))  # Span two columns
        times = voltage_data[:, 0]
        times = times - times[0] if len(times) > 0 else times
        
        colors = ['black', 'gold', 'gray', 'red']
        anchor_names = ['Nero', 'Giallo', 'Grigio', 'Rosso']
        
        for i in range(4):
            if voltage_data.shape[1] > i + 1:
                voltages = voltage_data[:, i + 1]
                ax5.plot(times, voltages, color=colors[i], linewidth=2.5, 
                        label=f'Anchor {anchor_names[i]}', alpha=0.8, marker='o', markersize=2)
        
        ax5.set_xlabel('Time [s]', fontsize=PLOT_FONT_SIZE+1)
        ax5.set_ylabel('Voltage [V]', fontsize=PLOT_FONT_SIZE+1)
        ax5.set_title('Anchor Voltages Over Time', fontsize=PLOT_FONT_SIZE+2, pad=12)
        ax5.grid(True, linestyle=':', alpha=0.7)
        ax5.legend(fontsize=PLOT_FONT_SIZE, loc='upper right')
    else:
        # If no voltage data, show metrics summary
        ax5 = plt.subplot(2, 3, (5, 6))
        ax5.axis('off')
        
        # Create metrics display
        metrics_text = f"""ANALYSIS SUMMARY

RMSE Total: {rmse_ekf_cf_tot:.4f} m ({rmse_ekf_cf_tot*100:.2f} cm)
StdDev Total: {stddev_ekf_cf_tot:.4f} m ({stddev_ekf_cf_tot*100:.2f} cm)

RMSE per axis:
  X: {rmse_ekf_cf_axis[0]*100:.2f} cm    Y: {rmse_ekf_cf_axis[1]*100:.2f} cm    Z: {rmse_ekf_cf_axis[2]*100:.2f} cm

StdDev per axis:
  X: {stddev_ekf_cf_axis[0]*100:.2f} cm    Y: {stddev_ekf_cf_axis[1]*100:.2f} cm    Z: {stddev_ekf_cf_axis[2]*100:.2f} cm

Data points: {len(t)}    Duration: {t[-1] - t[0]:.2f} s"""
        
        ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.90, 
                       wspace=0.3, hspace=0.4)
    
    # Save the comprehensive figure
    comprehensive_path = bag_out_dir / f"{bag_dir.name}_comprehensive.pdf"
    plt.savefig(comprehensive_path, bbox_inches='tight')
    plt.close()


def worker(bag_dir, out_root):
    """Worker function for multiprocessing batch analysis."""
    try:
        csv, plot, r_ekf_cf, std_ekf_cf, bag_out_dir = analyse_bag(bag_dir, out_root)
        return (bag_dir.name, r_ekf_cf, std_ekf_cf, str(csv), str(plot), str(bag_out_dir))
    except Exception as e:
        return (bag_dir.name, 'ERROR', 'ERROR', str(e), '', '')
