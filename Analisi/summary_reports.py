#!/usr/bin/env python3
"""
summary_reports.py
-----------------
Functions for creating summary reports and figures from batch analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .rosbag_reader import extract_streams
    from .data_processing import apply_static_transform, align_streams, compute_rmse
    from .config import (
        PLOT_FONT_SIZE, DPI, ERROR_PLOT_Y_LIM, ERROR_TOTAL_Y_LIM,
        ANCHOR_FRAMES, ANCHOR_COLORS
    )
except ImportError:
    from rosbag_reader import extract_streams
    from data_processing import apply_static_transform, align_streams, compute_rmse
    from config import (
        PLOT_FONT_SIZE, DPI, ERROR_PLOT_Y_LIM, ERROR_TOTAL_Y_LIM,
        ANCHOR_FRAMES, ANCHOR_COLORS
    )


def read_excel_list(excel_path: str):
    """Return dataframe with columns category, label, rosbags(list)."""
    df = pd.read_excel(excel_path, sheet_name=0)
    header_row = df.iloc[0, 2:].reset_index(drop=True)
    data = df.iloc[1:].reset_index(drop=True)

    categories = data.iloc[:, 0].fillna(method='ffill')
    labels = data.iloc[:, 1]
    bag_cols = data.iloc[:, 2:]

    bag_lists = []
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
                        'label': labels,
                        'rosbags': bag_lists})
    return out


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
    
    # Create figure with subplots: 6 columns per category
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
        
        # Create plots for this category
        _create_category_plots(fig, cat_idx, len(categories), category, best_bag, best_rmse_cm,
                              t, ref, ekf_cf, err_ekf_cf, anchors, voltage_data)
    
    plt.tight_layout()
    summary_path = out_dir / "category_summary_comprehensive.pdf"
    plt.savefig(summary_path, bbox_inches='tight')
    
    # Also save as PNG for easy viewing
    summary_png_path = out_dir / "category_summary_comprehensive.png"
    plt.savefig(summary_png_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    return summary_path


def _create_category_plots(fig, cat_idx, n_categories, category, best_bag, best_rmse_cm,
                          t, ref, ekf_cf, err_ekf_cf, anchors, voltage_data):
    """Create plots for a single category in the summary figure."""
    # 3D Trajectory plot with RMSE in title
    ax1 = fig.add_subplot(n_categories, 6, cat_idx*6 + 1, projection='3d')
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
    ax1.set_title(f'{category} - 3D\nRMSE: {best_rmse_cm:.2f} cm\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
    if cat_idx == 0:
        ax1.legend(fontsize=PLOT_FONT_SIZE-2, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # XY Trajectory plot
    ax2 = fig.add_subplot(n_categories, 6, cat_idx*6 + 2)
    ax2.plot(ref[:, 0], ref[:, 1], 'b-', linewidth=2, label='Vicon', alpha=0.8)
    ax2.plot(ekf_cf[:, 0], ekf_cf[:, 1], 'r--', linewidth=1.5, label='EKF_CF', alpha=0.8)
    
    # Plot anchors and start/end points
    for i, name in enumerate(ANCHOR_FRAMES):
        if anchors[name].size > 0:
            pos = anchors[name][-1, 1:4]
            ax2.scatter(pos[0], pos[1], color=ANCHOR_COLORS[i], marker='*', 
                       s=60, label=name, edgecolors='k', alpha=0.8)
    
    if len(ref) > 0:
        ax2.scatter(ref[0, 0], ref[0, 1], color='green', marker='o', s=100, 
                   label='Start', edgecolors='k', zorder=10)
        ax2.scatter(ref[-1, 0], ref[-1, 1], color='red', marker='s', s=100, 
                   label='End', edgecolors='k', zorder=10)
    
    ax2.set_xlabel('X [m]', fontsize=PLOT_FONT_SIZE-1)
    ax2.set_ylabel('Y [m]', fontsize=PLOT_FONT_SIZE-1)
    ax2.set_title(f'{category} - XY\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
    ax2.grid(True, alpha=0.7)
    ax2.set_aspect('equal', adjustable='box')
    
    # XZ Trajectory plot
    ax3 = fig.add_subplot(n_categories, 6, cat_idx*6 + 3)
    ax3.plot(ref[:, 0], ref[:, 2], 'b-', linewidth=2, label='Vicon', alpha=0.8)
    ax3.plot(ekf_cf[:, 0], ekf_cf[:, 2], 'r--', linewidth=1.5, label='EKF_CF', alpha=0.8)
    
    # Plot anchors (XZ projection)
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
    ax4 = fig.add_subplot(n_categories, 6, cat_idx*6 + 4)
    total_err_ekf_cf = np.sqrt(np.sum(err_ekf_cf**2, axis=1))
    ax4.plot(t, total_err_ekf_cf, 'r-', linewidth=1.5, label='EKF_CF Total Error', alpha=0.8)
    ax4.set_ylabel('Errore totale [m]', fontsize=PLOT_FONT_SIZE-1)
    ax4.set_xlabel('Tempo [s]', fontsize=PLOT_FONT_SIZE-1)
    ax4.set_ylim(ERROR_TOTAL_Y_LIM)
    ax4.grid(True, linestyle=':', alpha=0.7)
    if cat_idx == 0:
        ax4.legend(fontsize=PLOT_FONT_SIZE-2)
    
    # XYZ Error components plot
    ax5 = fig.add_subplot(n_categories, 6, cat_idx*6 + 5)
    labels = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    for i in range(3):
        ax5.plot(t, err_ekf_cf[:, i], color=colors[i], linewidth=1.2, 
                label=f'Errore {labels[i]}', alpha=0.8)
    ax5.set_ylabel('Errore [m]', fontsize=PLOT_FONT_SIZE-1)
    ax5.set_xlabel('Tempo [s]', fontsize=PLOT_FONT_SIZE-1)
    ax5.set_ylim(ERROR_PLOT_Y_LIM)
    ax5.set_title(f'{category} - Errori XYZ\n({best_bag["label"]})', fontsize=PLOT_FONT_SIZE)
    ax5.grid(True, linestyle=':', alpha=0.7)
    if cat_idx == 0:
        ax5.legend(fontsize=PLOT_FONT_SIZE-2)
    
    # Voltage plot
    ax6 = fig.add_subplot(n_categories, 6, cat_idx*6 + 6)
    if voltage_data.size > 0:
        v_times = voltage_data[:, 0]
        v_times = v_times - v_times[0] if len(v_times) > 0 else v_times
        
        colors_v = ['black', 'gold', 'gray', 'red']
        anchor_names = ['Nero', 'Giallo', 'Grigio', 'Rosso']
        
        for i in range(4):
            if voltage_data.shape[1] > i + 1:
                v_voltages = voltage_data[:, i + 1]
                ax6.plot(v_times, v_voltages, color=colors_v[i], linewidth=1.2, 
                        label=anchor_names[i], alpha=0.8)
        
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
