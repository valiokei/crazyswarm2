#!/usr/bin/env python3
"""
analyze_rosbags_refactored.py
----------------------------
Refactored version of analyze_rosbags_v2.py with improved organization.

Analizza ROS 2 Jazzy rosbags per calcolare RMSE tra:
  • Posa di riferimento Vicon  :  child = vicon/magnetic_drone/magnetic_drone, parent = vicon/world
  • Stimatore EKF_CF           :  child = EKF_CF,                    parent = vicon/world

Lo script supporta due modalità:

MODALITÀ BATCH:
  1. Legge un file Excel con metadati che elenca tutti i test e le cartelle rosbag associate.
  2. Itera (multiprocesso) su ogni bag ed estrae i topic `/tf` e `/cf21/VoltagesCalibrated`.
  3. Sincronizza i flussi di pose (nearest‑neighbour entro --sync-epsilon s).
  4. Calcola RMSE & MAE per asse e totale, li salva in CSV.
  5. Genera grafici publication‑ready (conformi IEEE TRO) in PNG & PDF.
  6. Crea grafici traiettoria 3D con posizioni degli anchor.
  7. Visualizza errori puntuali lungo la traiettoria.
  8. Crea grafici delle tensioni per tutti e 4 gli anchor nel tempo.
  9. Calcola metriche di deviazione standard.
  10. Genera figura riassuntiva con i migliori punteggi per categoria inclusi i grafici di tensione.
  11. Crea snapshot 3D con forme geometriche e texture dei robot (RoboDog e Crazyflie) all'inizio, metà e fine traiettoria.

MODALITÀ SINGOLA:
  Analizza una singola rosbag specificata con --single, generando tutti i grafici e metriche.
  Include automaticamente la generazione di snapshot 3D che mostrano:
  - RoboDog rappresentato come parallelepipedo con dimensioni realistiche Unitree Go2 (84.5×29.0×40.0 cm)
  - Crazyflie riferimento (Vicon) come sfera blu (diametro 9.2 cm) 
  - Crazyflie stimato (EKF) come sfera rossa (diametro 9.2 cm)
  - Versione avanzata con texture procedurali per maggior realismo
  - Linea di distanza tra riferimento e stima con valore numerico
  - Terne di riferimento coordinate (X-rosso, Y-verde, Z-blu) per:
    * Frame mondo (World) all'origine
    * Frame Vicon di riferimento 
    * Frame EKF stimato
    * Frame RoboDog (se disponibile)
  - Orientamento dinamico del RoboDog basato sulla sua tf reale (seduto/alzato/orientamento)

Note: Questa versione analizza solo EKF_CF vs Vicon (nessuna trasformazione cf21 o Opt).

Requirements:
  sudo apt install python3-rosbag2-py python3-rclpy python3-numpy python3-pandas python3-matplotlib python3-scipy
  pip install tqdm

Usage example
-------------

# Modalità batch - da dentro la cartella con le cartelle dei rosbags
cd Analisi/ && python analyze_rosbags_refactored.py --excel ETH.xlsx --root /home/valiokei/GitHub/ros_ws/src/crazyswarm2/rosbags --out /home/valiokei/GitHub/ros_ws/src/crazyswarm2/rosbags/test_results --workers 10

# Modalità singola rosbag
python analyze_rosbags_refactored.py --single /path/to/rosbag2_folder --out results

# per mandare i risultati sulla cartella OneDrive condivisa (modalità batch)
python analyze_rosbags_refactored.py --excel '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/ROSBAGS/ETH.xlsx' --root '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/ROSBAGS' --out '/mnt/c/Users/valio/OneDrive - Università degli Studi di Perugia (1)/ETH_Valerio_Magnetico/risultati' --workers 4
"""

import argparse
import sys
import pathlib
import multiprocessing as mp
from functools import partial
import pandas as pd
from tqdm import tqdm

# Import the refactored modules
try:
    from .summary_reports import (
        read_excel_list, create_category_summary_figure, 
        create_summary_figure, create_rmse_markdown_table
    )
    from .analysis import analyse_bag_with_display, worker
except ImportError:
    from summary_reports import (
        read_excel_list, create_category_summary_figure, 
        create_summary_figure, create_rmse_markdown_table
    )
    from analysis import analyse_bag_with_display, worker


def main():
    parser = argparse.ArgumentParser(description='ROS2 bag analyser - Batch or Single mode (Refactored)')
    
    # Create mutually exclusive group for batch vs single mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--excel', help='Excel list of tests (batch mode)')
    mode_group.add_argument('--single', help='Single rosbag directory path')
    
    # Common arguments
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count()-1), 
                       help='Number of worker processes (batch mode only)')
    
    # Batch mode specific arguments
    parser.add_argument('--root', help='Root folder containing rosbag dirs (required for batch mode)')
    
    args = parser.parse_args()

    # Validate arguments based on mode
    if args.excel and not args.root:
        parser.error("--root is required when using --excel (batch mode)")
    
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        # Single rosbag mode
        print("Running in SINGLE rosbag mode (Refactored)")
        single_bag_path = pathlib.Path(args.single).expanduser()
        
        if not single_bag_path.exists():
            print(f"ERROR: Rosbag directory {single_bag_path} not found")
            sys.exit(1)
            
        if not single_bag_path.is_dir():
            print(f"ERROR: {single_bag_path} is not a directory")
            sys.exit(1)
            
        print(f"Analyzing single rosbag: {single_bag_path}")
        
        try:
            csv_path, plot_path, rmse_total, stddev_total, bag_out_dir = analyse_bag_with_display(single_bag_path, out_dir)
            
            print(f"Analysis completed!")
            print(f"RMSE Total: {rmse_total:.4f} m ({rmse_total*100:.2f} cm)")
            print(f"StdDev Total: {stddev_total:.4f} m ({stddev_total*100:.2f} cm)")
            print(f"Results saved in: {bag_out_dir}")
            print(f"CSV metrics: {csv_path}")
            print(f"Plots saved in: {bag_out_dir}")
            print("✓ 3D Robot snapshots (geometric + textured) included in analysis output!")
            print("All plots have been displayed and saved!")
            
        except Exception as e:
            print(f"ERROR analyzing rosbag: {e}")
            sys.exit(1)
            
    else:
        # Batch mode
        print("Running in BATCH mode (Refactored)")
        excel_path = pathlib.Path(args.excel).expanduser()
        root_dir = pathlib.Path(args.root).expanduser()

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
                               columns=['bag', 'RMSE_ekf_cf', 'StdDev_ekf_cf', 'csv', 'plot', 'bag_dir'])
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
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
