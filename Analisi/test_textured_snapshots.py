#!/usr/bin/env python3
"""
Test script per verificare la funzionalità di snapshot 3D con texture.
Questo script crea dati di test e genera entrambe le versioni degli snapshot.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Aggiungi il percorso del modulo Analisi
sys.path.append(str(Path(__file__).parent))

from plotting import create_3d_robot_snapshots, create_3d_robot_snapshots_with_textures, create_3d_robot_snapshots_enhanced

def create_test_data():
    """Crea dati di test per simulare una traiettoria."""
    n_points = 100
    t = np.linspace(0, 10, n_points)
    
    # Traiettoria circolare per il drone
    radius = 1.0
    ref_pos = np.column_stack([
        radius * np.cos(0.5 * t),
        radius * np.sin(0.5 * t), 
        1.0 + 0.3 * np.sin(t)
    ])
    
    # EKF con rumore
    noise = 0.05 * np.random.randn(n_points, 3)
    ekf_pos = ref_pos + noise
    
    # RoboDog che segue il drone
    dog_pos = np.column_stack([
        0.8 * radius * np.cos(0.5 * t),
        0.8 * radius * np.sin(0.5 * t),
        0.5 * np.ones(n_points)  # Altezza costante del frame del RoboDog
    ])
    
    # Quaternioni di test (orientamento dinamico)
    ref_quat = np.column_stack([
        0.1 * np.sin(t), 0.1 * np.cos(t), np.zeros(n_points), 
        np.ones(n_points)  # w component
    ])
    ekf_quat = ref_quat + 0.02 * np.random.randn(n_points, 4)
    dog_quat = np.column_stack([
        np.zeros(n_points), np.zeros(n_points), 0.2 * np.sin(0.3 * t),
        np.ones(n_points)
    ])
    
    # Normalizza i quaternioni
    for quat in [ref_quat, ekf_quat, dog_quat]:
        for i in range(len(quat)):
            quat[i] = quat[i] / np.linalg.norm(quat[i])
    
    return t, ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat

def test_snapshots():
    """Test delle funzioni di snapshot 3D."""
    print("Creazione dati di test...")
    t, ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat = create_test_data()
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    print("Test 1: Snapshot geometrici...")
    try:
        fig1 = create_3d_robot_snapshots(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t)
        fig1.savefig(output_dir / "test_geometric_snapshots.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("✓ Snapshot geometrici creati con successo!")
    except Exception as e:
        print(f"✗ Errore negli snapshot geometrici: {e}")
    
    print("Test 2: Snapshot con texture...")
    try:
        fig2 = create_3d_robot_snapshots_with_textures(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t)
        fig2.savefig(output_dir / "test_textured_snapshots.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("✓ Snapshot con texture creati con successo!")
    except Exception as e:
        print(f"✗ Errore negli snapshot con texture: {e}")
    
    print("Test 3: Funzione enhanced (geometrica)...")
    try:
        fig3 = create_3d_robot_snapshots_enhanced(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t, 
                                                 title="Test Enhanced - Geometric", use_textures=False)
        fig3.savefig(output_dir / "test_enhanced_geometric.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("✓ Snapshot enhanced (geometrici) creati con successo!")
    except Exception as e:
        print(f"✗ Errore negli snapshot enhanced (geometrici): {e}")
    
    print("Test 4: Funzione enhanced (texture)...")
    try:
        fig4 = create_3d_robot_snapshots_enhanced(ref_pos, ekf_pos, dog_pos, ref_quat, ekf_quat, dog_quat, t,
                                                 title="Test Enhanced - Textured", use_textures=True)
        fig4.savefig(output_dir / "test_enhanced_textured.png", dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("✓ Snapshot enhanced (texture) creati con successo!")
    except Exception as e:
        print(f"✗ Errore negli snapshot enhanced (texture): {e}")
    
    print(f"\nTutti i file di test sono stati salvati in: {output_dir.absolute()}")
    print("Puoi aprire le immagini per verificare la qualità delle visualizzazioni.")

if __name__ == "__main__":
    test_snapshots()
