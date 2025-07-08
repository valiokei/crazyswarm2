#!/usr/bin/env python3
"""
config.py
---------
Configuration constants and settings for ROS bag analysis.
"""

# ROS Topics
TF_TOPIC = '/tf'
VOLTAGE_TOPIC = '/cf21/VoltagesCalibrated'

# Frame names for transformations
CH_REF_CHILD = 'vicon/magnetic_drone/magnetic_drone'
CH_REF_PARENT = 'vicon/world'
CH_EKF_CF_CHILD = 'EKF_CF'
CH_EKF_CF_PARENT = 'vicon/world'

DOG_CHILD = 'vicon/robodog_magnetic/robodog_magnetic'
DOG_PARENT = 'vicon/world'

# Anchor configuration
ANCHOR_FRAMES = ['Nero', 'Giallo', 'Grigio', 'Rosso']
ANCHOR_COLORS = ['black', 'gold', 'gray', 'red']
ANCHOR_PARENT = 'world'

# Analysis parameters
SYNC_EPSILON = 0.01  # [s] max gap between reference & estimates

# Plot configuration
PLOT_FONT_SIZE = 9
DPI = 300

# Plot axis limits for uniformity
ERROR_PLOT_Y_LIM = (-0.7, 0.7)    # Y-axis limits for error plots per axis
ERROR_TOTAL_Y_LIM = (0.0, 1.0)     # Y-axis limits for total error plot
TRAJ_3D_X_LIM = (-3.0, 3.0)        # X-axis limits for 3D trajectory plots
TRAJ_3D_Y_LIM = (-3.0, 3.0)        # Y-axis limits for 3D trajectory plots
TRAJ_3D_Z_LIM = (0.0, 2.5)         # Z-axis limits for 3D trajectory plots

# RoboDog plot axis limits (potentially larger area)
DOG_TRAJ_3D_X_LIM = (-5.0, 5.0)    # X-axis limits for robodog 3D trajectory plots
DOG_TRAJ_3D_Y_LIM = (-5.0, 5.0)    # Y-axis limits for robodog 3D trajectory plots
DOG_TRAJ_3D_Z_LIM = (0.0, 3.0)     # Z-axis limits for robodog 3D trajectory plots

# Robot physical dimensions (for 3D visualization)
# Unitree Go2 quadruped robot dimensions (official specifications)
ROBODOG_LENGTH = 0.845  # [m] length along X-axis (84.5 cm)
ROBODOG_WIDTH = 0.290   # [m] width along Y-axis (29.0 cm)
ROBODOG_HEIGHT = 0.400  # [m] height along Z-axis (40.0 cm when standing)
ROBODOG_FRAME_HEIGHT = 0.15  # [m] height from ground to frame on back (approximate)

# Crazyflie 2.1 dimensions (official specifications)
CRAZYFLIE_DIAMETER = 0.092  # [m] motor-to-motor diameter (9.2 cm)
CRAZYFLIE_HEIGHT = 0.029    # [m] total height with propellers (2.9 cm)

# 3D Snapshot configuration
SNAPSHOT_POSITIONS = ['start', 'middle', 'end']  # Positions for trajectory snapshots
