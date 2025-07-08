#!/usr/bin/env python3
"""
data_processing.py
-----------------
Functions for processing and analyzing extracted data.
"""

import numpy as np
try:
    from .config import SYNC_EPSILON
except ImportError:
    from config import SYNC_EPSILON


def apply_static_transform(streams, anchors, voltage_data):
    """
    Apply static transformations if needed.
    Currently disabled since all frames are already in vicon/world.
    """
    # All frames are already in vicon/world, no transformation needed
    return streams, anchors, voltage_data


def align_streams(streams):
    """
    Align reference and estimation streams using nearest neighbor interpolation.
    
    Args:
        streams (dict): Dictionary containing pose streams
        
    Returns:
        tuple: (times, ref_xyz, ekf_cf_xyz) aligned arrays
    """
    if streams['ekf_cf'].size == 0 or streams['ref'].size == 0:
        raise RuntimeError("Missing one or more streams")
    
    base_t = streams['ekf_cf'][:, 0]   # timestamps of EKF_CF
    ref_xyz = np.zeros((len(base_t), 3))
    ref_ok = np.zeros(len(base_t), dtype=bool)

    # index pointers
    idx_ref = 0
    ref_arr = streams['ref']

    for i, t in enumerate(base_t):
        # REF - find nearest neighbor within epsilon
        while idx_ref + 1 < len(ref_arr) and ref_arr[idx_ref + 1, 0] <= t:
            idx_ref += 1
        if abs(ref_arr[idx_ref, 0] - t) <= SYNC_EPSILON:
            ref_xyz[i] = ref_arr[idx_ref, 1:4]
            ref_ok[i] = True

    mask = ref_ok
    t_out = base_t[mask]
    ekf_cf_xyz = streams['ekf_cf'][mask][:, 1:4]
    ref_xyz = ref_xyz[mask]
    return t_out, ref_xyz, ekf_cf_xyz


def compute_rmse(ref, est):
    """
    Compute Root Mean Square Error between reference and estimate.
    
    Args:
        ref (np.ndarray): Reference trajectory (N, 3)
        est (np.ndarray): Estimated trajectory (N, 3)
        
    Returns:
        tuple: (rmse_per_axis, rmse_total, error_array)
    """
    err = est - ref
    rmse = np.sqrt(np.mean(err**2, axis=0))
    rmse_total = np.sqrt(np.mean(err**2))
    return rmse, rmse_total, err


def compute_stddev(err):
    """
    Compute standard deviation of the error.
    
    Args:
        err (np.ndarray): Error array (N, 3)
        
    Returns:
        tuple: (stddev_per_axis, stddev_total)
    """
    stddev_axis = np.std(err, axis=0)
    stddev_total = np.std(np.sqrt(np.sum(err**2, axis=1)))
    return stddev_axis, stddev_total


def align_streams_with_orientation(streams):
    """
    Align reference and estimation streams including orientation data.
    
    Args:
        streams (dict): Dictionary containing pose streams with quaternions
        
    Returns:
        tuple: (times, ref_pose, ekf_cf_pose) where poses include [x,y,z,qx,qy,qz,qw]
    """
    if streams['ekf_cf'].size == 0 or streams['ref'].size == 0:
        raise RuntimeError("Missing one or more streams")
    
    ekf_cf_data = streams['ekf_cf']
    ref_data = streams['ref']
    
    # Determine data format
    ekf_cf_has_quat = ekf_cf_data.shape[1] >= 8
    ref_has_quat = ref_data.shape[1] >= 8
    
    base_t = ekf_cf_data[:, 0]   # timestamps of EKF_CF
    
    # Initialize output arrays
    pose_size = 7 if (ekf_cf_has_quat and ref_has_quat) else 3
    ref_pose = np.zeros((len(base_t), pose_size))
    ref_ok = np.zeros(len(base_t), dtype=bool)

    # index pointers
    idx_ref = 0

    for i, t in enumerate(base_t):
        # REF - find nearest neighbor within epsilon
        while idx_ref + 1 < len(ref_data) and ref_data[idx_ref + 1, 0] <= t:
            idx_ref += 1
        if abs(ref_data[idx_ref, 0] - t) <= SYNC_EPSILON:
            if pose_size == 7:
                ref_pose[i] = ref_data[idx_ref, 1:8]  # position + quaternion
            else:
                ref_pose[i] = ref_data[idx_ref, 1:4]  # position only
            ref_ok[i] = True

    mask = ref_ok
    t_out = base_t[mask]
    
    if pose_size == 7:
        ekf_cf_pose = ekf_cf_data[mask][:, 1:8]  # position + quaternion
    else:
        ekf_cf_pose = ekf_cf_data[mask][:, 1:4]  # position only
    
    ref_pose = ref_pose[mask]
    
    return t_out, ref_pose, ekf_cf_pose
