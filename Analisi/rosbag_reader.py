#!/usr/bin/env python3
"""
rosbag_reader.py
---------------
Functions for reading and extracting data from ROS bags.
"""

import pathlib
import numpy as np
import yaml
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

try:
    from .config import (
        TF_TOPIC, VOLTAGE_TOPIC, CH_REF_CHILD, CH_REF_PARENT,
        CH_EKF_CF_CHILD, CH_EKF_CF_PARENT, DOG_CHILD, DOG_PARENT,
        ANCHOR_FRAMES, ANCHOR_PARENT
    )
except ImportError:
    from config import (
        TF_TOPIC, VOLTAGE_TOPIC, CH_REF_CHILD, CH_REF_PARENT,
        CH_EKF_CF_CHILD, CH_EKF_CF_PARENT, DOG_CHILD, DOG_PARENT,
        ANCHOR_FRAMES, ANCHOR_PARENT
    )


def detect_storage_id(bag_dir: Path) -> str:
    """Detect the storage identifier from bag metadata."""
    meta = bag_dir / 'metadata.yaml'
    if meta.exists():
        with open(meta, 'r') as f:
            return yaml.safe_load(f).get('storage_identifier', '')
    return ''


def rosbag_iter_topics(bag_path: pathlib.Path, topics):
    """Yield (topic, time_sec, msg) for specified topics."""
    reader = SequentialReader()
    
    sid = detect_storage_id(bag_path)
    
    storage_options = StorageOptions(uri=str(bag_path), storage_id=sid)
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


def rosbag_iter_tf(bag_path: pathlib.Path):
    """Yield (time_sec, transform_msg) only for TF_TOPIC."""
    for topic, stamp_sec, msg in rosbag_iter_topics(bag_path, [TF_TOPIC]):
        if topic == TF_TOPIC:
            for tf in msg.transforms:
                yield stamp_sec, tf


def extract_streams(bag_dir: pathlib.Path):
    """
    Extract pose streams, anchor positions, and voltage data from a rosbag.
    
    Returns:
        streams (dict): Dictionary with 'ref', 'ekf_cf', 'dog' pose streams
        anchors (dict): Dictionary with anchor positions
        voltage_data (np.ndarray): Voltage data array
    """
    streams = {'ref': [], 'ekf_cf': [], 'dog': []}
    anchors = {name: [] for name in ANCHOR_FRAMES}
    
    for t, tf in rosbag_iter_tf(bag_dir):
        child = tf.child_frame_id
        parent = tf.header.frame_id
        
        if child == CH_REF_CHILD and parent == CH_REF_PARENT:
            streams['ref'].append((t,
                                   tf.transform.translation.x,
                                   tf.transform.translation.y,
                                   tf.transform.translation.z,
                                   tf.transform.rotation.x,
                                   tf.transform.rotation.y,
                                   tf.transform.rotation.z,
                                   tf.transform.rotation.w))
        elif child == CH_EKF_CF_CHILD and parent == CH_EKF_CF_PARENT:
            streams['ekf_cf'].append((t,
                                     tf.transform.translation.x,
                                     tf.transform.translation.y,
                                     tf.transform.translation.z,
                                     tf.transform.rotation.x,
                                     tf.transform.rotation.y,
                                     tf.transform.rotation.z,
                                     tf.transform.rotation.w))
        elif child == DOG_CHILD and parent == DOG_PARENT:
            streams['dog'].append((t,
                                  tf.transform.translation.x,
                                  tf.transform.translation.y,
                                  tf.transform.translation.z,
                                  tf.transform.rotation.x,
                                  tf.transform.rotation.y,
                                  tf.transform.rotation.z,
                                  tf.transform.rotation.w))
        # Capture anchor positions
        elif child in ANCHOR_FRAMES and parent == ANCHOR_PARENT:
            anchors[child].append((t,
                                  tf.transform.translation.x,
                                  tf.transform.translation.y,
                                  tf.transform.translation.z))

    # Convert lists to numpy arrays
    for k in streams:
        if streams[k]:
            streams[k] = np.asarray(streams[k])
        else:
            streams[k] = np.empty((0, 4))
    
    for k in anchors:
        if anchors[k]:
            anchors[k] = np.asarray(anchors[k])
        else:
            anchors[k] = np.empty((0, 4))
    
    # Extract voltage data
    voltage_data = extract_voltage_data(bag_dir)
    
    return streams, anchors, voltage_data


def extract_voltage_data(bag_dir: pathlib.Path):
    """Extract voltage data from the VOLTAGE_TOPIC with 4 anchor values."""
    voltage_data = []
    
    for topic, stamp_sec, msg in rosbag_iter_topics(bag_dir, [VOLTAGE_TOPIC]):
        if topic == VOLTAGE_TOPIC:
            try:
                voltages = None
                
                # Try to extract the 4 voltage values for the anchors
                if hasattr(msg, 'data') and hasattr(msg.data, '__len__') and len(msg.data) == 4:
                    voltages = list(msg.data)
                elif hasattr(msg, 'voltages') and hasattr(msg.voltages, '__len__') and len(msg.voltages) == 4:
                    voltages = list(msg.voltages)
                elif hasattr(msg, 'voltage') and hasattr(msg.voltage, '__len__') and len(msg.voltage) == 4:
                    voltages = list(msg.voltage)
                else:
                    # Try to find any array field with 4 elements
                    for field_name in dir(msg):
                        if not field_name.startswith('_'):
                            field_value = getattr(msg, field_name)
                            if hasattr(field_value, '__len__') and len(field_value) == 4:
                                try:
                                    voltages = [float(v) for v in field_value]
                                    break
                                except (ValueError, TypeError):
                                    continue
                
                if voltages is not None and len(voltages) == 4:
                    voltage_data.append([stamp_sec] + voltages)
                
            except Exception as e:
                print(f"Warning: Could not extract voltage data from message: {e}")
                continue
    
    if voltage_data:
        return np.array(voltage_data)  # Shape: (n_samples, 5) = [time, v1, v2, v3, v4]
    else:
        return np.empty((0, 5))
