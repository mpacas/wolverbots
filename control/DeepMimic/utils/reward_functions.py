"""
Additional reward functions for evaluating gait quality.
"""

import numpy as np

def compute_gait_quality(data, model):
    """
    Compute metrics for gait quality.
    Used for logging during training.
    """
    metrics = {}
    
    # Check symmetry between left and right legs
    try:
        right_knee = data.qpos[model.joint_name2id('right knee')]
        left_knee = data.qpos[model.joint_name2id('left knee')]
        metrics['symmetry'] = 1.0 - abs(right_knee + left_knee) / 2.0
    except:
        metrics['symmetry'] = 0.0
    
    # Check stability (low angular velocity)
    angular_vel = np.linalg.norm(data.qvel[3:6])
    metrics['stability'] = np.exp(-angular_vel)
    
    # Forward progress
    metrics['forward_speed'] = data.qvel[0]
    
    # Energy efficiency (lower is better)
    metrics['energy'] = -np.sum(np.abs(data.ctrl))
    
    return metrics