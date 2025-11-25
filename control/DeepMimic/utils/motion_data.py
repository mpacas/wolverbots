"""
Utilities for creating and loading motion reference data.
"""

import numpy as np

def create_motion_dataset(output_file='mocap_data/reference_motions.npz'):
    """
    Create a basic motion dataset for initial testing.
    Replace this with real mocap data later.
    """
    print("WARNING:Creating SYNTHETIC motion dataset...")
    
    # Walking cycle
    walk_frames = 60
    walk_positions = []
    walk_velocities = []
    
    for i in range(walk_frames):
        phase = (i / walk_frames) * 2 * np.pi
        
        # Create a simple walking pattern
        # Adjust these values based on your robot's joint limits
        pos = np.zeros(30)  # Adjust size for your robot
        vel = np.zeros(29)  # Adjust size for your robot
        
        # Basic leg swing pattern
        pos[2] = 0.8  # Height
        pos[9] = 0.3 * np.sin(phase)  # Right hip
        pos[11] = -0.3 * np.sin(phase)  # Right knee
        pos[13] = 0.3 * np.sin(phase + np.pi)  # Left hip
        pos[15] = -0.3 * np.sin(phase + np.pi)  # Left knee
        
        walk_positions.append(pos)
        walk_velocities.append(vel)
    
    # Convert lists to arrays
    walk_positions = np.array(walk_positions)
    walk_velocities = np.array(walk_velocities)
    
    # Standing pose
    stand_pos = np.zeros((1, 30))
    stand_pos[0, 2] = 0.8  # Standing height
    stand_vel = np.zeros((1, 29))
    
    # Save dataset - flatten the structure
    np.savez(output_file,
             walk_positions=walk_positions,
             walk_velocities=walk_velocities,
             walk_length=walk_frames,
             stand_positions=stand_pos,
             stand_velocities=stand_vel,
             stand_length=1)
    
    print(f"Motion dataset saved to {output_file}")
    return output_file