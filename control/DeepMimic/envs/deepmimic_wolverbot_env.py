"""
DeepMimic implementation for your WolverBot
Extends your existing wolverbot_env.py with motion imitation
"""

import numpy as np
import os
import sys
from gymnasium import spaces

# Add parent directory to path to import your existing robot
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wolverbot_env import HumanoidEnv

class DeepMimicWolverBot(HumanoidEnv):
    """
    Your WolverBot with DeepMimic motion imitation.
    This builds directly on your working HumanoidEnv.
    """
    
    def __init__(
        self,
        xml_file='scene.xml',  # Your scene file that includes robot.xml
        reference_motion_file=None,
        imitation_weight=0.7,
        task_weight=0.3,
        enable_command_following=True,
        early_termination_threshold=2.0,
        **kwargs
    ):
        """
        Initialize DeepMimic environment for your robot.
        
        Args:
            xml_file: Your scene.xml file
            reference_motion_file: Path to motion data or None for synthetic
            imitation_weight: Weight for motion imitation reward (0.7 recommended)
            task_weight: Weight for task completion reward
            enable_command_following: Allow velocity commands for A* integration
            early_termination_threshold: Max deviation before episode ends
        """
        
        # Motion tracking setup
        self.reference_motions = {}
        self.current_motion_type = 'walk'
        self.motion_phase = 0.0
        self.phase_speed = 1.0
        
        # Reward weights (will decrease imitation over training)
        self.imitation_weight = imitation_weight
        self.task_weight = task_weight
        
        # Command following for A* path integration
        self.enable_command_following = enable_command_following
        self.target_velocity = np.zeros(3)  # [vx, vy, vtheta]
        
        # Early termination
        self.early_termination_threshold = early_termination_threshold
        
        # Joint mapping for your robot (based on your robot.xml)
        # These are the joints we care about for locomotion
        self.joint_names = [
            'right leg twist', 'right leg lateral', 'right hammy', 
            'right knee', 'right foot', 'right foot twist',
            'left leg twist', 'left leg lateral', 'left hammy',
            'left knee', 'left foot lift', 'left foot twist'
        ]
        
        # Initialize parent environment with your robot
        super().__init__(xml_file=xml_file, **kwargs)

        
        
        # Fix observation space based on ACTUAL observation size
        # Get a dummy observation to find true size
        dummy_qpos = np.zeros(self.model.nq)
        dummy_qvel = np.zeros(self.model.nv)
        self.set_state(dummy_qpos, dummy_qvel)
        actual_obs = self._get_obs()
        actual_obs_dim = actual_obs.shape[0]

        # Now create correct observation space
        if self.enable_command_following:
            total_obs_dim = actual_obs_dim + 3  # Add 3 for velocity commands
        else:
            total_obs_dim = actual_obs_dim

        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # Load reference motions
        self._load_reference_motions(reference_motion_file)
        
        print(f"DeepMimic WolverBot initialized!")
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Action space: {self.action_space.shape}")
        print(f"Tracking joints: {len(self.joint_names)} leg joints")

    def _get_joint_id(self, joint_name):
            """Get joint ID - works with new mujoco"""
            for i in range(self.model.njnt):
                if self.model.joint(i).name == joint_name:
                    return i
            raise ValueError(f"Joint '{joint_name}' not found")
    
    def _load_reference_motions(self, motion_file):
        """Load or generate reference motions"""
        if motion_file and os.path.exists(motion_file):
            # Load real motion data
            print(f"Loading motion data from {motion_file}")
            data = np.load(motion_file)
            
            # Load walk motion
            if 'walk_positions' in data:
                self.reference_motions['walk'] = {
                    'positions': data['walk_positions'],
                    'velocities': data['walk_velocities'],
                    'length': int(data['walk_length'])
                }
            else:
                self.reference_motions['walk'] = self._generate_walk_cycle()
            
            # Load stand motion
            if 'stand_positions' in data:
                self.reference_motions['stand'] = {
                    'positions': data['stand_positions'],
                    'velocities': data['stand_velocities'],
                    'length': int(data['stand_length'])
                }
            else:
                self.reference_motions['stand'] = self._generate_stand_pose()
            
            # Generate kick (not in basic dataset)
            self.reference_motions['kick'] = self._generate_kick_motion()
        else:
            # Generate synthetic motions for initial testing
            print("Generating synthetic reference motions...")
            self.reference_motions = {
                'walk': self._generate_walk_cycle(),
                'stand': self._generate_stand_pose(),
                'kick': self._generate_kick_motion()
            }
    
    def _generate_walk_cycle(self):
        """
        Generate synthetic walking cycle for your robot.
        This creates a basic but functional walking pattern.
        """
        cycle_frames = 60  # 1 second at 60Hz
        motion = {
            'positions': [],
            'velocities': [],
            'length': cycle_frames
        }
        
        for i in range(cycle_frames):
            phase = (i / cycle_frames) * 2 * np.pi
            
            # Create joint positions for walking
            # Using your robot's joint indices
            qpos = np.zeros(self.model.nq)
            qvel = np.zeros(self.model.nv)
            
            # Set base height (keep robot at walking height)
            qpos[2] = 0.8  # z-position
            
            # Hip movements (alternating)
            # Right leg
            qpos[self._get_joint_id('right leg lateral')] = 0.1 * np.sin(phase)
            qpos[self._get_joint_id('right hammy')] = 0.3 * np.sin(phase)
            qpos[self._get_joint_id('right knee')] = -0.3 * np.sin(phase)
            qpos[self._get_joint_id('right foot')] = 0.15 * np.sin(phase)
            
            # Left leg (opposite phase)
            qpos[self._get_joint_id('left leg lateral')] = 0.1 * np.sin(phase + np.pi)
            qpos[self._get_joint_id('left hammy')] = 0.3 * np.sin(phase + np.pi)
            qpos[self._get_joint_id('left knee')] = -0.3 * np.sin(phase + np.pi)
            qpos[self._get_joint_id('left foot lift')] = 0.15 * np.sin(phase + np.pi)
            
            # Add small velocities for smoothness
            if i > 0:
                qvel = (qpos - motion['positions'][-1]) * 60.0  # Approximate velocity
            
            motion['positions'].append(qpos.copy())
            motion['velocities'].append(qvel.copy())
        
        return motion
    
    def _generate_stand_pose(self):
        """Generate standing pose"""
        motion = {
            'positions': [np.zeros(self.model.nq)],
            'velocities': [np.zeros(self.model.nv)],
            'length': 1
        }
        # Set standing height
        motion['positions'][0][2] = 0.8
        return motion
    
    def _generate_kick_motion(self):
        """Generate kicking motion"""
        kick_frames = 30  # 0.5 second kick
        motion = {
            'positions': [],
            'velocities': [],
            'length': kick_frames
        }
        
        for i in range(kick_frames):
            phase = (i / kick_frames) * np.pi  # Half cycle for kick
            
            qpos = np.zeros(self.model.nq)
            qvel = np.zeros(self.model.nv)
            
            # Base position
            qpos[2] = 0.8
            
            # Kicking with right leg
            if i < kick_frames // 2:
                # Wind up
                qpos[self._get_joint_id('right hammy')] = -0.5 * np.sin(phase)
                qpos[self._get_joint_id('right knee')] = -0.8 * np.sin(phase)
            else:
                # Kick
                qpos[self._get_joint_id('right hammy')] = 0.8 * np.sin(phase)
                qpos[self._get_joint_id('right knee')] = 0.3 * np.sin(phase)
            
            # Left leg stable
            qpos[self._get_joint_id('left knee')] = -0.1
            
            if i > 0:
                qvel = (qpos - motion['positions'][-1]) * 60.0
            
            motion['positions'].append(qpos.copy())
            motion['velocities'].append(qvel.copy())
        
        return motion
    
    def set_command(self, vx, vy, vtheta):
        """
        Set velocity command for the robot.
        Use this to integrate with your A* path planner.
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            vtheta: Angular velocity (rad/s)
        """
        self.target_velocity = np.array([vx, vy, vtheta])
        
        # Adjust motion type based on speed
        speed = np.sqrt(vx**2 + vy**2)
        if speed < 0.1:
            self.current_motion_type = 'stand'
            self.phase_speed = 0.0
        else:
            self.current_motion_type = 'walk'
            self.phase_speed = np.clip(speed, 0.5, 2.0)  # Adjust phase speed with command
    
    def compute_imitation_reward(self):
        """
        Core DeepMimic reward calculation.
        Compares current pose to reference motion.
        """
        # Get current reference frame
        motion = self.reference_motions[self.current_motion_type]
        frame_idx = int(self.motion_phase * motion['length']) % motion['length']
        
        ref_pos = motion['positions'][frame_idx]
        ref_vel = motion['velocities'][frame_idx]
        
        curr_pos = self.data.qpos.copy()
        curr_vel = self.data.qvel.copy()
        
        # Focus on leg joints (ignore arms since you don't have elbows)
        # Weight different components differently
        rewards = {}
        
        # Joint position reward (most important for gait)
        joint_pos_diff = 0.0
        for joint_name in self.joint_names:
            if joint_name in ['right leg lateral', 'left leg lateral',
                             'right hammy', 'left hammy',
                             'right knee', 'left knee']:
                joint_id = self._get_joint_id(joint_name)
                diff = curr_pos[joint_id] - ref_pos[joint_id]
                joint_pos_diff += diff * diff
        
        rewards['joint_pos'] = 0.65 * np.exp(-2.0 * joint_pos_diff)
        
        # Root orientation reward (keep upright)
        root_quat_diff = np.sum((curr_pos[3:7] - ref_pos[3:7])**2)
        rewards['root_orient'] = 0.15 * np.exp(-2.0 * root_quat_diff)
        
        # Root position reward (stay at right height)
        root_pos_diff = np.sum((curr_pos[0:3] - ref_pos[0:3])**2)
        rewards['root_pos'] = 0.1 * np.exp(-10.0 * root_pos_diff)
        
        # Velocity reward (smooth motion)
        vel_diff = np.mean((curr_vel[:12] - ref_vel[:12])**2)  # Focus on leg velocities
        rewards['velocity'] = 0.1 * np.exp(-0.1 * vel_diff)
        
        total_reward = sum(rewards.values())
        return total_reward, rewards
    
    def compute_task_reward(self):
        """
        Task-specific rewards (walking forward, following commands, etc.)
        """
        rewards = {}
        
        # Forward progress reward (from parent class)
        xy_velocity = self.data.qvel[:2]
        rewards['forward'] = xy_velocity[0]  # Reward forward motion
        
        # Command following reward (if enabled)
        if self.enable_command_following:
            vel_error = np.linalg.norm(xy_velocity - self.target_velocity[:2])
            rewards['command'] = np.exp(-2.0 * vel_error)
            
            # Angular velocity matching
            angular_error = abs(self.data.qvel[5] - self.target_velocity[2])
            rewards['rotation'] = np.exp(-2.0 * angular_error)
        
        # Staying upright (from parent class)
        rewards['upright'] = float(self.is_healthy)
        
        return rewards
    
    def check_early_termination(self):
        """
        Check if robot has deviated too far from reference.
        This guides exploration during training.
        """
        motion = self.reference_motions[self.current_motion_type]
        frame_idx = int(self.motion_phase * motion['length']) % motion['length']
        ref_pos = motion['positions'][frame_idx]
        curr_pos = self.data.qpos.copy()
        
        # Check key joint deviations
        deviation = 0.0
        for joint_name in ['right knee', 'left knee', 'right hammy', 'left hammy']:
            joint_id = self._get_joint_id(joint_name)
            deviation += abs(curr_pos[joint_id] - ref_pos[joint_id])
        
        return deviation > self.early_termination_threshold
    
    def step(self, action):
        """
        Step the environment with DeepMimic rewards.
        """
        # Execute action using parent class physics
        obs, original_reward, terminated, truncated, info = super().step(action)
        
        # Compute imitation reward
        imitation_reward, imitation_components = self.compute_imitation_reward()
        
        # Compute task rewards
        task_rewards = self.compute_task_reward()
        task_reward = sum(task_rewards.values()) * 0.1  # Scale task rewards
        
        # Combine rewards
        total_reward = (
            self.imitation_weight * imitation_reward +
            self.task_weight * task_reward
        )
        
        # Update motion phase
        self.motion_phase += self.phase_speed * self.dt
        if self.motion_phase >= 1.0:
            self.motion_phase = 0.0
        
        # Check early termination
        if self.check_early_termination() and self.imitation_weight > 0.3:
            terminated = True
            info['early_termination'] = True
        
        # Add command to observation if enabled
        if self.enable_command_following:
            obs = np.concatenate([obs, self.target_velocity])
        
        # Store detailed reward info
        info.update({
            'reward_imitation': imitation_reward,
            'reward_task': task_reward,
            'imitation_components': imitation_components,
            'task_components': task_rewards,
            'motion_phase': self.motion_phase,
            'current_motion': self.current_motion_type
        })
        
        return obs, total_reward, terminated, truncated, info
    
    def reset_model(self):
        """
        Reset to a state close to reference motion.
        This speeds up learning significantly.
        """
        # Call parent reset
        obs = super().reset_model()
        
        # Initialize at random point in motion cycle
        self.motion_phase = self.np_random.uniform(0, 1)
        
        # Get reference pose
        motion = self.reference_motions[self.current_motion_type]
        frame_idx = int(self.motion_phase * motion['length']) % motion['length']
        ref_pos = motion['positions'][frame_idx]
        ref_vel = motion['velocities'][frame_idx]
        
        # Blend current pose with reference (70% reference, 30% random)
        noise_scale = 0.02
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Keep some randomness but bias toward reference
        for joint_name in self.joint_names:
            try:
                joint_id = self._get_joint_id(joint_name)
                qpos[joint_id] = 0.7 * ref_pos[joint_id] + 0.3 * qpos[joint_id]
                qpos[joint_id] += self.np_random.uniform(-noise_scale, noise_scale)
            except:
                pass  # Skip if joint doesn't exist
        
        # Force safe height if too low (Training safety check)
        # WolverBot typically needs ~0.9m to be safely on the ground.
        # If mocap says < 0.6, it's definitely too low.
        if qpos[2] < 0.6:
            # print(f"Warning: Spawn height too low ({qpos[2]:.3f}m). Forcing to 0.9m.")
            qpos[2] = 0.9
            
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        
        # Add command to observation if enabled
        if self.enable_command_following:
            obs = np.concatenate([obs, self.target_velocity])
        
        return obs
    
    def set_motion_type(self, motion_type):
        """Switch between walk, stand, kick, etc."""
        if motion_type in self.reference_motions:
            self.current_motion_type = motion_type
            self.motion_phase = 0.0