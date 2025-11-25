"""
Main training script for DeepMimic WolverBot.
Run this to train your robot with motion imitation.
"""

import os
import sys
import numpy as np
from datetime import datetime
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.deepmimic_wolverbot_env import DeepMimicWolverBot
from utils.motion_data import create_motion_dataset

def make_env(rank, motion_file=None):
    """Create environment for training"""
    def _init():
        # Get absolute path to scene.xml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, '..', 'scene.xml')
        xml_path = os.path.abspath(xml_path)  # Convert to absolute path
        
        env = DeepMimicWolverBot(
            xml_file=xml_path,  # Use absolute path
            reference_motion_file=motion_file,
            imitation_weight=0.7,
            task_weight=0.3,
            enable_command_following=True,
            early_termination_threshold=2.0,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.01,
            healthy_reward=2.0,
            reset_noise_scale=0.02
        )
        env = Monitor(env)
        return env
    return _init

def train_deepmimic(
    total_timesteps=1_000_000,
    n_envs=1,  # Use 1 for single GPU
    save_freq=10_000,
    eval_freq=5_000,
    log_dir="logs",
    model_dir="models"
):
    """
    Train WolverBot with DeepMimic approach.
    
    Args:
        total_timesteps: Total training steps (1M = ~2 hours on gaming GPU)
        n_envs: Number of parallel environments
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        log_dir: Directory for tensorboard logs
        model_dir: Directory for saved models
    """
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("mocap_data", exist_ok=True)
    
    # Create or load motion data
    motion_file = "mocap_data/reference_motions.npz"
    if not os.path.exists(motion_file):
        print("Creating motion reference data...")
        motion_file = create_motion_dataset(motion_file)
    
    # Create training environment(s)
    print(f"Creating {n_envs} training environment(s)...")
    env = DummyVecEnv([make_env(i, motion_file) for i in range(n_envs)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(99, motion_file)])
    
    # Training hyperparameters optimized for DeepMimic
    model = PPO(
        "MlpPolicy",
        env,
        # Learning parameters
        learning_rate=3e-4,  # Standard for DeepMimic
        n_steps=2048 // n_envs,  # Steps per environment
        batch_size=64,
        n_epochs=10,
        gamma=0.95,  # Slightly lower for motion imitation
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,  # No entropy for imitation
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # Network architecture (smaller for faster training)
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            activation_fn=nn.ReLU
        ),
        
        # Other settings
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",  # Use "cpu" if no GPU
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="wolverbot_deepmimic",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("\n" + "="*50)
    print("Starting DeepMimic Training")
    print("="*50)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Estimated time: {total_timesteps/500_000:.1f} hours on gaming GPU")
    print(f"Tensorboard logs: {log_dir}")
    print(f"Models saved to: {model_dir}")
    print("\nTraining... (Press Ctrl+C to stop early and save)")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "wolverbot_deepmimic_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_env = make_env(0, motion_file)()
    obs = test_env.reset()
    
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print(f"  Imitation reward: {info.get('reward_imitation', 0):.3f}")
            print(f"  Task reward: {info.get('reward_task', 0):.3f}")
            break
    
    test_env.close()
    env.close()
    eval_env.close()
    
    return model

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("No GPU found, using CPU (training will be slower)")
        device = "cpu"
    
    # Start training
    # For initial test: 100k steps (~10-15 minutes on GPU)
    # For real training: 1-5M steps (~2-10 hours on GPU)
    model = train_deepmimic(
        total_timesteps=100_000,  # Start small for testing
        n_envs=1,  # Single environment for limited GPU memory
        save_freq=10_000,
        eval_freq=5_000
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Check tensorboard logs: tensorboard --logdir logs")
    print("2. Test the model: python test_trained_model.py")
    print("3. Increase timesteps for better results")