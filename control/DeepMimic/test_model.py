"""
Test and visualize your trained DeepMimic model.
Use this to see how well the robot learned.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.deepmimic_wolverbot import DeepMimicWolverBot

def test_model(model_path="models/wolverbot_deepmimic_final.zip", 
               n_episodes=5,
               render=False,
               save_video=False):
    """
    Test trained model and collect metrics.
    
    Args:
        model_path: Path to trained model
        n_episodes: Number of test episodes
        render: Whether to render (requires display)
        save_video: Whether to save video
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    env = DeepMimicWolverBot(
        xml_file='../scene.xml',
        reference_motion_file="mocap_data/reference_motions.npz",
        enable_command_following=True
    )
    
    # Test metrics
    episode_rewards = []
    episode_lengths = []
    imitation_scores = []
    task_scores = []
    
    for episode in range(n_episodes):
        obs, info = env.reset() # Stable Baselines 3 / Gymnasium API requires handling tuple return
        done = False
        truncated = False
        
        ep_reward = 0
        ep_length = 0
        ep_imitation = []
        ep_task = []
        
        # Test different commands
        if episode == 0:
            env.set_command(0.5, 0, 0)  # Walk forward
        elif episode == 1:
            env.set_command(0, 0.3, 0)  # Walk sideways
        elif episode == 2:
            env.set_command(0.3, 0, 0.5)  # Walk and turn
        else:
            env.set_command(0.5, 0, 0)  # Default forward
        
        print(f"\nEpisode {episode + 1}: Command = {env.target_velocity}")
        
        while not done and not truncated and ep_length < 1000:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Collect metrics
            ep_reward += reward
            ep_length += 1
            ep_imitation.append(info.get('reward_imitation', 0))
            ep_task.append(info.get('reward_task', 0))
            
            if render and ep_length % 10 == 0:
                env.render()
        
        # Store episode metrics
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        imitation_scores.append(np.mean(ep_imitation))
        task_scores.append(np.mean(ep_task))
        
        print(f"  Total Reward: {ep_reward:.2f}")
        print(f"  Episode Length: {ep_length}")
        print(f"  Avg Imitation Score: {imitation_scores[-1]:.3f}")
        print(f"  Avg Task Score: {task_scores[-1]:.3f}")
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Imitation: {np.mean(imitation_scores):.3f}")
    print(f"Average Task: {np.mean(task_scores):.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].bar(range(1, n_episodes+1), episode_rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    
    axes[0, 1].bar(range(1, n_episodes+1), episode_lengths)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    
    axes[1, 0].bar(range(1, n_episodes+1), imitation_scores)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Imitation Score')
    axes[1, 0].set_title('Motion Quality')
    
    axes[1, 1].bar(range(1, n_episodes+1), task_scores)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Task Score')
    axes[1, 1].set_title('Task Performance')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    print(f"\nResults plot saved to test_results.png")
    
    env.close()

def test_command_following(model_path="models/wolverbot_deepmimic_final.zip"):
    """
    Test how well the robot follows velocity commands.
    This simulates integration with your A* planner.
    """
    print("\nTesting command following (A* integration)...")
    
    model = PPO.load(model_path)
    env = DeepMimicWolverBot(
        xml_file='../scene.xml',
        enable_command_following=True
    )
    
    # Simulate A* path with waypoints
    waypoints = [
        (0.5, 0.0, 0.0),   # Forward
        (0.3, 0.3, 0.0),   # Diagonal
        (0.0, 0.5, 0.0),   # Sideways
        (0.5, 0.0, 0.5),   # Forward with turn
        (0.0, 0.0, 0.0),   # Stop
    ]
    
    obs, info = env.reset()
    
    for i, (vx, vy, vtheta) in enumerate(waypoints):
        print(f"\nWaypoint {i+1}: vx={vx}, vy={vy}, vtheta={vtheta}")
        env.set_command(vx, vy, vtheta)
        
        # Execute for 100 steps
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            if step % 20 == 0:
                actual_vx = env.data.qvel[0]
                actual_vy = env.data.qvel[1]
                print(f"  Step {step}: Actual velocity = ({actual_vx:.2f}, {actual_vy:.2f})")
            
            if done:
                obs, info = env.reset()
                break
    
    env.close()
    print("\nCommand following test complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                       default="models/wolverbot_deepmimic_final.zip",
                       help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of test episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment")
    parser.add_argument("--test-commands", action="store_true",
                       help="Test command following")
    
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        test_model(args.model, args.episodes, args.render)
        
        if args.test_commands:
            test_command_following(args.model)
    else:
        print(f"Model not found at {args.model}")
        print("Please train a model first using: python train.py")