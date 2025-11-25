from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv
import numpy as np

def test_agent(model_path, num_episodes=10):
    """Test an agent over multiple episodes"""
    model = PPO.load(model_path)
    env = AgentDecisionEnv(grid_size=10)
    
    episode_rewards = []
    resources_collected = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        initial_resources = len(env.resources)
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        collected = initial_resources - len(env.resources)
        episode_rewards.append(total_reward)
        resources_collected.append(collected)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_resources': np.mean(resources_collected),
        'best_reward': np.max(episode_rewards)
    }

# Test the trained agent
print("Testing trained agent...")
results = test_agent("trained_agent", num_episodes=20)

print("\n=== Agent Performance ===")
print(f"Average Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
print(f"Average Resources Collected: {results['mean_resources']:.2f} / 5")
print(f"Best Episode Reward: {results['best_reward']:.2f}")