from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv

# Create environment
env = AgentDecisionEnv(grid_size=10)

# Load the existing model to continue training
model = PPO.load("trained_agent", env=env)

print("Continuing training for better performance...")

# Train for another 100,000 steps
model.learn(total_timesteps=100000)

# Save the improved model
model.save("trained_agent_v2")

print("Extended training complete! Model saved as 'trained_agent_v2'")