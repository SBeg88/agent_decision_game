from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_environment import AgentDecisionEnv

# Create environment
env = AgentDecisionEnv(grid_size=10)

# Verify environment is correct
check_env(env)

# Create RL agent using PPO algorithm
model = PPO(
    "MlpPolicy",  # Multi-layer perceptron policy
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
)

print("Training agent... This will take a few minutes.")

# Train the agent
model.learn(total_timesteps=50000)

# Save the trained model
model.save("trained_agent")

print("Training complete! Model saved as 'trained_agent'")

# Test the trained agent
obs, _ = env.reset()
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    print(f"Step {i}: Action={action}, Reward={reward}")
    if done:
        print(f"Episode finished after {i+1} steps")
        break