import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
import gymnasium as gym

class DeepPerceptionNetwork(nn.Module):
    """CNN-based perception system for grid world understanding"""
    
    def __init__(self, grid_size=10, n_channels=4):
        super().__init__()
        self.grid_size = grid_size
        
        # Convolutional layers to process grid as image
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Global context processing
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Feature extraction
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # Final feature vector
        
        # Dropout for regularisation
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Process grid state into feature representation"""
        # Convolutional processing
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pool to fixed size
        x = self.pool(x)
        
        # Flatten and process through FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class DeepPerceptionEnv(gym.Env):
    """Enhanced environment with CNN-based perception"""
    
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        
        # Action space remains the same
        self.action_space = spaces.Discrete(5)
        
        # Observation space is now the feature vector from CNN
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )
        
        # Initialise perception network
        self.perception_net = DeepPerceptionNetwork(grid_size)
        self.perception_net.eval()  # Set to evaluation mode
        
        self.reset()
    
    def _create_grid_channels(self):
        """Create multi-channel representation of grid state"""
        channels = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
        
        # Channel 0: Agent position
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        channels[0, agent_y, agent_x] = 1.0
        
        # Channel 1: Resources
        for resource in self.resources:
            x, y = int(resource[0]), int(resource[1])
            channels[1, y, x] = 1.0
        
        # Channel 2: Obstacles
        for obstacle in self.obstacles:
            x, y = int(obstacle[0]), int(obstacle[1])
            channels[2, y, x] = 1.0
        
        # Channel 3: Distance map from agent
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - agent_y)**2 + (j - agent_x)**2)
                channels[3, i, j] = 1.0 / (1.0 + dist)  # Inverse distance
        
        return channels
    
    def _get_observation(self):
        """Get CNN-processed observation"""
        # Create grid representation
        grid_channels = self._create_grid_channels()
        
        # Convert to tensor and add batch dimension
        grid_tensor = torch.FloatTensor(grid_channels).unsqueeze(0)
        
        # Process through perception network
        with torch.no_grad():
            features = self.perception_net(grid_tensor)
        
        # Return as numpy array
        return features.squeeze(0).numpy()
    
    def reset(self, seed=None):
        """Reset environment with deep perception"""
        super().reset(seed=seed)
        
        # Agent starts at random position
        self.agent_pos = np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ], dtype=np.float32)
        
        # Generate resources and obstacles
        self.resources = [
            np.array([np.random.randint(0, self.grid_size),
                     np.random.randint(0, self.grid_size)], dtype=np.float32)
            for _ in range(5)
        ]
        
        self.obstacles = [
            np.array([np.random.randint(0, self.grid_size),
                     np.random.randint(0, self.grid_size)], dtype=np.float32)
            for _ in range(8)
        ]
        
        self.steps = 0
        self.max_steps = 100
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action with deep perception"""
        self.steps += 1
        
        # Map action to direction
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[1] > 0:  # UP
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:  # DOWN
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # LEFT
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # RIGHT
            self.agent_pos[0] += 1
        # action == 4: STAY
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            if np.array_equal(self.agent_pos, obstacle):
                self.agent_pos = old_pos  # Revert move
                reward = -5
                return self._get_observation(), reward, False, False, {}
        
        # Check resource collection
        reward = -0.1  # Step penalty
        
        resources_to_remove = []
        for i, resource in enumerate(self.resources):
            if np.array_equal(self.agent_pos, resource):
                resources_to_remove.append(i)
                reward = 10
                break
        
        for i in reversed(resources_to_remove):
            del self.resources[i]
        
        # Check if done
        done = len(self.resources) == 0 or self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}


def train_with_deep_perception():
    """Train an agent using deep perception"""
    from stable_baselines3 import PPO
    
    print("Creating environment with deep CNN perception...")
    env = DeepPerceptionEnv(grid_size=10)
    
    print("Initialising PPO with deep learning policy...")
    model = PPO(
        "MlpPolicy",  # Will process the CNN features
        env,
        verbose=1,
        learning_rate=0.0001,  # Lower learning rate for stability
        n_steps=2048,
        batch_size=64,
        policy_kwargs={
            "net_arch": [256, 128, 64]  # Deeper policy network
        }
    )
    
    print("Training with deep perception... This may take longer than before.")
    model.learn(total_timesteps=100000)
    
    model.save("deep_perception_agent")
    print("Deep perception agent saved!")
    
    return model


if __name__ == "__main__":
    # Test the deep perception system
    print("Testing deep perception environment...")
    env = DeepPerceptionEnv(grid_size=10)
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation sample: {obs[:10]}")  # First 10 features
    
    # Run a few random steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
