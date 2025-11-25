import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AgentDecisionEnv(gym.Env):
    """Custom Environment for RL agents"""
    
    def __init__(self, grid_size=10):
        super().__init__()
        
        self.grid_size = grid_size
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        self.action_space = spaces.Discrete(5)
        
        # Observation space: agent position, nearest resource, nearest obstacle
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(6,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
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
    
    def _get_observation(self):
        """Get current state observation"""
        # Find nearest resource
        if self.resources:
            distances = [np.linalg.norm(self.agent_pos - r) for r in self.resources]
            nearest_resource = self.resources[np.argmin(distances)]
        else:
            nearest_resource = np.array([self.grid_size/2, self.grid_size/2])
        
        # Find nearest obstacle
        if self.obstacles:
            distances = [np.linalg.norm(self.agent_pos - o) for o in self.obstacles]
            nearest_obstacle = self.obstacles[np.argmin(distances)]
        else:
            nearest_obstacle = np.array([self.grid_size/2, self.grid_size/2])
        
        return np.concatenate([
            self.agent_pos,
            nearest_resource,
            nearest_obstacle
        ])
    
    def step(self, action):
        """Execute action and return result"""
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
        # action == 4: STAY (do nothing)
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            if np.array_equal(self.agent_pos, obstacle):
                self.agent_pos = old_pos  # Revert move
                reward = -5  # Penalty for hitting obstacle
                return self._get_observation(), reward, False, False, {}
        
        # Check resource collection
        reward = -0.1  # Small penalty for each step (encourages efficiency)
        
        resources_to_remove = []
        for i, resource in enumerate(self.resources):
            if np.array_equal(self.agent_pos, resource):
                resources_to_remove.append(i)
                reward = 10  # Big reward for collecting resource
                break
        
        # Remove collected resources (in reverse to maintain indices)
        for i in reversed(resources_to_remove):
            del self.resources[i]
        
        # Check if done
        done = len(self.resources) == 0 or self.steps >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}
