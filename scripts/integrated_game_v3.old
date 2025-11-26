"""
INTEGRATED MULTI-AGENT DECISION GAME V3
========================================
Combines:
- Reinforcement Learning (PPO)
- Monte Carlo Decision Making
- Deep CNN Perception
- Werewolf-inspired Opponent Modelling
- Bayesian Belief Updates
- Strategic Deception Detection
"""

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import math

# Import base modules (assuming they exist in same directory)
try:
    from rl_environment import AgentDecisionEnv
    from monte_carlo_decisions import MonteCarloSimulator, Decision
except ImportError:
    print("Warning: Base modules not found. Using simplified versions.")

# Initialize Pygame
pygame.init()

# Game settings
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
INFO_HEIGHT = 200  # Increased for more information display
SCREEN_HEIGHT = WINDOW_SIZE + INFO_HEIGHT
FPS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
PURPLE = (150, 0, 200)
ORANGE = (255, 165, 0)
DARK_BLUE = (0, 50, 150)
DARK_GREEN = (0, 150, 0)
PINK = (255, 192, 203)


class DeepPerceptionNetwork(nn.Module):
    """CNN-based perception for processing grid as image"""
    
    def __init__(self, grid_size=10, n_channels=5):
        super().__init__()
        self.grid_size = grid_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Feature extraction
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        features = self.fc3(x)
        return features


class WerewolfOpponentModel:
    """
    Bayesian opponent modelling inspired by Werewolf Game
    Tracks beliefs about opponent strategies and deception
    """
    
    def __init__(self, n_opponents=3):
        self.n_opponents = n_opponents
        
        # Strategy beliefs for each opponent (Bayesian priors)
        self.strategy_beliefs = {
            i: {
                'aggressive': 0.25,
                'balanced': 0.25,
                'conservative': 0.25,
                'deceptive': 0.25
            }
            for i in range(n_opponents)
        }
        
        # Action history for pattern detection
        self.action_history = defaultdict(lambda: deque(maxlen=30))
        
        # Deception indicators
        self.deception_scores = {i: 0.0 for i in range(n_opponents)}
        
        # Coalition detection
        self.coalition_matrix = np.zeros((n_opponents, n_opponents))
    
    def update_belief(self, opp_id: int, observation: Dict):
        """Bayesian belief update based on observed action"""
        # Prior beliefs
        beliefs = self.strategy_beliefs[opp_id]
        
        # Compute likelihoods
        likelihoods = {}
        for strategy in beliefs.keys():
            likelihoods[strategy] = self._compute_likelihood(
                observation, strategy
            )
        
        # Bayesian update
        evidence = sum(beliefs[s] * likelihoods[s] for s in beliefs)
        
        if evidence > 0:
            for strategy in beliefs:
                beliefs[strategy] = (beliefs[strategy] * likelihoods[strategy]) / evidence
        
        # Update deception score
        self._update_deception_score(opp_id, observation)
        
        # Track action history
        self.action_history[opp_id].append(observation)
    
    def _compute_likelihood(self, obs: Dict, strategy: str) -> float:
        """P(observation | strategy)"""
        action = obs.get('action', 4)
        aggressive_move = obs.get('moved_toward_resource', False)
        defensive_move = obs.get('avoided_opponent', False)
        
        if strategy == 'aggressive':
            if aggressive_move:
                return 0.8
            elif action == 4:  # STAY
                return 0.1
            return 0.3
            
        elif strategy == 'conservative':
            if defensive_move:
                return 0.7
            elif action == 4:
                return 0.5
            return 0.2
            
        elif strategy == 'balanced':
            return 0.4  # Balanced has uniform likelihood
            
        else:  # deceptive
            # Deceptive agents show inconsistent patterns
            if obs.get('pattern_break', False):
                return 0.7
            return 0.3
    
    def _update_deception_score(self, opp_id: int, obs: Dict):
        """Detect deceptive patterns like in Werewolf Game"""
        history = list(self.action_history[opp_id])
        
        if len(history) < 10:
            return
        
        # Check for strategy shifts (like werewolves changing behavior)
        first_half = history[:len(history)//2]
        second_half = history[len(history)//2:]
        
        # Simple pattern difference metric
        pattern_shift = self._calculate_pattern_shift(first_half, second_half)
        
        # Update deception score
        self.deception_scores[opp_id] = min(1.0, pattern_shift)
    
    def _calculate_pattern_shift(self, hist1: List, hist2: List) -> float:
        """Calculate behavioral pattern shift between two periods"""
        if not hist1 or not hist2:
            return 0.0
        
        # Extract action distributions
        actions1 = [h.get('action', 4) for h in hist1]
        actions2 = [h.get('action', 4) for h in hist2]
        
        # Calculate distribution difference
        dist1 = np.bincount(actions1, minlength=5) + 1e-10
        dist2 = np.bincount(actions2, minlength=5) + 1e-10
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()
        
        # KL divergence as measure of shift
        kl_div = np.sum(dist1 * np.log(dist1 / dist2))
        
        return min(1.0, kl_div / 2.0)
    
    def get_analysis(self, opp_id: int) -> Dict:
        """Get complete opponent analysis"""
        beliefs = self.strategy_beliefs[opp_id]
        dominant_strategy = max(beliefs.items(), key=lambda x: x[1])
        
        return {
            'beliefs': beliefs,
            'dominant_strategy': dominant_strategy[0],
            'confidence': dominant_strategy[1],
            'is_deceptive': self.deception_scores[opp_id] > 0.5,
            'deception_score': self.deception_scores[opp_id]
        }


class EnhancedStrategicAgent:
    """
    Advanced agent combining all AI systems:
    - RL for movement
    - Monte Carlo for high-level decisions  
    - Deep CNN perception
    - Werewolf-inspired opponent modelling
    """
    
    def __init__(self, agent_id, color, model_path=None, use_deep_perception=True):
        self.agent_id = agent_id
        self.color = color
        self.position = np.array([0, 0])
        
        # RL model for movement
        self.rl_model = None
        if model_path:
            try:
                self.rl_model = PPO.load(model_path)
            except:
                print(f"Could not load model for agent {agent_id}")
        
        # Deep perception
        self.use_deep_perception = use_deep_perception
        if use_deep_perception:
            self.perception_net = DeepPerceptionNetwork()
            self.perception_net.eval()
        
        # Monte Carlo decision maker
        self.mc_simulator = MonteCarloSimulator(n_simulations=100)
        
        # Opponent modelling (for multi-agent scenarios)
        self.opponent_model = None  # Will be set if there are opponents
        
        # Agent state
        self.strategy = "Balanced"
        self.resources_collected = 0
        self.total_reward = 0
        self.decision_timer = 0
        self.decision_interval = 15
        
        # Strategy history for analysis
        self.strategy_history = []
        self.action_history = deque(maxlen=50)
    
    def set_opponent_model(self, n_opponents):
        """Initialize opponent model"""
        self.opponent_model = WerewolfOpponentModel(n_opponents)
    
    def create_grid_perception(self, env, other_agents=None):
        """Create CNN input from game state"""
        channels = np.zeros((5, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        # Channel 0: Self position
        channels[0, int(self.position[1]), int(self.position[0])] = 1.0
        
        # Channel 1: Resources
        for resource in env.resources:
            x, y = int(resource[0]), int(resource[1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                channels[1, y, x] = 1.0
        
        # Channel 2: Obstacles
        for obstacle in env.obstacles:
            x, y = int(obstacle[0]), int(obstacle[1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                channels[2, y, x] = 1.0
        
        # Channel 3: Other agents
        if other_agents:
            for agent in other_agents:
                if agent.agent_id != self.agent_id:
                    x, y = int(agent.position[0]), int(agent.position[1])
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        channels[3, y, x] = 1.0
        
        # Channel 4: Distance/influence map
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                dist = np.sqrt((i - self.position[1])**2 + (j - self.position[0])**2)
                channels[4, i, j] = 1.0 / (1.0 + dist)
        
        return channels
    
    def get_deep_perception(self, env, other_agents=None):
        """Get CNN-processed perception"""
        grid_channels = self.create_grid_perception(env, other_agents)
        grid_tensor = torch.FloatTensor(grid_channels).unsqueeze(0)
        
        with torch.no_grad():
            features = self.perception_net(grid_tensor)
        
        return features.squeeze(0).numpy()
    
    def choose_strategy(self, env, other_agents=None):
        """
        High-level strategy decision using Monte Carlo and opponent modelling
        Inspired by Werewolf Game's revelation timing
        """
        
        # Define strategic decisions
        decisions = []
        
        # Base decisions
        base_decisions = [
            Decision("Aggressive", cost=1.0, 
                    uncertain_parameters={'success_rate': (0.4, 0.8)},
                    category="exploration"),
            Decision("Balanced", cost=0.7,
                    uncertain_parameters={'success_rate': (0.5, 0.85)},
                    category="mixed"),
            Decision("Conservative", cost=0.5,
                    uncertain_parameters={'success_rate': (0.6, 0.9)},
                    category="resource")
        ]
        
        # Add deceptive strategies if opponents detected deception
        if self.opponent_model and other_agents:
            for opp_id, agent in enumerate(other_agents):
                if agent.agent_id != self.agent_id:
                    analysis = self.opponent_model.get_analysis(opp_id)
                    if analysis['is_deceptive']:
                        decisions.append(
                            Decision("Counter-Deceptive", cost=0.8,
                                    uncertain_parameters={'success_rate': (0.5, 0.9)},
                                    category="counter")
                        )
                        break
        
        if not decisions:
            decisions = base_decisions
        
        # Current scenario
        scenario = {
            'reward': len(env.resources) * 3,
            'environment': 'competitive' if other_agents else 'solo',
            'competition_level': 'high' if other_agents and len(other_agents) > 2 else 'low'
        }
        
        # Run Monte Carlo analysis
        results = {}
        for decision in decisions:
            results[decision.name] = self.mc_simulator.analyze_decision(
                decision, scenario
            )
        
        # Choose best strategy
        best = max(results.items(), key=lambda x: x[1]['mean'])
        old_strategy = self.strategy
        self.strategy = best[0]
        
        # Track decision
        self.strategy_history.append({
            'step': self.decision_timer,
            'strategy': self.strategy,
            'expected_value': best[1]['mean']
        })
        
        return {
            'changed': old_strategy != self.strategy,
            'old_strategy': old_strategy,
            'new_strategy': self.strategy,
            'expected_value': best[1]['mean']
        }
    
    def get_action(self, env, other_agents=None):
        """
        Get action combining RL and strategic decisions
        """
        
        # Periodic strategy reconsideration
        strategy_info = None
        if self.decision_timer % self.decision_interval == 0:
            strategy_info = self.choose_strategy(env, other_agents)
        
        self.decision_timer += 1
        
        # Get observation
        if self.use_deep_perception:
            observation = self.get_deep_perception(env, other_agents)
        else:
            # Fallback to simple observation
            observation = env._get_observation() if hasattr(env, '_get_observation') else np.zeros(6)
        
        # Get action from RL model or use strategy-based heuristic
        if self.rl_model:
            action, _ = self.rl_model.predict(observation, deterministic=True)
        else:
            # Heuristic based on strategy
            if self.strategy == "Aggressive":
                action = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
            elif self.strategy == "Conservative":
                action = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.1, 0.1, 0.1, 0.6])
            else:  # Balanced
                action = np.random.randint(0, 5)
        
        # Track action
        self.action_history.append(action)
        
        return action, strategy_info
    
    def update_opponent_models(self, other_agents):
        """Update beliefs about other agents"""
        if not self.opponent_model or not other_agents:
            return
        
        for opp_id, agent in enumerate(other_agents):
            if agent.agent_id != self.agent_id:
                # Create observation about opponent
                obs = {
                    'action': agent.action_history[-1] if agent.action_history else 4,
                    'moved_toward_resource': False,  # Would need to track this
                    'avoided_opponent': False,
                    'pattern_break': len(agent.strategy_history) > 1 and 
                                   agent.strategy_history[-1] != agent.strategy_history[-2]
                                   if hasattr(agent, 'strategy_history') else False
                }
                
                self.opponent_model.update_belief(opp_id, obs)


class IntegratedGameV3:
    """
    Main game class with full AI integration:
    - Multiple AI agents
    - Deep learning perception
    - Werewolf-inspired opponent modelling
    - Strategic deception and coalition detection
    """
    
    def __init__(self, n_agents=3, use_deep_learning=True):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Agent Decision Game V3 - Full AI Integration")
        self.clock = pygame.time.Clock()
        
        # Environment
        self.env = self.create_environment()
        
        # Create agents with different strategies
        self.agents = []
        agent_colors = [BLUE, ORANGE, PURPLE]
        agent_strategies = ["Balanced", "Aggressive", "Conservative"]
        
        for i in range(n_agents):
            agent = EnhancedStrategicAgent(
                agent_id=i,
                color=agent_colors[i % len(agent_colors)],
                model_path=None,  # Would load trained models here
                use_deep_perception=use_deep_learning
            )
            
            # Set initial position
            agent.position = np.array([
                np.random.randint(0, GRID_SIZE),
                np.random.randint(0, GRID_SIZE)
            ], dtype=np.float32)
            
            # Set initial strategy
            agent.strategy = agent_strategies[i % len(agent_strategies)]
            
            # Initialize opponent model
            if n_agents > 1:
                agent.set_opponent_model(n_agents - 1)
            
            self.agents.append(agent)
        
        # Game state
        self.episode = 1
        self.steps = 0
        self.running = True
        self.paused = False
        
        # UI elements
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        print("\n" + "="*70)
        print("  INTEGRATED MULTI-AGENT DECISION GAME V3")
        print("="*70)
        print("\nFeatures:")
        print("  • Deep CNN Perception")
        print("  • Werewolf-inspired Opponent Modelling")
        print("  • Bayesian Belief Updates")
        print("  • Strategic Deception Detection")
        print("  • Monte Carlo Decision Making")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset episode")
        print("  D: Toggle debug info")
        print("  Q: Quit")
        print("\n" + "="*70)
    
    def create_environment(self):
        """Create game environment"""
        # Simple environment for demonstration
        env = type('Environment', (), {})()
        env.resources = [
            np.array([np.random.randint(0, GRID_SIZE),
                     np.random.randint(0, GRID_SIZE)], dtype=np.float32)
            for _ in range(5)
        ]
        env.obstacles = [
            np.array([np.random.randint(0, GRID_SIZE),
                     np.random.randint(0, GRID_SIZE)], dtype=np.float32)
            for _ in range(8)
        ]
        return env
    
    def draw_grid(self):
        """Draw the game grid with all elements"""
        self.screen.fill(WHITE)
        
        # Draw grid lines
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_SIZE, y))
        
        # Draw resources
        for resource in self.env.resources:
            x, y = int(resource[0]), int(resource[1])
            pygame.draw.circle(self.screen, GREEN,
                             (x * CELL_SIZE + CELL_SIZE // 2,
                              y * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 3)
        
        # Draw obstacles
        for obstacle in self.env.obstacles:
            x, y = int(obstacle[0]), int(obstacle[1])
            pygame.draw.rect(self.screen, RED,
                           (x * CELL_SIZE + 5, y * CELL_SIZE + 5,
                            CELL_SIZE - 10, CELL_SIZE - 10))
        
        # Draw agents with strategy-based visual cues
        for agent in self.agents:
            x, y = int(agent.position[0]), int(agent.position[1])
            
            # Outer ring shows strategy
            strategy_colors = {
                'Aggressive': RED,
                'Balanced': YELLOW,
                'Conservative': GREEN,
                'Counter-Deceptive': PURPLE
            }
            accent_color = strategy_colors.get(agent.strategy, GRAY)
            
            # Draw deception indicator if detected
            if agent.opponent_model:
                for opp_id in range(len(self.agents) - 1):
                    analysis = agent.opponent_model.get_analysis(opp_id)
                    if analysis['is_deceptive']:
                        # Draw warning indicator
                        pygame.draw.circle(self.screen, PINK,
                                         (x * CELL_SIZE + CELL_SIZE // 2,
                                          y * CELL_SIZE + CELL_SIZE // 2),
                                         CELL_SIZE // 2)
                        break
            
            # Draw agent
            pygame.draw.circle(self.screen, accent_color,
                             (x * CELL_SIZE + CELL_SIZE // 2,
                              y * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 4 + 4)
            pygame.draw.circle(self.screen, agent.color,
                             (x * CELL_SIZE + CELL_SIZE // 2,
                              y * CELL_SIZE + CELL_SIZE // 2),
                             CELL_SIZE // 4)
            
            # Draw agent ID
            id_text = self.font_small.render(str(agent.agent_id), True, WHITE)
            self.screen.blit(id_text,
                           (x * CELL_SIZE + CELL_SIZE // 2 - 5,
                            y * CELL_SIZE + CELL_SIZE // 2 - 8))
    
    def draw_info_panel(self):
        """Draw detailed information panel"""
        # Panel background
        panel_rect = pygame.Rect(0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT)
        pygame.draw.rect(self.screen, GRAY, panel_rect)
        pygame.draw.line(self.screen, BLACK, (0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE), 3)
        
        # Episode info
        info_text = self.font_medium.render(
            f"Episode: {self.episode} | Step: {self.steps}",
            True, BLACK
        )
        self.screen.blit(info_text, (10, WINDOW_SIZE + 10))
        
        # Agent information
        y_offset = WINDOW_SIZE + 40
        for i, agent in enumerate(self.agents):
            # Agent header
            agent_text = self.font_small.render(
                f"Agent {agent.agent_id}: {agent.strategy}",
                True, agent.color
            )
            self.screen.blit(agent_text, (10 + i * 200, y_offset))
            
            # Opponent beliefs (if available)
            if agent.opponent_model and len(self.agents) > 1:
                y_offset_beliefs = y_offset + 25
                for opp_id in range(len(self.agents)):
                    if opp_id != agent.agent_id:
                        analysis = agent.opponent_model.get_analysis(
                            opp_id if opp_id < agent.agent_id else opp_id - 1
                        )
                        
                        # Show dominant belief
                        belief_text = self.font_small.render(
                            f"→ A{opp_id}: {analysis['dominant_strategy'][:3]} "
                            f"({analysis['confidence']:.0%})",
                            True, BLACK
                        )
                        self.screen.blit(belief_text, (20 + i * 200, y_offset_beliefs))
                        
                        # Deception warning
                        if analysis['is_deceptive']:
                            dec_text = self.font_small.render(
                                f"  ⚠ Deceptive!",
                                True, RED
                            )
                            self.screen.blit(dec_text, (20 + i * 200, y_offset_beliefs + 20))
                        
                        y_offset_beliefs += 40
        
        # Resources remaining
        resource_text = self.font_small.render(
            f"Resources: {len(self.env.resources)} | Obstacles: {len(self.env.obstacles)}",
            True, BLACK
        )
        self.screen.blit(resource_text, (10, WINDOW_SIZE + INFO_HEIGHT - 30))
        
        # Pause indicator
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, RED)
            text_rect = pause_text.get_rect(center=(WINDOW_SIZE//2, WINDOW_SIZE//2))
            self.screen.blit(pause_text, text_rect)
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                if event.key == pygame.K_r:
                    self.reset_episode()
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_d:
                    self.debug_mode = not hasattr(self, 'debug_mode') or not self.debug_mode
    
    def reset_episode(self):
        """Reset to new episode"""
        self.env = self.create_environment()
        
        # Reset agents
        for agent in self.agents:
            agent.position = np.array([
                np.random.randint(0, GRID_SIZE),
                np.random.randint(0, GRID_SIZE)
            ], dtype=np.float32)
            agent.resources_collected = 0
            agent.total_reward = 0
            agent.decision_timer = 0
            agent.strategy_history = []
            agent.action_history.clear()
            
            # Reset opponent model
            if agent.opponent_model:
                agent.opponent_model = WerewolfOpponentModel(len(self.agents) - 1)
        
        self.steps = 0
        self.episode += 1
        print(f"\n=== Episode {self.episode} ===")
    
    def step_game(self):
        """Execute one game step for all agents"""
        if self.paused:
            return
        
        # Each agent chooses action
        for agent in self.agents:
            # Get action with strategic decision
            action, strategy_info = agent.get_action(self.env, self.agents)
            
            # Execute action
            old_pos = agent.position.copy()
            
            if action == 0 and agent.position[1] > 0:  # UP
                agent.position[1] -= 1
            elif action == 1 and agent.position[1] < GRID_SIZE - 1:  # DOWN
                agent.position[1] += 1
            elif action == 2 and agent.position[0] > 0:  # LEFT
                agent.position[0] -= 1
            elif action == 3 and agent.position[0] < GRID_SIZE - 1:  # RIGHT
                agent.position[0] += 1
            
            # Check collision with obstacles
            for obstacle in self.env.obstacles:
                if np.array_equal(agent.position, obstacle):
                    agent.position = old_pos
                    agent.total_reward -= 5
                    break
            
            # Check resource collection
            resources_to_remove = []
            for i, resource in enumerate(self.env.resources):
                if np.allclose(agent.position, resource, atol=0.5):
                    resources_to_remove.append(i)
                    agent.resources_collected += 1
                    agent.total_reward += 10
                    print(f"  Agent {agent.agent_id} collected resource! "
                          f"Total: {agent.resources_collected}")
                    break
            
            # Remove collected resources
            for i in reversed(resources_to_remove):
                del self.env.resources[i]
            
            # Handle strategy change
            if strategy_info and strategy_info['changed']:
                print(f"  Agent {agent.agent_id}: Strategy → {strategy_info['new_strategy']}")
        
        # Update opponent models
        for agent in self.agents:
            agent.update_opponent_models(self.agents)
        
        self.steps += 1
        
        # Check episode end
        if len(self.env.resources) == 0 or self.steps >= 100:
            print(f"\nEpisode {self.episode} Complete!")
            for agent in self.agents:
                print(f"  Agent {agent.agent_id}: "
                      f"Resources: {agent.resources_collected}, "
                      f"Reward: {agent.total_reward:.1f}")
            
            # Track performance
            for agent in self.agents:
                self.performance_history[agent.agent_id].append(agent.total_reward)
            
            pygame.time.wait(2000)
            self.reset_episode()
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.step_game()
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Show final statistics
        print("\n" + "="*70)
        print("  GAME STATISTICS")
        print("="*70)
        
        for agent_id, rewards in self.performance_history.items():
            if rewards:
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                print(f"Agent {agent_id}: Avg Reward = {avg_reward:.2f} ± {std_reward:.2f}")
        
        pygame.quit()


def main():
    """Start the integrated game v3"""
    try:
        # Check if we have required modules
        import torch
        import pygame
        
        game = IntegratedGameV3(n_agents=3, use_deep_learning=True)
        game.run()
    except ImportError as e:
        print(f"\nError: Missing required module - {e}")
        print("Please install: pip install pygame torch numpy")
    except Exception as e:
        print(f"\nError starting game: {e}")
        raise


if __name__ == "__main__":
    main()
