import pygame
import numpy as np
from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv
from monte_carlo_decisions import MonteCarloSimulator, Decision

# Initialize Pygame
pygame.init()

# Game settings
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
INFO_HEIGHT = 150
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

class StrategicAgent:
    """AI Agent that combines RL with Monte Carlo strategic decision-making"""
    
    def __init__(self, model_path, agent_id=1, color=BLUE):
        self.model = PPO.load(model_path)
        self.decision_simulator = MonteCarloSimulator(n_simulations=500)
        self.agent_id = agent_id
        self.color = color
        
        # Strategic state
        self.strategy = "Balanced"
        self.decision_timer = 0
        self.decision_interval = 20  # Reconsider strategy every N steps
        self.strategy_history = []
        
        # Performance tracking
        self.total_reward = 0
        self.resources_collected = 0
        self.obstacles_hit = 0
        self.decisions_made = 0
    
    def choose_strategy(self, env):
        """Use Monte Carlo simulation to choose optimal strategy"""
        
        # Define strategic decisions
        decisions = [
            Decision(
                name="Aggressive",
                cost=1.0,
                uncertain_parameters={'success_rate': (0.4, 0.8)},
                category="exploration"
            ),
            Decision(
                name="Balanced",
                cost=0.7,
                uncertain_parameters={'success_rate': (0.55, 0.85)},
                category="mixed"
            ),
            Decision(
                name="Conservative",
                cost=0.5,
                uncertain_parameters={'success_rate': (0.65, 0.95)},
                category="resource"
            )
        ]
        
        # Define current scenario based on game state
        scenario = {
            'reward': len(env.resources) * 3,  # Value of remaining resources
            'environment': 'uncertain' if len(env.obstacles) > 5 else 'stable',
            'competition_level': 'high' if self.resources_collected < 2 else 'low'
        }
        
        # Run Monte Carlo analysis
        results = {}
        for decision in decisions:
            results[decision.name] = self.decision_simulator.analyze_decision(
                decision, scenario)
        
        # Choose strategy with highest expected value
        best = max(results.items(), key=lambda x: x[1]['mean'])
        old_strategy = self.strategy
        self.strategy = best[0]
        
        # Track decision
        self.decisions_made += 1
        self.strategy_history.append({
            'step': self.decision_timer,
            'strategy': self.strategy,
            'expected_value': best[1]['mean'],
            'scenario': scenario
        })
        
        # Return info about strategy change
        changed = old_strategy != self.strategy
        return {
            'changed': changed,
            'old_strategy': old_strategy,
            'new_strategy': self.strategy,
            'expected_value': best[1]['mean'],
            'all_results': results
        }
    
    def get_action(self, obs, env):
        """Get action from RL model"""
        
        # Reconsider strategy periodically
        strategy_info = None
        if self.decision_timer % self.decision_interval == 0:
            strategy_info = self.choose_strategy(env)
        
        self.decision_timer += 1
        
        # Get action from RL model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Strategy could modify action probabilities in future versions
        # For now, we just track the strategic intent
        
        return action, strategy_info
    
    def update_stats(self, reward):
        """Update agent statistics"""
        self.total_reward += reward
        
        if reward == 10:
            self.resources_collected += 1
        elif reward == -5:
            self.obstacles_hit += 1

class IntegratedGame:
    """Main game class integrating RL and Monte Carlo decisions"""
    
    def __init__(self, model_path="trained_agent"):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, SCREEN_HEIGHT))
        pygame.display.set_caption("Integrated RL + Monte Carlo Decision Game")
        self.clock = pygame.time.Clock()
        
        # Create environment and agent
        self.env = AgentDecisionEnv(grid_size=GRID_SIZE)
        self.agent = StrategicAgent(model_path, agent_id=1, color=BLUE)
        
        # Game state
        self.obs, _ = self.env.reset()
        self.episode = 1
        self.steps = 0
        self.running = True
        self.paused = False
        
        # UI elements
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        
        # Strategy change notification
        self.strategy_notification = None
        self.notification_timer = 0
    
    def draw_grid(self):
        """Draw the game grid"""
        # Fill background
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
        
        # Draw agent with strategy-based color accent
        x, y = int(self.env.agent_pos[0]), int(self.env.agent_pos[1])
        
        # Outer ring color based on strategy
        strategy_colors = {
            'Aggressive': RED,
            'Balanced': YELLOW,
            'Conservative': GREEN
        }
        accent_color = strategy_colors.get(self.agent.strategy, YELLOW)
        
        pygame.draw.circle(self.screen, accent_color,
                         (x * CELL_SIZE + CELL_SIZE // 2,
                          y * CELL_SIZE + CELL_SIZE // 2),
                         CELL_SIZE // 4 + 4)
        pygame.draw.circle(self.screen, self.agent.color,
                         (x * CELL_SIZE + CELL_SIZE // 2,
                          y * CELL_SIZE + CELL_SIZE // 2),
                         CELL_SIZE // 4)
    
    def draw_info_panel(self):
        """Draw information panel at bottom"""
        # Panel background
        panel_rect = pygame.Rect(0, WINDOW_SIZE, WINDOW_SIZE, INFO_HEIGHT)
        pygame.draw.rect(self.screen, GRAY, panel_rect)
        pygame.draw.line(self.screen, BLACK, (0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE), 3)
        
        # Episode and step info
        info_text = self.font_medium.render(
            f"Episode: {self.episode} | Step: {self.steps} | Reward: {self.agent.total_reward:.1f}",
            True, BLACK
        )
        self.screen.blit(info_text, (10, WINDOW_SIZE + 10))
        
        # Resources info
        resource_text = self.font_small.render(
            f"Resources: {self.agent.resources_collected}/5 remaining | Obstacles hit: {self.agent.obstacles_hit}",
            True, BLACK
        )
        self.screen.blit(resource_text, (10, WINDOW_SIZE + 40))
        
        # Strategy info
        strategy_color = {
            'Aggressive': RED,
            'Balanced': ORANGE,
            'Conservative': GREEN
        }.get(self.agent.strategy, BLACK)
        
        strategy_text = self.font_medium.render(
            f"Strategy: {self.agent.strategy}",
            True, strategy_color
        )
        self.screen.blit(strategy_text, (10, WINDOW_SIZE + 70))
        
        # Decision count
        decision_text = self.font_small.render(
            f"Strategic decisions: {self.agent.decisions_made}",
            True, BLACK
        )
        self.screen.blit(decision_text, (10, WINDOW_SIZE + 105))
        
        # Strategy change notification
        if self.strategy_notification and self.notification_timer > 0:
            notif_text = self.font_small.render(
                f"Strategy changed: {self.strategy_notification['old_strategy']} → {self.strategy_notification['new_strategy']}",
                True, PURPLE
            )
            self.screen.blit(notif_text, (300, WINDOW_SIZE + 105))
            self.notification_timer -= 1
        
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
                    print("PAUSED" if self.paused else "RESUMED")
    
    def reset_episode(self):
        """Reset to new episode"""
        self.obs, _ = self.env.reset()
        self.agent = StrategicAgent("trained_agent", agent_id=1, color=BLUE)
        self.steps = 0
        self.episode += 1
        self.strategy_notification = None
        print(f"\n=== Episode {self.episode} ===")
    
    def step_game(self):
        """Execute one game step"""
        if self.paused:
            return
        
        # Agent chooses action with strategic decision-making
        action, strategy_info = self.agent.get_action(self.obs, self.env)
        
        # Execute action in environment
        self.obs, reward, done, _, _ = self.env.step(action)
        
        # Update agent stats
        self.agent.update_stats(reward)
        self.steps += 1
        
        # Handle strategy change notification
        if strategy_info and strategy_info['changed']:
            self.strategy_notification = strategy_info
            self.notification_timer = 60  # Show for 60 frames
            print(f"\n  Step {self.steps}: Strategy changed to {strategy_info['new_strategy']}")
            print(f"    Expected Value: {strategy_info['expected_value']:.2f}")
        
        # Print important events
        if reward == 10:
            print(f"  Step {self.steps}: Resource collected! Total: {self.agent.resources_collected}")
        elif reward == -5:
            print(f"  Step {self.steps}: Hit obstacle!")
        
        # Handle episode end
        if done:
            print(f"\nEpisode {self.episode} Complete!")
            print(f"  Total Reward: {self.agent.total_reward:.2f}")
            print(f"  Resources Collected: {self.agent.resources_collected}/5")
            print(f"  Strategic Decisions Made: {self.agent.decisions_made}")
            pygame.time.wait(2000)
            self.reset_episode()
    
    def run(self):
        """Main game loop"""
        print("\n" + "="*70)
        print("  INTEGRATED RL + MONTE CARLO DECISION GAME")
        print("="*70)
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset episode")
        print("  Q: Quit")
        print("\nThe agent uses:")
        print("  - Reinforcement Learning for movement")
        print("  - Monte Carlo simulation for strategic decisions")
        print("\nWatch the strategy indicator (agent's outer ring color):")
        print("  Red: Aggressive | Yellow: Balanced | Green: Conservative")
        print("\n" + "="*70)
        print(f"\n=== Episode {self.episode} ===")
        
        while self.running:
            self.handle_events()
            self.step_game()
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        print("\nGame closed. Thank you for playing!")

def main():
    """Start the integrated game"""
    try:
        game = IntegratedGame(model_path="trained_agent")
        game.run()
    except FileNotFoundError:
        print("\nError: trained_agent.zip not found!")
        print("Please run 'python train_agent.py' first to train an agent.")
    except Exception as e:
        print(f"\nError starting game: {e}")
        raise

if __name__ == "__main__":
    main()
