import pygame
import numpy as np
from stable_baselines3 import PPO
from rl_environment import AgentDecisionEnv

# Initialize Pygame
pygame.init()

# Game settings
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 5  # Slower so you can watch

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
GRAY = (200, 200, 200)

def draw_grid(screen, env):
    """Draw the game grid"""
    screen.fill(WHITE)
    
    # Draw grid lines
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))
    
    # Draw resources (green circles)
    for resource in env.resources:
        x, y = int(resource[0]), int(resource[1])
        pygame.draw.circle(screen, GREEN, 
                         (x * CELL_SIZE + CELL_SIZE // 2, 
                          y * CELL_SIZE + CELL_SIZE // 2), 
                         CELL_SIZE // 3)
    
    # Draw obstacles (red squares)
    for obstacle in env.obstacles:
        x, y = int(obstacle[0]), int(obstacle[1])
        pygame.draw.rect(screen, RED, 
                       (x * CELL_SIZE + 5, y * CELL_SIZE + 5, 
                        CELL_SIZE - 10, CELL_SIZE - 10))
    
    # Draw agent (blue circle)
    x, y = int(env.agent_pos[0]), int(env.agent_pos[1])
    pygame.draw.circle(screen, BLUE, 
                     (x * CELL_SIZE + CELL_SIZE // 2, 
                      y * CELL_SIZE + CELL_SIZE // 2), 
                     CELL_SIZE // 4)

def main():
    # Load the trained model
    model = PPO.load("trained_agent")
    
    # Create environment
    env = AgentDecisionEnv(grid_size=GRID_SIZE)
    
    # Setup display
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Trained AI Agent Demo")
    clock = pygame.time.Clock()
    
    # Reset environment
    obs, _ = env.reset()
    
    total_reward = 0
    episode = 1
    running = True
    
    print("Watch the AI agent navigate! Press Q to quit, R to reset.")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0
                    episode += 1
                    print(f"\n=== Episode {episode} ===")
        
        # Agent takes action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        total_reward += reward
        
        # Draw everything
        draw_grid(screen, env)
        
        # Display info
        font = pygame.font.Font(None, 36)
        info_text = font.render(f"Episode: {episode} | Reward: {total_reward:.1f} | Resources: {len(env.resources)}", 
                               True, BLACK)
        screen.blit(info_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Reset if episode done
        if done:
            print(f"Episode {episode} finished! Total reward: {total_reward:.2f}")
            pygame.time.wait(2000)  # Wait 2 seconds
            obs, _ = env.reset()
            total_reward = 0
            episode += 1
            print(f"\n=== Episode {episode} ===")
    
    pygame.quit()

if __name__ == "__main__":
    main()