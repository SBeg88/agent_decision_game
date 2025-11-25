import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Game settings
GRID_SIZE = 10  # 10x10 grid
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
GRAY = (200, 200, 200)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.agents = []  # Will hold agent positions
        self.resources = []  # Resource locations
        self.obstacles = []  # Obstacle locations
        
        # Create random resources and obstacles
        self.generate_world()
    
    def generate_world(self):
        """Generate random resources and obstacles"""
        # Add 5 resources
        for _ in range(5):
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            self.resources.append(pos)
        
        # Add 8 obstacles
        for _ in range(8):
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            self.obstacles.append(pos)
    
    def draw(self, screen):
        """Draw the grid world"""
        # Draw grid lines
        for x in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, CELL_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))
        
        # Draw resources (green circles)
        for resource in self.resources:
            x, y = resource
            pygame.draw.circle(screen, GREEN, 
                             (x * CELL_SIZE + CELL_SIZE // 2, 
                              y * CELL_SIZE + CELL_SIZE // 2), 
                             CELL_SIZE // 3)
        
        # Draw obstacles (red squares)
        for obstacle in self.obstacles:
            x, y = obstacle
            pygame.draw.rect(screen, RED, 
                           (x * CELL_SIZE + 5, y * CELL_SIZE + 5, 
                            CELL_SIZE - 10, CELL_SIZE - 10))

class Agent:
    def __init__(self, agent_id, start_pos, color):
        self.id = agent_id
        self.pos = start_pos
        self.color = color
        self.score = 0
        self.inventory = []
    
    def move(self, direction, world):
        """Move agent in a direction if valid"""
        x, y = self.pos
        new_pos = self.pos
        
        if direction == "UP" and y > 0:
            new_pos = (x, y - 1)
        elif direction == "DOWN" and y < world.grid_size - 1:
            new_pos = (x, y + 1)
        elif direction == "LEFT" and x > 0:
            new_pos = (x - 1, y)
        elif direction == "RIGHT" and x < world.grid_size - 1:
            new_pos = (x + 1, y)
        
        # Check if move is valid (not an obstacle)
        if new_pos not in world.obstacles:
            self.pos = new_pos
            
            # Check if agent collected a resource
            if new_pos in world.resources:
                world.resources.remove(new_pos)
                self.score += 10
                print(f"Agent {self.id} collected resource! Score: {self.score}")
    
    def draw(self, screen):
        """Draw the agent"""
        x, y = self.pos
        pygame.draw.circle(screen, self.color, 
                         (x * CELL_SIZE + CELL_SIZE // 2, 
                          y * CELL_SIZE + CELL_SIZE // 2), 
                         CELL_SIZE // 4)
        
        # Draw agent ID
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, BLACK)
        screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 2 - 5, 
                          y * CELL_SIZE + CELL_SIZE // 2 - 8))

def main():
    """Main game loop"""
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Multi-Agent Decision Game")
    clock = pygame.time.Clock()
    
    # Create world and agents
    world = GridWorld()
    agents = [
        Agent(1, (0, 0), BLUE),
        Agent(2, (9, 9), (255, 165, 0))  # Orange
    ]
    
    running = True
    while running:
        screen.fill(WHITE)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Manual control for Agent 1 (for testing)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    agents[0].move("UP", world)
                elif event.key == pygame.K_DOWN:
                    agents[0].move("DOWN", world)
                elif event.key == pygame.K_LEFT:
                    agents[0].move("LEFT", world)
                elif event.key == pygame.K_RIGHT:
                    agents[0].move("RIGHT", world)
        
        # Agent 2 makes random moves (placeholder for AI)
        random_direction = np.random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        agents[1].move(random_direction, world)
        
        # Draw everything
        world.draw(screen)
        for agent in agents:
            agent.draw(screen)
        
        # Display scores
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"A1: {agents[0].score}  A2: {agents[1].score}", 
                                True, BLACK)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()