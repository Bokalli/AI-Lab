import random  # Import the random library for generating random numbers
from collections import deque  # Import deque for BFS implementation

import pygame  # Import the Pygame library for game development

# Constants
GRID_SIZE = 20  # Size of the grid (20x20)
CELL_SIZE = 30  # Size of each cell in pixels
WINDOW_SIZE = GRID_SIZE * CELL_SIZE  # Total window size
OBSTACLE_COUNT = 40  # Number of obstacles to be placed in the grid
REWARD = 10  # Reward for reaching the target
PENALTY = -5  # Penalty for hitting an obstacle

# Colors
WHITE = (255, 255, 255)  # Color for the background
BLACK = (0, 0, 0)  # Color for obstacles
GREEN = (0, 255, 0)  # Color for grid lines
RED = (255, 0, 0)  # Color for the target
BLUE = (0, 0, 255)  # Color for the agent

# Initialize Pygame
pygame.init()  # Start Pygame
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))  # Create the game window
pygame.display.set_caption("Adaptive Grid World Agent")  # Set the window title


# Function to create the grid and place obstacles
def create_grid():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # Initialize grid with zeros
    for _ in range(OBSTACLE_COUNT):  # Loop to place obstacles
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)  # Random position
        if grid[y][x] == 0:  # Ensure no overlap with existing obstacles
            grid[y][x] = 1  # Mark as obstacle
    return grid  # Return the created grid


# Create the grid
grid = create_grid()  # Generate the grid with obstacles

# Agent and target positions
agent_pos = (0, 0)  # Initial position of the agent (top-left corner)
target_pos = (GRID_SIZE - 1, GRID_SIZE - 1)  # Position of the target (bottom-right corner)
grid[target_pos[1]][target_pos[0]] = 2  # Mark target in the grid


class Agent:
    def __init__(self, grid, start_pos):
        self.grid = grid  # Store the grid
        self.position = start_pos  # Set the agent's starting position
        self.score = 0  # Initialize score to zero

    def decide(self):
        """Use BFS to find the next move towards the target."""

        target = target_pos  # Set the target position
        queue = deque([self.position])  # Initialize the queue with the agent's current position
        visited = {self.position: None}  # Track visited cells and their predecessors

        # Start the BFS loop
        while queue:
            current = queue.popleft()  # Dequeue the first position to explore
            if current == target:  # Check if the current position is the target
                break  # Exit the loop if the target is reached

            x, y = current  # Deconstruct the current position into x and y coordinates
            # Explore each of the four possible directions (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (x + dx, y + dy)  # Calculate the neighbor's position
                # Check if the neighbor is within grid bounds
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                    # Ensure the neighbor is not an obstacle and hasn't been visited
                    if self.grid[neighbor[1]][neighbor[0]] != 1 and neighbor not in visited:
                        visited[neighbor] = current  # Mark the neighbor as visited and record its predecessor
                        queue.append(neighbor)  # Add the neighbor to the queue for further exploration

        # Backtrack to find the path from target to agent's current position
        path = []  # Initialize an empty list to store the path
        while current:  # Continue until there are no predecessors left
            path.append(current)  # Add the current position to the path
            current = visited[current]  # Move to the predecessor of the current position
        path.reverse()  # Reverse the path to get it from the agent to the target

        # Choose the next position based on the path
        if len(path) > 1:  # Ensure there is a next position to move to
            next_pos = path[1]  # Set the next position to the second position in the path
            self.update_score(next_pos)  # Update the score based on the next position
            return next_pos  # Return the next position for the agent to move to
        return self.position  # If no valid move, stay in the current position

    def update_score(self, next_pos):
        """Update the agent's score based on the next position."""

        if next_pos == target_pos:  # Check if the next position is the target
            self.score += REWARD  # Increase the score by the defined reward amount for reaching the target
        elif self.grid[next_pos[1]][next_pos[0]] == 1:  # Check if the next position is an obstacle
            self.score += PENALTY  # Decrease the score by the penalty amount for hitting an obstacle


# Main Loop
running = True  # Flag to control the main game loop
while running:
    grid = create_grid()  # Recreate the grid for a new game session
    agent = Agent(grid, agent_pos)  # Create a new agent with the grid

    while True:  # Game loop
        for event in pygame.event.get():  # Event handling
            if event.type == pygame.QUIT:  # Check for quit event
                running = False  # Exit the main loop
                break  # Exit the event loop

        window.fill(WHITE)  # Clear the window with a white background

        # Draw the grid
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)  # Calculate cell rectangle
                if grid[y][x] == 1:  # If the cell is an obstacle
                    pygame.draw.rect(window, BLACK, rect)  # Draw the obstacle
                elif grid[y][x] == 2:  # If the cell is the target
                    pygame.draw.rect(window, RED, rect)  # Draw the target
                elif (x, y) == agent.position:  # If the cell is the agent's position
                    pygame.draw.rect(window, BLUE, rect)  # Draw the agent
                pygame.draw.rect(window, GREEN, rect, 1)  # Draw grid lines

        # Decision-making
        next_position = agent.decide()  # Determine the next position for the agent
        agent.position = next_position  # Update the agent's position

        # Display score
        font = pygame.font.Font(None, 36)  # Create a font object for displaying the score
        score_text = font.render(f'Score: {agent.score}', True, BLACK)  # Render the score text
        window.blit(score_text, (10, 10))  # Blit the score text on the window

        pygame.display.flip()  # Update the display
        pygame.time.delay(200)  # Delay for visual effect

pygame.quit()  # Quit Pygame
