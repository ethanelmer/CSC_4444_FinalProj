import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import heapq
import time

class Maze:
    def initialize_Maze(self, dimensions : list, seed = None):
        self.dimensions = dimensions
        self.ndim = len(dimensions)

        if seed is not None:
            random.seed(seed)

        shape = tuple(2*dim + 1 for dim in dimensions)
        self.grid = np.ones(shape, dtype=int)

        self.generate_maze()
        
    def generate_maze(self):

        # initialize the visited nodes to false
        visited = np.zeros(self.dimensions, dtype=bool)

        def carve (position):
            visited[tuple(position)] = True

            grid_pos = tuple(2*pos + 1 for pos in position)
            self.grid[grid_pos] = 0

            # Create list of possible directions in n dimensions
            directions = []
            for dim in range(self.ndim):
                # For each dimension, can go +1 or -1
                for step in [-1, 1]:
                    direction = [0] * self.ndim
                    direction[dim] = step
                    directions.append(direction)

            random.shuffle(directions)

            # Try each direction
            for direction in directions:
                new_pos = [position[i] + direction[i] for i in range(self.ndim)]
                
                # Check if new position is valid
                if all(0 <= new_pos[i] < self.dimensions[i] for i in range(self.ndim)) and not visited[tuple(new_pos)]:
                    # Carve passage between cells
                    wall_pos = tuple(2*position[i] + 1 + direction[i] for i in range(self.ndim))
                    self.grid[wall_pos] = 0
                    
                    # Recursively carve from new position
                    carve(new_pos)

        # Start carving from origin
        start_pos = [0] * self.ndim
        carve(start_pos)

        self.start = tuple(1 for _ in range(self.ndim))
        self.goal = tuple(2*dim - 1 for dim in self.dimensions)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
    def neighbors(self,pos):
        for dim in range(self.ndim):
            for step in [-1, 1]:
                new_pos = list(pos)
                new_pos[dim] += step
                new_pos = tuple(new_pos)
                
                # Check if valid position and is a path
                if all(0 <= new_pos[i] < self.grid.shape[i] for i in range(self.ndim)) and self.grid[new_pos] == 0:
                    yield new_pos
                    
                    
class SearchAgent:
    def initialize_Agent(self, maze, goal, start):
        self.maze = maze
        
    def bfs(self):
        start = self.maze.start
        goal = self.maze.goal
        nodes = deque([start])
        came_from = {start: None}
        visited = {start}
        while nodes:
            current = nodes.popleft()
            if current == goal:
                break
            for neighbor in self.maze.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    nodes.append(neighbor)
                yield visited, came_from
            yield visited, came_from
            
    def a_star(self):
        start = self.maze.start
        goal = self.maze.goal
        def heuristic(position):
            return sum(abs(position[i] - goal[i]) for i in range(self.maze.ndim))
        open_set = []
        heapq.heappush(open_set, (heuristic(start), start))
        came_from = {start: None}
        g_score = {start: 0}
        visited = set()
        while open_set:
            i, current = heapq.heappop(open_set)
            if current == goal:
                break
            visited.add(current)
            for neighbor in self.maze.neighbors(current):
                if neighbor not in visited:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor), neighbor))
                yield visited, came_from
            yield visited, came_from
            
def animate_search(maze, search_gen, pause = 0.005):
    height = maze.grid.shape[0]
    width = maze.grid.shape[1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    image = np.zeros((height, width, 3), dtype=float)
    image[maze.grid == 1] = [0, 0, 0]  # walls
    image[maze.grid == 0] = [1, 1, 1]  # paths
    im = ax.imshow(image, interpolation='nearest')
    
    start_row, start_column = maze.start
    goal_row, goal_column = maze.goal
    ax.text(start_column, start_row, 'S', color='red', weight='bold',
            fontsize=14, ha='center', va='center')
    ax.text(goal_column, goal_row, 'G', color='red', weight='bold',
            fontsize=14, ha='center', va='center')
    
    for visited, came_from in search_gen:
        disp = image.copy()
        for (r,c) in visited:
            disp[r][c] = [0.6, 0.8, 1]
        im.set_data(disp)
        plt.pause(pause)
        
    path = []
    node = maze.goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    for (row,column) in path:
        image[row][column] = [1, 0.6, 0.6]
    image[maze.start] = [0, 1, 0]
    image[maze.goal] = [1, 0, 0]
    
    plt.title(f'Path Found with Length: {len(path)}')
    plt.show()
    
if __name__ == "__main__":
    MAZE_WIDTH = 20
    MAZE_HEIGHT = 20
    maze = Maze()
    maze.initialize_Maze([MAZE_WIDTH, MAZE_HEIGHT], seed = random.randint(0, 100))
    
    search_agent = SearchAgent()
    search_agent.initialize_Agent(maze, maze.start, maze.goal)
    
    print("Starting BFS Search")
    animate_search(maze, search_agent.bfs(), pause = 0.005)
    
    print("Starting A* Search")
    animate_search(maze, search_agent.a_star(), pause = 0.005)