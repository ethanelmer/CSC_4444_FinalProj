import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import random
from collections import deque
import heapq
import time

class Maze:
    def initialize_Maze(self, dimensions : list, seed = None):
        #Set dimensions
        self.dimensions = dimensions
        self.ndim = len(dimensions)

        #Double check and set seed if not
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
                for step in [-1, 1]:
                    direction = [0] * self.ndim
                    direction[dim] = step
                    directions.append(direction)

            random.shuffle(directions)

            for direction in directions:
                new_pos = [position[i] + direction[i] for i in range(self.ndim)]
                
                if all(0 <= new_pos[i] < self.dimensions[i] for i in range(self.ndim)) and not visited[tuple(new_pos)]:
                    wall_pos = tuple(2*position[i] + 1 + direction[i] for i in range(self.ndim))
                    self.grid[wall_pos] = 0
                    
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
        #Initialize BFS
        start = self.maze.start
        goal = self.maze.goal
        nodes = deque([start])
        came_from = {start: None}
        visited = {start}
        while nodes:
            # Pop the first node from the queue
            current = nodes.popleft()
            if current == goal:
                break # Found the goal
            for neighbor in self.maze.neighbors(current):
                if neighbor not in visited:
                    # Add neighbor to visited set and queue
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    nodes.append(neighbor)
                yield visited, came_from
            yield visited, came_from
            
    def a_star(self):
        #Initialize A*
        start = self.maze.start
        goal = self.maze.goal
        
        # Heuristic function (Manhattan distance)
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
                    # Calculate tentative g_score
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor), neighbor))
                yield visited, came_from
            yield visited, came_from
            
def animate_search_3d(maze, search_gen, pause=0.02):
    """
    Visualize maze search algorithm in 3D for a 3D maze.
    
    Parameters:
    maze: The 3D Maze object
    search_gen: Generator from search algorithm (BFS, A*)
    pause: Pause duration between frames
    """
    if maze.ndim != 3:
        print("This visualization function is for 3D mazes only.")
        return
        
    depth, height, width = maze.grid.shape
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(0, width-1)
    ax.set_ylim(0, height-1)
    ax.set_zlim(0, depth-1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Function to create a cube at a specific position
    def plot_cube(position, color, alpha=0.3, edgecolor='k'):
        z, y, x = position  # Position in grid
        
        # Define vertices of the cube
        vertices = [
            [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
            [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
        ]
        
        # Define the faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
            [vertices[3], vertices[2], vertices[6], vertices[7]],  # Back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # Right
        ]
        
        # Create collection of polygons
        poly = Poly3DCollection(faces, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor(edgecolor)
        return ax.add_collection3d(poly)
    
    # Create initial visualization
    wall_positions = np.argwhere(maze.grid == 1)
    wall_collections = []
    for pos in wall_positions:
        # Use a sampling approach to reduce visual clutter while keeping structure visible
        if (pos[0] % 2 == 0 or pos[1] % 2 == 0 or pos[2] % 2 == 0) and np.random.random() < 0.1:
            wall_collections.append(plot_cube(tuple(pos), 'black', alpha=0.15, edgecolor='gray'))
    
    # Start and goal positions
    start_z, start_y, start_x = maze.start
    goal_z, goal_y, goal_x = maze.goal
    
    ax.scatter([start_x+0.5], [start_y+0.5], [start_z+0.5], color='green', s=200, marker='o', label='Start')
    ax.scatter([goal_x+0.5], [goal_y+0.5], [goal_z+0.5], color='red', s=200, marker='*', label='Goal')
    
    plot_cube(maze.start, 'green', alpha=0.6)
    plot_cube(maze.goal, 'red', alpha=0.6)
    
    # Track visited nodes
    visited_nodes = []
    visited_cubes = []
    
    # Process the search algorithm
    for visited, came_from in search_gen:
        for pos in visited:
            if pos not in visited_nodes and pos != maze.start and pos != maze.goal:
                visited_nodes.append(pos)
                # Add a semi-transparent blue cube for each visited node
                cube = plot_cube(pos, [0.6, 0.8, 1.0], alpha=0.4)
                ax.add_collection3d(cube)
                visited_cubes.append((pos, cube))
        
        ax.view_init(elev=30, azim=(time.time() * 10) % 360)
        plt.pause(pause)
    
    # Reconstruct path
    path = []
    node = maze.goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    
    # reduce the transparency of visited nodes that are not part of the path
    for pos, cube in visited_cubes:
        if pos not in path:
            cube.set_alpha(0.2)
    
    # Highlight the path with red cubes
    path_cubes = []
    for pos in path:
        if pos != maze.start and pos != maze.goal:  # Don't cover start/goal
            path_cubes.append(plot_cube(pos, [1.0, 0.6, 0.6], alpha=0.7))
    
    # Also draw a line through the path centers for better visibility
    path_x = [pos[2] + 0.5 for pos in path]
    path_y = [pos[1] + 0.5 for pos in path]
    path_z = [pos[0] + 0.5 for pos in path]
    path_line = ax.plot(path_x, path_y, path_z, 'y-', linewidth=3, label='Path')
    
    ax.legend()
    plt.title(f'3D Maze Solved - Path Length: {len(path)}')
    
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()

def animate_search(maze, search_gen, pause = 0.005):

    if maze.ndim > 3:
        print("This visualization function is for 2D and 3D mazes only.")
        return
    
    if maze.ndim == 3:
        animate_search_3d(maze, search_gen, pause)
        return
    
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
    maze.initialize_Maze([5, 5, 5], seed = random.randint(0, 100))
    
    search_agent = SearchAgent()
    search_agent.initialize_Agent(maze, maze.start, maze.goal)
    
    print("Starting BFS Search")
    animate_search(maze, search_agent.bfs(), pause = 0.005)
    
    print("Starting A* Search")
    animate_search(maze, search_agent.a_star(), pause = 0.005)