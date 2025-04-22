import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import heapq
import time

class Maze:
    def initialize_Maze(self, width, height, seed = None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
        self.grid = np.ones((2*height+1, 2*width+1), dtype=int)
        self.generate_maze()
        
    def generate_maze(self):
        Height, Width = self.height, self.width
        visited = [[False]*Width for i in range(Height)]
        def carve (row, column):
            visited[row][column] = True
            self.grid[2*row+1][2*column+1] = 0
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            for direction_row, direction_column in directions:
                new_row = row + direction_row
                new_column = column + direction_column
                if 0 <= new_row < Height and 0 <= new_column < Width and not visited[new_row][new_column]:
                    self.grid[2*row+1 + direction_row][2*column+1 + direction_column] = 0
                    carve(new_row, new_column)
        carve(0, 0)
        self.start = (1, 1)
        self.goal = (2*Height-1, 2*Width-1)
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
        
    def neighbors(self,pos):
        row, column = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for direction_row, direction_column in directions:
            new_row = row + direction_row
            new_column = column + direction_column
            if 0 <= new_row < self.grid.shape[0] and 0 <= new_column < self.grid.shape[1] and self.grid[new_row][new_column] == 0:
                if(self.grid[new_row][new_column] == 0):
                    yield (new_row, new_column)
                    
                    
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
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])
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
    maze.initialize_Maze(MAZE_WIDTH, MAZE_HEIGHT, seed = random.randint(0, 100))
    
    search_agent = SearchAgent()
    search_agent.initialize_Agent(maze, maze.start, maze.goal)
    
    print("Starting BFS Search")
    animate_search(maze, search_agent.bfs(), pause = 0.005)
    
    print("Starting A* Search")
    animate_search(maze, search_agent.a_star(), pause = 0.005)