import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import heapq

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