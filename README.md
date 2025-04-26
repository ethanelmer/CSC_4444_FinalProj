# CSC_4444_FinalProj
Final for CSC 4444 w/ Jassim

# N-Dimensional Maze Solver

This project implements a maze generation and solving algorithm that works in n-dimensions. The system includes visualization capabilities for 2D and 3D mazes, along with performance benchmarking tools to compare different search algorithms.

## Features

- Maze generation in 2D, 3D and theoretically any n-dimensions
- Search algorithms:
  - Breadth-First Search (BFS)
  - A* Search
- Interactive visualization for 2D and 3D mazes
- Benchmarking tool to compare algorithm performance

## Dependencies

The project requires the following Python packages:

```python
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CSC_4444_FinalProj
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Maze Search Visualization

To see the maze generation and search algorithms in action with visualization:

```bash
python maze_search.py
```

This will:
1. Generate a 5x5x5 3D maze
2. Visualize a BFS search solution 
3. Visualize an A* search solution

You can modify the dimensions and size in the `__main__` block of maze_search.py.

### Running the Benchmark

To compare the performance of BFS and A* algorithms:

```bash
python benchmark.py
```

The benchmark tool will prompt you to:
1. Enter the number of dimensions (2 or 3)
2. Enter the size of the maze
3. Enter the number of iterations to run
4. Choose whether to plot the results

The results are saved to `benchmark_results.csv` and an optional performance graph (`per_iter_dashboard.png`) will be generated if requested.

## Benchmark Interpretation

The benchmark measures and compares:
- **Runtime (ms)**: How long each algorithm takes to find a solution
- **Nodes expanded**: How many maze cells were visited during the search
- **Path length**: The length of the final solution path

A* typically visits fewer nodes but may have slightly higher computational overhead per node compared to BFS.

## Visualization Controls

- 2D visualization: Static image showing the maze, visited nodes, and final path
- 3D visualization: Interactive 3D model with:
  - Automatic rotation for better perspective
  - Semi-transparent visualization of walls, visited nodes, and solution path
  - Markers for start and goal positions

## Example Output

When running the benchmark, you'll see progress information:
```
Running iteration 1 of 10 for 2D maze with size 20x20 using A* algorithm.
Algorithm: A*, Dimensions: 2, Size: 20x20, Runtime: 42.16 ms, Nodes Visited: 187, Path Length: 69
Results saved to benchmark_results.csv
```

## Notes on Performance

- 3D mazes with sizes larger than 5x5x5 may take significantly longer to generate and solve
- For larger mazes, consider reducing the visualization pause time or turning off visualization entirely for pure benchmarking