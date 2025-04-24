import random
import time
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
from maze_search import Maze, SearchAgent

def plot_results(csv_path, size):
    df = pd.read_csv(csv_path)
    metrics = ["runtime", "nodes", "length"]
    titles = ["Runtime (ms)", "Nodes expanded", "Path length"]
    colors = {"A*": "tab:blue", "BFS": "tab:red"}

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 9),
                             sharex=True, constrained_layout=True)

    for ax, metric, title in zip(axes, metrics, titles):
        for algorithm, sub in df.groupby("algorithm"):
            ax.plot(sub["iteration"],
                    sub[metric],
                    marker="o", markersize=4,
                    linestyle="-",
                    label=algorithm,
                    color=colors[algorithm])
        ax.set_ylabel(title)
        ax.grid(axis="y", ls="--", alpha=.4)

    axes[-1].set_xlabel("Iteration")
    axes[0].legend(loc="upper left")
    fig.suptitle("Per-iteration performance"
                 + (f" | size {size}" if size else ""),
                 fontsize=14)
    fig.savefig("per_iter_dashboard.png", dpi=150)
    print(f"Figure saved → per_iter_dashboard.png")
    plt.show()
def measure_path_length(came_from, goal):

    length = 0

    current = goal
    while current in came_from:
        current = came_from[current]
        length += 1
    return length

def run_benchmark(alg, agent, goal):

    if alg == 'A*':
        gen = agent.a_star()
    elif alg == 'BFS':
        gen = agent.bfs()

    start_time = time.perf_counter()
    for visited, came_from in gen:
        pass
    runtime_ms = (time.perf_counter() - start_time) * 1000
    nodes_visited = len(visited)
    path_length = measure_path_length(came_from, goal)

    return runtime_ms, nodes_visited, path_length


def main():

    dimensions = input("Enter the number of dimensions you would like to benchmark (2 or 3): ")
    benchmark_maze = Maze()
    try:
        dimensions = int(dimensions)
    except ValueError:
        print("Invalid input. Please enter an integer.")
        sys.exit(1)

    if dimensions not in [2, 3]:
        print("Invalid input. Please enter 2 or 3.")
        sys.exit(1)

    if dimensions == 2:
        size = input("Enter the size of the maze (e.g., 20 for a 20x20 maze): ")
        try:
            size = int(size)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sys.exit(1)

        iterations = input("Enter the number of iterations for the benchmark: ")
        try:
            iterations = int(iterations)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sys.exit(1)
        if size <= 0 or iterations <= 0:
            print("Size and iterations must be positive integers.")
            sys.exit(1)

    elif dimensions == 3:
        print("3D maze iteration takes a much longer time than 2D. "
              "Anything above 5x5x5 will take a long time to run. Please be patient.\n")
        size = input("Enter the size of the maze (e.g., 5 for a 5x5x5 maze): ")
        try:
            size = int(size)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sys.exit(1)
        iterations = input("Enter the number of iterations for the benchmark: ")
        try:
            iterations = int(iterations)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            sys.exit(1)
        if size <= 0 or iterations <= 0:
            print("Size and iterations must be positive integers.")
            sys.exit(1)

    benchmark_agent = SearchAgent()
    plot_result = input("Would you like to plot the results? (y/n): ").strip().lower()
    if plot_result is not None and plot_result != 'y' and plot_result != 'n':
        print("Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)
    with open('benchmark_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "iteration", "dimensions", "size", "runtime", "nodes", "length"])

        for rep in range(0, iterations):
            seed = random.randint(0, 1000000)
            if dimensions == 2:
                benchmark_maze.initialize_Maze([size, size], seed)
                size_label = str(size) + "x" + str(size)
            elif dimensions == 3:
                benchmark_maze.initialize_Maze([size, size, size], seed)
                size_label = str(size) + "x" + str(size) + "x" + str(size)
            benchmark_agent.initialize_Agent(benchmark_maze, benchmark_maze.start, benchmark_maze.goal)
            for algorithm in ['A*', 'BFS']:
                print(f"Running iteration {rep + 1} of {iterations} for {dimensions}D maze with size {size_label} using {algorithm} algorithm.")

                runtime, nodes, length = run_benchmark(algorithm, benchmark_agent, benchmark_maze.goal)
                print(f"Algorithm: {algorithm}, Dimensions: {dimensions}, Size: {size_label}, "
                        f"Runtime: {runtime:.2f} ms, Nodes Visited: {nodes}, Path Length: {length}")


                writer.writerow([algorithm, rep+1, dimensions, size_label, runtime, nodes, length])
                print(f"Results saved to benchmark_results.csv")
    if plot_result == 'y':
        plot_results('benchmark_results.csv', size_label)

if __name__ == "__main__":
    main()