from enum import Enum


class SearchAlgorithm(str, Enum):
    ASTAR = "astar"
    HILL_CLIMBING = "hill_climbing"
    BFS = "bfs"
    DFS = "dfs"
