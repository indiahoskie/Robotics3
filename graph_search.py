import numpy as np
from collections import deque
import heapq
from .graph import Cell
from .utils import trace_path



def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm. This algorithm is optional for P3.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    # Stack for DFS (LIFO - Last In First Out)
    stack = [start]
    
    # Keep track of visited cells
    visited = set()
    visited.add((start.i, start.j))
    
    # Keep track of parent relationships
    parent = {}
    parent[(start.i, start.j)] = None
    
    while stack:
        # Pop from the end (stack behavior)
        current = stack.pop()
        
        # Mark as visited for visualization
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            # Reconstruct path using parent dictionary
            graph.parent_dict = parent
            return trace_path(goal, graph)
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for neighbor in neighbors:
            neighbor_tuple = (neighbor.i, neighbor.j)
            
            # If not visited and not in collision
            if neighbor_tuple not in visited and not graph.check_collision(neighbor.i, neighbor.j):
                visited.add(neighbor_tuple)
                parent[neighbor_tuple] = current
                stack.append(neighbor)
    
    # If no path was found, return an empty list.
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    # Queue for BFS (FIFO - First In First Out)
    queue = deque([start])
    
    # Keep track of visited cells
    visited = set()
    visited.add((start.i, start.j))
    
    # Keep track of parent relationships
    parent = {}
    parent[(start.i, start.j)] = None
    
    while queue:
        # Dequeue from the front
        current = queue.popleft()
        
        # Mark as visited for visualization
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            # Reconstruct path using parent dictionary
            graph.parent_dict = parent
            return trace_path(goal, graph)
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for neighbor in neighbors:
            neighbor_tuple = (neighbor.i, neighbor.j)
            
            # If not visited and not in collision
            if neighbor_tuple not in visited and not graph.check_collision(neighbor.i, neighbor.j):
                visited.add(neighbor_tuple)
                parent[neighbor_tuple] = current
                queue.append(neighbor)
    
    # If no path was found, return an empty list.
    return []


def a_star_search(graph, start, goal):
    """A* Search algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Make sure all the node values are reset.
    
    # Heuristic function (Manhattan distance)
    def heuristic(cell1, cell2):
        return abs(cell1.i - cell2.i) + abs(cell1.j - cell2.j)
    
    # Priority queue: (f_score, counter, cell)
    # counter is used to break ties consistently
    counter = 0
    pq = [(0, counter, start)]
    counter += 1
    
    # Keep track of visited cells
    visited = set()
    
    # Keep track of parent relationships
    parent = {}
    parent[(start.i, start.j)] = None
    
    # g_score: cost from start to current node
    g_score = {}
    g_score[(start.i, start.j)] = 0
    
    # f_score: g_score + heuristic
    f_score = {}
    f_score[(start.i, start.j)] = heuristic(start, goal)
    
    while pq:
        # Get cell with lowest f_score
        current_f, _, current = heapq.heappop(pq)
        current_tuple = (current.i, current.j)
        
        # Skip if already visited
        if current_tuple in visited:
            continue
        
        # Mark as visited
        visited.add(current_tuple)
        graph.visited_cells.append(Cell(current.i, current.j))
        
        # Check if we reached the goal
        if current.i == goal.i and current.j == goal.j:
            # Reconstruct path using parent dictionary
            graph.parent_dict = parent
            return trace_path(goal, graph)
        
        # Explore neighbors
        neighbors = graph.find_neighbors(current.i, current.j)
        for neighbor in neighbors:
            neighbor_tuple = (neighbor.i, neighbor.j)
            
            # Skip if already visited or in collision
            if neighbor_tuple in visited or graph.check_collision(neighbor.i, neighbor.j):
                continue
            
            # Calculate tentative g_score
            tentative_g = g_score[current_tuple] + 1  # Assuming uniform cost of 1
            
            # If this path to neighbor is better than any previous one
            if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                parent[neighbor_tuple] = current
                g_score[neighbor_tuple] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor_tuple] = f
                heapq.heappush(pq, (f, counter, neighbor))
                counter += 1
    
    # If no path was found, return an empty list.
    return []
