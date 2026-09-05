# Import priority queue utilities for the A* open set
import heapq

# Import mathematical functions such as hypot()
import math

# Import NumPy for matrix/grid operations
import numpy as np

# Import binary dilation for obstacle inflation
from scipy.ndimage import binary_dilation


# A* path planning class
class AStarPlanner:

    # Initialize planner with occupancy grid
    def __init__(self, grid):

        # grid[y][x] = 0 free, 1 obstacle
        """
        grid[y][x] = 0 free, 1 obstacle
        """

        # Store occupancy grid
        self.grid = grid

        # Get grid height and width
        self.H, self.W = grid.shape


    # Heuristic function used by A*
    def heuristic(self, a, b):

        # Euclidean distance heuristic
        # Admissible when diagonal moves are allowed
        return math.hypot(a[0] - b[0], a[1] - b[1])


    # Check if coordinates are inside grid boundaries
    def in_bounds(self, x, y):

        # Return True if coordinates are valid
        return 0 <= x < self.W and 0 <= y < self.H


    # Check if a cell is free
    def is_free(self, x, y):

        # Return True if cell is not occupied
        return self.grid[y][x] == 0


    # Compute neighboring cells
    def neighbors(self, node):

        # Extract current node coordinates
        x, y = node

        # List that will contain neighbors
        neigh = []

        # Iterate through 8-connected motion model
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:

            # Compute neighbor coordinates
            nx, ny = x + dx, y + dy

            # Check if neighbor is inside map and free
            if self.in_bounds(nx, ny) and self.is_free(nx, ny):

                # Movement cost (1 for straight, sqrt(2) for diagonal)
                cost = math.hypot(dx, dy)

                # Add valid neighbor and movement cost
                neigh.append(((nx, ny), cost))

        # Return neighbor list
        return neigh


    # Main A* search algorithm
    def find_path(self, start, goal):

        # start, goal: (x, y) in pixel/grid coordinates
        """
        start, goal: (x, y) in pixel/grid coordinates
        """

        # Priority queue for frontier nodes
        open_set = []

        # Push starting node into heap
        heapq.heappush(open_set, (0.0, start))

        # Dictionary storing parent relationship
        came_from = {}

        # Cost-to-come dictionary
        g_score = {start: 0.0}

        # Continue until open set becomes empty
        while open_set:

            # Pop node with smallest f-score
            _, current = heapq.heappop(open_set)

            # Check if goal has been reached
            if current == goal:

                # Reconstruct and return path
                return self.reconstruct_path(came_from, current)

            # Explore neighbors of current node
            for neighbor, step_cost in self.neighbors(current):

                # Compute tentative path cost
                tentative_g = g_score[current] + step_cost

                # Update node if new path is better
                if neighbor not in g_score or tentative_g < g_score[neighbor]:

                    # Save parent node
                    came_from[neighbor] = current

                    # Update best known path cost
                    g_score[neighbor] = tentative_g

                    # Compute total estimated cost
                    f = tentative_g + self.heuristic(neighbor, goal)

                    # Push neighbor into priority queue
                    heapq.heappush(open_set, (f, neighbor))

        # Return None if no valid path exists
        return None  # path impossible


    # Reconstruct final path from parent dictionary
    def reconstruct_path(self, came_from, current):

        # Start path from goal node
        path = [current]

        # Follow parent chain backwards
        while current in came_from:

            # Move to parent node
            current = came_from[current]

            # Append parent node to path
            path.append(current)

        # Reverse path to obtain start -> goal order
        path.reverse()

        # Return reconstructed path
        return path


# Convert dense path into sparse waypoints
def path_to_waypoints(path, step=15):

    # path: list of (x,y)
    # step: every how many cells to create a waypoint
    """
    path: list of (x,y)
    step: every how many cells to create a waypoint
    """

    # Sample path every "step" cells
    waypoints = path[::step]

    # Ensure final goal point is included
    if waypoints[-1] != path[-1]:

        # Append final path point
        waypoints.append(path[-1])

    # Return waypoint list
    return waypoints




# Inflate occupied cells by a given radius
def inflate_obstacles(grid, radius):

    # Create empty structuring element
    structure = np.zeros((2*radius+1, 2*radius+1))

    # Compute center coordinates of kernel
    cy = cx = radius

    # Iterate over kernel rows
    for y in range(2*radius+1):

        # Iterate over kernel columns
        for x in range(2*radius+1):

            # Fill circular region inside inflation radius
            if math.hypot(x-cx, y-cy) <= radius:

                # Mark kernel cell as active
                structure[y, x] = 1

    # Apply binary dilation and return inflated obstacle map
    return binary_dilation(grid, structure=structure).astype(int)