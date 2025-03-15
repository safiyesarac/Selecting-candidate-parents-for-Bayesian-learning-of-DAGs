def has_cycle(graph):
    # Color each node: 0 = unvisited, 1 = visiting, 2 = visited
    color = {node: 0 for node in graph}
    
    def dfs(node):
        if color[node] == 1:  # Node is currently being visited (cycle detected)
            return True
        if color[node] == 2:  # Node has already been visited (no cycle here)
            return False
        
        color[node] = 1  # Mark the node as being visited
        for neighbor in graph[node]:
            if dfs(neighbor):  # Recursively visit neighbors
                return True
        
        color[node] = 2  # Mark the node as fully visited
        return False

    # Check for cycles in the entire graph
    for node in graph:
        if color[node] == 0:  # If unvisited
            if dfs(node):
                return True
    return False

# Define the graph
graph = {
    0: (),
    1: (0,),
    2: (1,),
    3: (1,),
    4: (1, 2),
    5: (0, 1, 3),
    6: (0, 1, 4),
    7: (1, 2, 4, 5, 6),
    8: (2, 3, 4, 6),
    9: (0, 1, 5, 7, 8),
    10: (2, 3, 7, 8),
    11: (2, 4, 5, 8, 10),
    12: (0, 4, 6, 8, 10),
    13: (2, 5, 6),
    14: (0, 2, 3, 7, 9, 11)
}

# Check for cycle in the graph
if has_cycle(graph):
    print("The graph contains a cycle.")
else:
    print("The graph is acyclic (no cycle detected).")
