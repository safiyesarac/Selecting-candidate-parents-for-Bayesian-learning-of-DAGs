import numpy as np
import random

def is_dag(adj_matrix):
    """Return True if 'adj_matrix' is a DAG (no cycles), False otherwise."""
    n = adj_matrix.shape[0]
    visited = [False] * n
    temp_mark = [False] * n

    def visit(node):
        if temp_mark[node]:
            return False  # cycle
        if not visited[node]:
            temp_mark[node] = True
            for nxt in range(n):
                if adj_matrix[node, nxt] == 1:
                    if not visit(nxt):
                        return False
            temp_mark[node] = False
            visited[node] = True
        return True

    for i in range(n):
        if not visited[i]:
            if not visit(i):
                return False
    return True

def sample_one_dag(num_nodes, p_edge=0.2):
    """Sample ONE DAG from a uniform prior, ensuring acyclicity."""
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Only consider edges from i -> j
            if np.random.rand() < p_edge:
                adj_matrix[i, j] = 1
    return adj_matrix

def sample_n_dags(num_nodes, n_samples, p_edge=0.2):
    """Sample multiple DAGs from a uniform prior."""
    dag_list = []
    for _ in range(n_samples):
        dag = sample_one_dag(num_nodes, p_edge)
        while not is_dag(dag):  # Ensure it's a valid DAG
            dag = sample_one_dag(num_nodes, p_edge)
        dag_list.append(dag)
    return dag_list

def average_dags(dag_list):
    """Compute the weighted average adjacency matrix of the DAGs."""
    if not dag_list:
        return None
    sum_matrix = np.zeros_like(dag_list[0], dtype=float)
    for dag in dag_list:
        sum_matrix += dag
    return sum_matrix / len(dag_list)

def build_consensus_dag(avg_adj, threshold=0.5):
    """Construct a consensus DAG from the average adjacency matrix, ensuring acyclicity."""
    n = avg_adj.shape[0]
    consensus = np.zeros((n, n), dtype=int)
    
    # Sort edges by average weight and add edges that do not create cycles
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if avg_adj[i, j] >= threshold:
                edges.append((i, j))
    
    # Try adding edges one by one, ensuring no cycles
    for i, j in edges:
        consensus[i, j] = 1
        if not is_dag(consensus):  # If adding this edge creates a cycle, remove it
            consensus[i, j] = 0
    
    return consensus

def get_consensus_dag(num_nodes=15, n_samples=1000, p_edge=0.2, threshold=0.203):
    """Generate a consensus DAG (average Bayesian network) from a uniform prior."""
    np.random.seed(42)

    # Sample DAGs from a uniform prior
    dags = sample_n_dags(num_nodes, n_samples, p_edge=p_edge)
    print(f"Sampled {len(dags)} DAGs.")
    
    # Compute the average adjacency matrix
    avg_adj = average_dags(dags)
    print("Average adjacency matrix:\n", avg_adj)
    
    # Build the consensus DAG from the average adjacency matrix
    consensus_dag = build_consensus_dag(avg_adj, threshold=threshold)
    
    print(f"\nConsensus DAG (threshold={threshold}):\n", consensus_dag)
    print("Is it acyclic?", is_dag(consensus_dag))

    return consensus_dag

# # Example usage
get_consensus_dag()
