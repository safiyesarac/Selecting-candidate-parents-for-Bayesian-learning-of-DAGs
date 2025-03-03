import numpy as np

def is_dag(adj_matrix):
    """Check if 'adj_matrix' is a Directed Acyclic Graph (DAG)."""
    n = adj_matrix.shape[0]
    visited = [False]*n
    temp_mark = [False]*n

    def visit(node):
        if temp_mark[node]:
            # already in the recursion stack => cycle
            return False
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

def sample_one_dag(num_nodes, p_edge=0.2, max_attempts=1000):
    """
    Sample ONE DAG from a Bernoulli prior with edge probability p_edge, 
    ensuring it's acyclic. If we can't get a DAG after max_attempts, 
    raise an error.
    """
    for _ in range(max_attempts):
        # Random adjacency (i->j edges)
        # We'll exclude self-loops by setting the diagonal to 0 below
        rand_adj = (np.random.rand(num_nodes, num_nodes) < p_edge).astype(int)
        np.fill_diagonal(rand_adj, 0)

        if is_dag(rand_adj):
            return rand_adj
    raise ValueError(f"Could not sample a DAG without cycles in {max_attempts} attempts.")

def sample_n_dags(num_nodes, n_samples, p_edge=0.2):
    """
    Sample n_samples DAGs from the prior p(G) where each edge is 
    included with probability p_edge (if the result is acyclic).
    """
    dag_list = []
    for _ in range(n_samples):
        dag = sample_one_dag(num_nodes, p_edge=p_edge)
        dag_list.append(dag)
    return dag_list

def average_dags(dag_list):
    """
    Given a list of adjacency matrices (all shape=(num_nodes, num_nodes)),
    compute the average adjacency matrix.
    """
    if not dag_list:
        return None
    sum_matrix = np.zeros_like(dag_list[0], dtype=float)
    for dag in dag_list:
        sum_matrix += dag
    avg_matrix = sum_matrix / len(dag_list)
    return avg_matrix

# -------------------------
# Example Usage
# -------------------------
def get_consesus_dag(num_nodes=60,n_samples=1000,p_edge=0.01):
    np.random.seed(42)

    # 1) Sample n_samples DAGs
    dags = sample_n_dags(num_nodes, n_samples, p_edge=p_edge)

    # 2) Compute the average adjacency
    avg_adj = average_dags(dags)

    # Print results
    print(f"Sampled {len(dags)} DAGs.")
    print("Average adjacency matrix (empirical edge probability):\n", avg_adj)

    # Optional: threshold to get a 'consensus' DAG
    threshold = 0.35
    consensus_dag = (avg_adj >= threshold).astype(int)
    print(f"Consensus DAG (threshold={threshold}):\n", consensus_dag)
