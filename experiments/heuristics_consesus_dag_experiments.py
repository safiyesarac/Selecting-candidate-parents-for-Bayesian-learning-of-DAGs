#!/usr/bin/env python3

"""
experiment_consensus_data.py

Workflow:
  1) Read a file of sampled DAGs (each DAG is an adjacency matrix in text).
  2) Form a consensus DAG by thresholding edge frequencies.
  3) Assign discrete CPDs for that consensus DAG, then sample synthetic data.
  4) Compute BDeu local scores from the synthetic dataset for each node + possible parent set.
  5) Write these local scores to a Gobnilp-style .jkl file.
  6) Run your existing coverage fraction experiment script
     (e.g., experiment_heuristics_with_timeout.py)
     by passing the generated .jkl file and the DAG file you want to measure coverage against.
"""

import argparse
import numpy as np
import pandas as pd
import random
import os


# 1. Exact Parent-Set Coverage
def exact_coverage(true_parents, heuristic_output):
    coverage = []
    for node, true_parent_set in true_parents.items():
        if true_parent_set in heuristic_output.get(node, []):
            coverage.append(1)
        else:
            coverage.append(0)
    return np.mean(coverage)

# 2. Subset/Superset Coverage
def subset_superset_coverage(true_parents, heuristic_output):
    coverage = []
    for node, true_parent_set in true_parents.items():
        heuristic_parent_sets = heuristic_output.get(node, [])
        for heuristic_set in heuristic_parent_sets:
            if set(true_parent_set).issubset(set(heuristic_set)) or set(heuristic_set).issubset(set(true_parent_set)):
                coverage.append(1)
                break
        else:
            coverage.append(0)
    return np.mean(coverage)

# 3. Precision, Recall, F1
def precision_recall_f1(true_parents, heuristic_output):
    true_edges = set()
    heuristic_edges = set()
    
    # Collect edges from true parents and heuristic output
    for node, parents in true_parents.items():
        for parent in parents:
            true_edges.add((parent, node))
    
    for node, parents in heuristic_output.items():
        for parent in parents:
            heuristic_edges.add((parent, node))
    
    # Compute precision, recall, F1
    tp = len(true_edges & heuristic_edges)
    fp = len(heuristic_edges - true_edges)
    fn = len(true_edges - heuristic_edges)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

# 4. Structural Hamming Distance (SHD)
def structural_hamming_distance(true_parents, heuristic_output):
    true_edges = set()
    heuristic_edges = set()
    
    # Collect edges from true parents and heuristic output
    for node, parents in true_parents.items():
        for parent in parents:
            true_edges.add((parent, node))
    
    for node, parents in heuristic_output.items():
        for parent in parents:
            heuristic_edges.add((parent, node))
    
    # SHD = number of differences in edges (insertions, deletions, reversals)
    shd = len(true_edges.symmetric_difference(heuristic_edges))
    return shd

# 5. Rank-Based Metrics (Rank of True Parent Set)
def rank_coverage(true_parents, heuristic_output):
    ranks = []
    for node, true_parent_set in true_parents.items():
        heuristic_parent_sets = heuristic_output.get(node, [])
        try:
            rank = heuristic_parent_sets.index(true_parent_set) + 1  # Rank starts from 1
        except ValueError:
            rank = len(heuristic_parent_sets) + 1  # If not found, place at end
        ranks.append(rank)
    return np.mean(ranks)


# If you have a "data_io" or "heuristics" package, import them as needed:
# import data_io
# import heuristics
# import coverage

##############################################################################
# Utility: read DAGs from text file
##############################################################################

def topological_sort(adj):
    """
    Simple topological sort for a DAG 
    (assuming it's actually a DAG).
    """
    n = adj.shape[0]
    in_degree = np.sum(adj, axis=0)
    queue = [i for i in range(n) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop()
        order.append(node)
        for j in range(n):
            if adj[node, j] == 1:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    if len(order) < n:
        raise ValueError("Not a DAG (cycle found).")
    return order

def sample_data_from_dag_discrete(adj, n_samples=1000, arity=2, seed=42):
    """
    For simplicity, assume each variable is discrete with 'arity' states: {0..(arity-1)}.
    We'll assign random CPDs to each node given its parents, then forward-sample.

    Return: a numpy array of shape (n_samples, n_vars), each entry in [0..(arity-1)].
    Also return the random CPDs used (for reference).
    """
    random.seed(seed)
    np.random.seed(seed)

    n = adj.shape[0]
    # figure out a topological order
    topo = topological_sort(adj)

    # For each node, create a CPD: p(X_i | Parents(i))
    # The shape of a CPD for node i with k parents is: (arity^k, arity)
    # We'll store CPDs in a dict cpd[i], which is an array of shape (arity^k, arity).
    cpd = {}

    # We'll store an index function to go from parent assignments -> row in CPD
    parent_index_func = {}

    for node in topo:
        parents = np.where(adj[:, node] == 1)[0]
        k = len(parents)
        # number of possible parent configurations
        n_parent_configs = arity**k
        
        # random CPD table
        table = np.random.rand(n_parent_configs, arity)
        # normalize each row to sum=1
        table = table / table.sum(axis=1, keepdims=True)

        cpd[node] = table

        # We'll define a function to map (values_of_parents) -> row index
        # for node i in CPD table
        def make_index_function(parents_list):
            # closure over parents_list
            def index_of(parent_vals):
                # parent_vals is a list/tuple of length k
                idx = 0
                for val in parent_vals:
                    idx = idx * arity + val
                return idx
            return index_of

        parent_index_func[node] = make_index_function(parents)

    # Now, generate data by forward sampling in topological order
    data = np.zeros((n_samples, n), dtype=int)

    for node in topo:
        parents = np.where(adj[:, node] == 1)[0]
        idx_func = parent_index_func[node]
        for s in range(n_samples):
            # gather parent values
            pvals = data[s, parents]
            row_idx = idx_func(tuple(pvals))
            probs = cpd[node][row_idx, :]
            # sample X_node from the distribution
            x_val = np.random.choice(np.arange(arity), p=probs)
            data[s, node] = x_val

    return data, cpd



##############################################################################
# 4) Main workflow
##############################################################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold in [0..1] for including edge in consensus DAG.")
    parser.add_argument("--n_samples_data", type=int, default=1000,
                        help="Number of data samples to generate from the consensus DAG.")
    parser.add_argument("--arity", type=int, default=2,
                        help="Discrete arity for each variable (2=Bernoulli, etc.).")
    
    
    
    parser.add_argument("--log_csv", type=str, default="coverage_log.csv",
                        help="Where to store coverage fraction results, if coverage_dags is used.")
    args = parser.parse_args()

    # 1) Read DAGs & form consensus
    import compute_consesus_dag
    c_dag = compute_consesus_dag.get_consesus_dag()
    if c_dag is None:
        print("[ERROR] No DAGs parsed or empty list.")
        return

    n_vars = c_dag.shape[0]
    print(f"[INFO] Consensus DAG has shape: {c_dag.shape}")
    print(c_dag)

    # 2) Sample data from the consensus DAG (discrete)
    data, cpd = sample_data_from_dag_discrete(c_dag, 
                                              n_samples=args.n_samples_data,
                                              arity=args.arity)
    print(f"[INFO] Sampled {args.n_samples_data} rows of synthetic discrete data. shape={data.shape}")

    # 3) Compute local BDeu scores
    from datetime import datetime

# Get current date and time formatted safely for a filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import data_preparation
    data_preparation.save_data(data, '/home/gulce/Downloads/thesis/data/consesus/consesus_dag_'+timestamp+'.csv', '/home/gulce/Downloads/thesis/data/consesus/consesus_dag_'+timestamp+'.dat')
    data_preparation.compute_bdeu_scores(data, '/home/gulce/Downloads/thesis/data/consesus/consesus_dag_'+timestamp+'.dat', '/home/gulce/Downloads/thesis/data/consesus/consesus_dag_'+timestamp+'.jkl')
    print("[INFO] Computed local BDeu scores for each node and possible parent sets.")
    import data_io
    import heuristics
    parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/consesus/consesus_dag_'+timestamp+'.jkl')
    scores = heuristics.GobnilpScores(parsed_scores)
    n = scores.n
    

    
    
    
    # We'll define a toy function below for demonstration.




if __name__ == "__main__":
    main()
