#!/usr/bin/env python3

"""
experiment_consensus_data.py

Workflow:
  1) Read a file of sampled DAGs (each DAG is an adjacency matrix in text).
  2) Form a consensus DAG by thresholding edge frequencies.
  3) Assign discrete CPDs for that consensus DAG, then sample synthetic data.
  4) Compute BDeu local scores from the synthetic dataset for each node + possible parent set.
  5) Write these local scores to a Gobnilp-style .jkl file.
  6) Run your existing coverage fraction experiment script by passing
     the generated .jkl file and the DAG file you want to measure coverage against.
"""

import argparse
import numpy as np
import pandas as pd
import random
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------------
# 1) Exact Coverage
# --------------------------------------------------------------------------
def exact_coverage(true_parents, heuristic_output):
    """
    Returns the fraction of nodes for which the heuristic
    exactly matches the true parent set for that node.
    """
    coverage = []
    for node, true_parent_set in true_parents.items():
        # Convert to tuple for consistent comparison
        true_parent_set = tuple(true_parent_set) if isinstance(true_parent_set, list) else true_parent_set

        # 'heuristic_output.get(node, [])' should be a list/tuple of candidate sets
        found = any(np.array_equal(true_parent_set, heuristic_set)
                    for heuristic_set in heuristic_output.get(node, []))
        coverage.append(1 if found else 0)
    return np.mean(coverage)

# --------------------------------------------------------------------------
# 2) Average Parent Coverage (node-by-node coverage of true parents)
# --------------------------------------------------------------------------
def average_parent_coverage(true_parents, heuristic_output):
    """
    For each node:
      - let T = true parent set
      - let P = predicted parent set
      Coverage for that node = |T ∩ P| / |T| (the fraction of true parents included)
      If T is empty, coverage is 1.0 for that node.

    Returns the average coverage across all nodes.
    """
    total_coverage = 0.0
    n = len(true_parents)

    for node, true_set in true_parents.items():
        pred_set = heuristic_output.get(node, ())
        if len(true_set) == 0:
            # If no true parents, count coverage as 1
            total_coverage += 1.0
        else:
            intersection_size = len(set(true_set).intersection(pred_set))
            total_coverage += intersection_size / len(true_set)

    return total_coverage / n

# --------------------------------------------------------------------------
# 3) Precision, Recall, F1
# --------------------------------------------------------------------------
def precision_recall_f1(true_parents, heuristic_output):
    """
    Edge-based precision/recall: each (parent->child) is an edge.
    """
    true_edges = set()
    heuristic_edges = set()

    # Collect edges from 'true_parents'
    for node, parents in true_parents.items():
        for parent in parents:
            true_edges.add((parent, node))

    # Collect edges from 'heuristic_output'
    for node, parents in heuristic_output.items():
        for parent in parents:
            heuristic_edges.add((parent, node))

    tp = len(true_edges & heuristic_edges)
    fp = len(heuristic_edges - true_edges)
    fn = len(true_edges - heuristic_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# --------------------------------------------------------------------------
# 4) Structural Hamming Distance (SHD)
# --------------------------------------------------------------------------
def structural_hamming_distance(true_parents, heuristic_output):
    """
    SHD = # of edges that differ between the true graph and the predicted graph.
    (Counting each mismatch in direction or presence/absence.)
    """
    true_edges = set()
    heuristic_edges = set()

    for node, parents in true_parents.items():
        for parent in parents:
            true_edges.add((parent, node))

    for node, parents in heuristic_output.items():
        for parent in parents:
            heuristic_edges.add((parent, node))

    return len(true_edges.symmetric_difference(heuristic_edges))

# --------------------------------------------------------------------------
# 5) Rank-Based Metric
# --------------------------------------------------------------------------
def rank_coverage(true_parents, heuristic_output):
    """
    If 'heuristic_output[node]' is a list of candidate sets in some rank order,
    find the position of the true parent set. Then average across nodes.
    """
    ranks = []
    for node, true_parent_set in true_parents.items():
        heuristic_parent_sets = heuristic_output.get(node, [])
        try:
            rank = heuristic_parent_sets.index(true_parent_set) + 1
        except ValueError:
            rank = len(heuristic_parent_sets) + 1
        ranks.append(rank)
    return np.mean(ranks)

# --------------------------------------------------------------------------
# 6) Adjacency -> Dictionary
# --------------------------------------------------------------------------
def adj_matrix_to_dict(adj_matrix):
    """
    Convert adjacency matrix into {node: (list_of_parents...)}.
    """
    n = adj_matrix.shape[0]
    consensus_dict = {}
    for node in range(n):
        parent_indices = np.where(adj_matrix[:, node] == 1)[0]
        consensus_dict[node] = tuple(parent_indices)
    return consensus_dict

# --------------------------------------------------------------------------
# 7) Sample Data
# --------------------------------------------------------------------------
def topological_sort(adj):
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
    import random
    random.seed(seed)
    np.random.seed(seed)

    n = adj.shape[0]
    topo = topological_sort(adj)

    # Build CPDs
    cpd = {}
    parent_index_func = {}

    for node in topo:
        parents = np.where(adj[:, node] == 1)[0]
        k = len(parents)
        n_parent_configs = arity**k
        table = np.random.rand(n_parent_configs, arity)
        table /= table.sum(axis=1, keepdims=True)
        cpd[node] = table

        def make_index_function(parents_list):
            def index_of(parent_vals):
                idx = 0
                for val in parent_vals:
                    idx = idx * arity + val
                return idx
            return index_of
        parent_index_func[node] = make_index_function(parents)

    data = np.zeros((n_samples, n), dtype=int)
    for node in topo:
        parents = np.where(adj[:, node] == 1)[0]
        idx_func = parent_index_func[node]
        for s in range(n_samples):
            pvals = data[s, parents]
            row_idx = idx_func(tuple(pvals))
            probs = cpd[node][row_idx, :]
            x_val = np.random.choice(np.arange(arity), p=probs)
            data[s, node] = x_val
    return data, cpd

# --------------------------------------------------------------------------
# 8) Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for including edges in consensus DAG.")
    parser.add_argument("--n_samples_data", type=int, default=1000,
                        help="Number of data samples to generate.")
    parser.add_argument("--arity", type=int, default=2,
                        help="Discrete arity for each variable.")
    parser.add_argument("--log_csv", type=str, default="coverage_log.csv",
                        help="Where to store coverage fraction results.")
    args = parser.parse_args()

    # 1) Build the consensus DAG
    import compute_consesus_dag
    c_dag = compute_consesus_dag.get_consensus_dag()
    if c_dag is None:
        print("[ERROR] No DAGs parsed or empty list.")
        return

    print(f"[INFO] Consensus DAG shape = {c_dag.shape}")
    print(c_dag)

    # 2) Sample data from DAG
    data, cpd = sample_data_from_dag_discrete(
        c_dag,
        n_samples=args.n_samples_data,
        arity=args.arity
    )
    print(f"[INFO] Sampled {args.n_samples_data} rows (shape={data.shape})")

    # 3) Compute local BDeu scores
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import data_preparation
    df_data = pd.DataFrame(data)
    import sumu
    from sumu.candidates import candidate_parent_algorithm as cpa
    import data_io
    import heuristics

    out_csv = f"/home/gulce/Downloads/thesis/data/consesus/consesus_dag_{timestamp}.csv"
    out_dat = f"/home/gulce/Downloads/thesis/data/consesus/consesus_dag_{timestamp}.dat"
    out_jkl = f"/home/gulce/Downloads/thesis/data/consesus/consesus_dag_{timestamp}.jkl"

    data_preparation.save_data(df_data, out_csv, out_dat)
    data_preparation.compute_bdeu_scores(out_dat, out_jkl)
    print("[INFO] Computed local BDeu scores for each node + parent set.")

    parsed_scores = data_io.parse_gobnilp_jkl(out_jkl)
    scores = heuristics.GobnilpScores(parsed_scores)
    n = scores.n

    df_sumu = pd.read_csv(out_csv, skiprows=[1])  # skip second line of metadata
    mydata = sumu.Data(df_sumu.values)
    num_data_rows = df_sumu.shape[0]

    # 4) Prepare candidate algos
    candidate_algos = {
        "top": (cpa["top"], {"scores": scores, "n": n}),
        "opt": (cpa["opt"], {"scores": scores, "n": n}),
        "mb": (cpa["mb"], {"data": mydata, "fill": "random"}),
        "pc": (cpa["pc"], {"data": mydata, "fill": "random"}),
        "ges": (cpa["ges"], {"scores": scores, "data": mydata, "fill": "top"}),
        "greedy": (cpa["greedy"], {"scores": scores}),
        "greedy-lite": (cpa["greedy-lite"], {"scores": scores}),
        "back-forth": (cpa["back-forth"], {"scores": scores, "data": scores.data}),
        "beam": (heuristics.beam_bdeu, {"scores": scores, "beam_size": 5}),
        "marginal_bdeu_parents": (heuristics.marginal_bdeu_parents, {"scores": scores, "n": n}),
        "voting_bdeu_parents": (heuristics.bdeu_score_based_voting, {"scores": scores}),
        "synergy": (heuristics.synergy_based_parent_selection, {"scores": scores}),
        "stability": (heuristics.stability_bdeu, {"scores": scores, "data": mydata}),
        "post": (heuristics.maximize_true_graph_posterior, {"scores": scores}),
    }

    # 5) Evaluate each algo at K=1..9
    consensus_parents = adj_matrix_to_dict(c_dag)
    results = []
    results.append(("Consensus Parents:", consensus_parents))

    import time
    for algo_name, (algo_func, algo_kwargs) in candidate_algos.items():
        print(f"\n*** Running algorithm: {algo_name} ***")
        for K in range(1, 10):
            print(f"   [K={K}] ...", end="", flush=True)
            start_time = time.time()
            candidate_parents = None
            try:
                tmp_result = algo_func(K, **algo_kwargs)
                if isinstance(tmp_result, tuple) and len(tmp_result) >= 1:
                    candidate_parents = tmp_result[0]
                else:
                    candidate_parents = tmp_result
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append((algo_name, K, None, "Error"))
                continue

            elapsed = time.time() - start_time
            if elapsed > 200:
                print(f"  [SKIPPED: took {elapsed:.1f}s > 200s]")
                results.append((algo_name, K, None, "Skipped"))
                break

            # Compute node-by-node coverage & other metrics
            exact_cov = exact_coverage(consensus_parents, candidate_parents)
            avg_cov = average_parent_coverage(consensus_parents, candidate_parents)
            precision, recall, f1 = precision_recall_f1(consensus_parents, candidate_parents)
            shd = structural_hamming_distance(consensus_parents, candidate_parents)
            rank_cov = rank_coverage(consensus_parents, candidate_parents)

            print(f"Metrics: Exact={exact_cov:.3f}, AvgCover={avg_cov:.3f}, "
                  f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, "
                  f"SHD={shd}, Rank={rank_cov:.3f}, time={elapsed:.1f}s")

            # Store each row
            results.append((
                algo_name,
                K,
                exact_cov,
                avg_cov,
                precision,
                recall,
                f1,
                shd,
                rank_cov,
                candidate_parents
            ))

    # 6) Save results to CSV (no subset/superset/jaccard)
    df_res = pd.DataFrame(results, columns=[
        "Algorithm",
        "K",
        "Exact Coverage",
        "Average Parent Coverage",
        "Precision",
        "Recall",
        "F1",
        "SHD",
        "Rank Coverage",
        "ParentSet"
    ])
    out_log_csv = f"/home/gulce/Downloads/thesis/data/consesus/consensus_log_{timestamp}.csv"
    df_res.to_csv(out_log_csv, index=False)
    print(f"[INFO] Results saved to {out_log_csv}")


if __name__ == "__main__":
    main()
