

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
from typing import Dict, Set, List

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

########################
# (B) Parameter Sampling
#########################

def sample_cpts_for_dag(
    dag: dict, num_states: int, alpha: float = 1.0
) -> dict:
    """
    Given a DAG (dict: node -> set_of_parents), sample each node's CPT from a Dirichlet(alpha) prior.
    """
    cpts = {}
    d = len(dag)
    for node in range(d):
        parents = dag[node]
        num_parent_combos = num_states ** len(parents)
        cpt = np.zeros((num_parent_combos, num_states))
        for row_idx in range(num_parent_combos):
            theta = np.random.gamma(alpha, 1.0, size=num_states)
            theta /= theta.sum()  # normalize
            cpt[row_idx, :] = theta
        cpts[node] = cpt
    return cpts


#########################
# (C) Data Generation   #
#########################

def get_topological_order(dag: Dict[int, Set[int]]) -> List[int]:
    """
    Returns a topological ordering of the DAG (node->parents).
    Simple BFS/Kahn's algorithm or DFS-based. 
    """
    d = len(dag)
    in_degree = {node: 0 for node in range(d)}
    for child, parents in dag.items():
        for p in parents:
            in_degree[child] += 1
    
    queue = [n for n in range(d) if in_degree[n] == 0]
    topo_order = []
    while queue:
        cur = queue.pop()
        topo_order.append(cur)
        # "Remove" cur
        for child, parents in dag.items():
            if cur in parents:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    
    if len(topo_order) != d:
        raise ValueError("Graph is not acyclic or an error occurred. Cannot find topological order.")
    return topo_order

def generate_data_from_dag(
    dag: dict,
    cpts: dict,
    num_samples: int,
    num_states: int
) -> np.ndarray:
    """
    Generate synthetic data from a DAG with known CPTs.
    """
    d = len(dag)
    data = np.zeros((num_samples, d), dtype=int)
    topo_order = get_topological_order(dag)
    for s in range(num_samples):
        row_vals = [None] * d
        for node in topo_order:
            parents = dag[node]
            parent_list = sorted(parents)
            parent_idx = 0
            for p in parent_list:
                parent_idx = parent_idx * num_states + row_vals[p]
            probs = cpts[node][parent_idx, :]
            child_val = np.random.choice(num_states, p=probs)
            row_vals[node] = child_val
        data[s, :] = row_vals
    return data

# --------------------------------------------------------------------------
# 8) Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples_data", type=int, default=10,
                        help="Number of data samples to generate.")
    parser.add_argument("--arity", type=int, default=2,
                        help="Discrete arity for each variable.")
    parser.add_argument("--log_csv", type=str, default="coverage_log.csv",
                        help="Where to store coverage fraction results.")
    # Add arguments for example
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet prior parameter.")
    parser.add_argument("--data_in", type=str, 
                        default="/home/gulce/Downloads/thesis/data/fair/test.txt",
                        help="File containing sampled DAGs.")
    parser.add_argument("--replicates", type=int, default=1,
                        help="Number of replicate data sets per DAG.")
    parser.add_argument("--sample_sizes", type=str, default="1000,5000",
                        help="Comma-separated list of sample sizes to test.")
    args = parser.parse_args()
    import sampling
    sampling.sample_from_exact_modular_fair_sampler(25,15,"/home/gulce/Downloads/thesis/data/fair/test.txt")
    
    # Convert string to list of int
    sample_sizes = [int(x) for x in args.sample_sizes.split(",")]
    replicates = args.replicates
    alpha = args.alpha
    num_states = args.arity

    # ------------------------------------------------------------------
    # [1] Read or sample DAGs
    # Here you have your own logic to read 'sampled_dags' from a file.
    # Suppose data_io.parse_dag_file(...) returns a list of adjacency dicts:
    import data_io
    # e.g. each element is {node: [parents...], ...}
    sampled_dags = data_io.parse_dag_file(args.data_in)
    logging.info(f"Loaded {len(sampled_dags)} DAGs from {args.data_in}")

    # We will store *all results* for all DAGs into this list,
    # then average across DAGs at the very end.
    all_results = []

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # For each DAG in your file
    for dag_index, true_dag in enumerate(sampled_dags):
        d = len(true_dag)

        # For each requested sample size and replicate
        for size in sample_sizes:
            
                logging.info(f"[DAG #{dag_index}] sample_size={size}, rep=1")

                # [2] Sample parameters for the DAG
                cpts = sample_cpts_for_dag(true_dag, num_states=num_states, alpha=alpha)

                # [3] Generate data
                data_mat = generate_data_from_dag(
                    dag=true_dag,
                    cpts=cpts,
                    num_samples=size,
                    num_states=num_states
                )

                # [4] Compute local BDeu scores -> produce .jkl
                # Save data into .csv/.dat
                import pandas as pd
                df_data = pd.DataFrame(data_mat)
                
                # Build unique filenames for each DAG/rep
                out_csv = f"/home/gulce/Downloads/thesis/data/fair/tmp/consensus_dag_{dag_index}_{timestamp}.csv"
                out_dat = f"/home/gulce/Downloads/thesis/data/fair/tmp/consensus_dag_{dag_index}_{timestamp}.dat"
                out_jkl = f"/home/gulce/Downloads/thesis/data/fair/tmp/consensus_dag_{dag_index}_{timestamp}.jkl"

                import data_preparation
                df_data.to_csv(out_csv, index=False)
                # Or use your own method that writes the second line, etc.
                data_preparation.save_data(df_data, out_csv, out_dat)
                data_preparation.compute_bdeu_scores(out_dat, out_jkl)

                # [5] Parse .jkl file, create sumu Data, etc.
                import sumu
                import heuristics
                parsed_scores = data_io.parse_gobnilp_jkl(out_jkl)
                scores = heuristics.GobnilpScores(parsed_scores)
                n = scores.n

                # sumu requires a Data object
                df_sumu = pd.read_csv(out_csv, skiprows=[1])  # skip second line if needed
                mydata = sumu.Data(df_sumu.values)

                # [6] Prepare candidate algorithms
                from sumu.candidates import candidate_parent_algorithm as cpa
                candidate_algos = {
                    "top": (cpa["top"], {"scores": scores, "n": n}),
                    # "opt": (cpa["opt"], {"scores": scores, "n": n}),
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

                # Convert true_dag to a consistent dict-of-tuples for coverage
                # If your true_dag is already { node : [parents...] }, do:
                true_parents = {}
                for node, parlist in true_dag.items():
                    true_parents[node] = tuple(parlist)

                import time

                # Evaluate each algo at K=1..9
                for algo_name, (algo_func, algo_kwargs) in candidate_algos.items():
                    for K in range(1, scores.n):
                        start_time = time.time()
                        candidate_parents = None
                        try:
                            tmp_result = algo_func(K, **algo_kwargs)
                            # Some algorithms might return (candidate_parents, something_else)
                            if isinstance(tmp_result, tuple) and len(tmp_result) >= 1:
                                candidate_parents = tmp_result[0]
                            else:
                                candidate_parents = tmp_result
                        except Exception as e:
                            logging.warning(f"[Algo={algo_name}, K={K}] error: {e}")
                            # You might store a row with Nones:
                            all_results.append((
                                dag_index, size, 1,
                                algo_name, K,
                                None, None, None, None, None, None
                            ))
                            continue

                        elapsed = time.time() - start_time
                        # If you have a time cutoff:
                        if elapsed > 200:
                            logging.warning(f"[Algo={algo_name}, K={K}] SKIPPED, took {elapsed:.1f}s > 200s")
                            # store skip
                            all_results.append((
                                dag_index, size, 1,
                                algo_name, K,
                                None, None, None, None, None, None
                            ))
                            break

                        # Compute coverage metrics
                        exact_cov = exact_coverage(true_parents, candidate_parents)
                        avg_cov = average_parent_coverage(true_parents, candidate_parents)
                        precision, recall, f1 = precision_recall_f1(true_parents, candidate_parents)
                        shd = structural_hamming_distance(true_parents, candidate_parents)
                        rank_cov = rank_coverage(true_parents, candidate_parents)

                        # Append to big list
                        all_results.append((
                            dag_index, size, 1,
                            algo_name, K,
                            exact_cov,
                            avg_cov,
                            precision,
                            recall,
                            f1,
                            shd
                            # you could also include rank_cov if you want
                        ))
    # ------------------------------------------------------------------
    # [7] After finishing *all DAGs*, group by (Algorithm, K) 
    #     and compute the average across DAGs (and possibly across replicates).
    df_cols = [
        "dag_index", "sample_size", "replicate",
        "Algorithm", "K",
        "Exact Coverage", "Average Parent Coverage",
        "Precision", "Recall", "F1", "SHD"
    ]
    df_all = pd.DataFrame(all_results, columns=df_cols)




    # If you want a separate average per sample_size, do:
    df_agg = (
        df_all.groupby(["Algorithm", "K", "sample_size"], dropna=True)
              [["Exact Coverage", "Average Parent Coverage",
                "Precision", "Recall", "F1", "SHD"]]
              .mean()
              .reset_index()
    )

    # [8] Save the aggregated results
    out_log_csv = f"/home/gulce/Downloads/thesis/data/consesus/consensus_log_{timestamp}.csv"
    df_agg.to_csv(out_log_csv, index=False)
    print(f"[INFO] Averaged results saved to {out_log_csv}")

if __name__ == "__main__":
    main()




