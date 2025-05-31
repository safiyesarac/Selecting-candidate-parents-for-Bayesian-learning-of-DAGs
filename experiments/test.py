import data_io
import heuristics

# parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/sachs/sachs_rounded_10000_tempered.jkl')
# scores = heuristics.GobnilpScores(parsed_scores)
# print(scores.scores[0])

# import sumu
# from sumu.candidates import candidate_parent_algorithm as cpa
# cpa.keys()

# opt=cpa["opt"]
# opt_parents=opt(7, scores=scores,n= scores.n)
# print("1 million datapoint" ,opt_parents)

# parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/sachs/sachs_500.jkl')
# scores = heuristics.GobnilpScores(parsed_scores)
# print(scores.scores[0])

# import sumu
# from sumu.candidates import candidate_parent_algorithm as cpa
# cpa.keys()

# opt=cpa["opt"]
# opt_parents_500=opt(3, scores=scores,n= scores.n)
# print("500 data point" ,opt_parents_500)
# import coverage 
# parents=coverage.get_true_parents("data/sachs/sachs_rounded.bif")
# print(parents)
# import data_io
# parsed_dags=data_io.parse_dag_file("/home/gulce/Downloads/thesis/temp_10000.txt")

# cov=coverage.coverage_fraction(parsed_dags[0],parsed_dags)
# print("cov ->", cov)

# opt=cpa["top"]
# opt_parents=opt(7, scores=scores,n= scores.n)
# print("1 million datapoint" ,opt_parents)

import sys
import re


import math
from decimal import Decimal, getcontext
import decimal




# def auto_tempered_gobnilp(input_file, output_file, target_max_diff=Decimal('-500'), precision=6):
#     with open(input_file, 'r') as fin:
#         original_lines = fin.readlines()

#     idx = 0
#     num_nodes = int(original_lines[idx].strip())
#     idx += 1

#     all_log_scores = []
#     positions = []  # Track the exact positions to replace scores later

#     # Read original scores and positions
#     for _ in range(num_nodes):
#         while idx < len(original_lines) and original_lines[idx].strip() == "":
#             idx += 1  # Skip empty lines

#         node_info = original_lines[idx].strip().split()
#         node_name, num_parent_sets = node_info[0], int(node_info[1])
#         idx += 1

#         parent_sets = []
#         for _ in range(num_parent_sets):
#             while idx < len(original_lines) and original_lines[idx].strip() == "":
#                 idx += 1  # Skip empty lines inside parent sets

#             line_parts = original_lines[idx].strip().split()
#             score = Decimal(line_parts[0])
#             parent_sets.append(score)
#             all_log_scores.append(score)
#             positions.append(idx)  # Remember the exact line to replace
#             idx += 1

#     # Automatic temperature selection
#     max_score, min_score = max(all_log_scores), min(all_log_scores)
#     max_diff = min_score - max_score
#     if max_diff == 0:
#         temperature = Decimal('1')
#     else:
#         temperature = max_diff / target_max_diff
#         if temperature < 1:
#             temperature = Decimal('1')

#     print(f"Automatically chosen temperature: {temperature}")

#     # Tempered normalization
#     tempered_scores = [lw / temperature for lw in all_log_scores]
#     max_tempered = max(tempered_scores)

#     sum_exp = sum(
#         (score - max_tempered).exp()
#         for score in tempered_scores
#         if (score - max_tempered) > Decimal('-700')
#     )
#     log_partition = max_tempered + sum_exp.ln()

#     # Final normalized and formatted scores
#     normalized_scores = [
#         f"{(score - log_partition):.{precision}f}" for score in tempered_scores
#     ]

#     # Write back the original file, replacing scores EXACTLY where they appeared
#     with open(output_file, 'w') as fout:
#         norm_idx = 0
#         for i, line in enumerate(original_lines):
#             if i in positions:
#                 parts = line.strip().split()
#                 # Replace only the score, keep the rest exactly as-is
#                 replaced_line = f"{normalized_scores[norm_idx]} {' '.join(parts[1:])}\n"
#                 fout.write(replaced_line)
#                 norm_idx += 1
#             else:
#                 fout.write(line)  # keep the line exactly as original


# # Example usage clearly and practically demonstrated:
# auto_tempered_gobnilp(
#     '/home/gulce/Downloads/thesis/data/sachs/sachs.jkl',
#     '/home/gulce/Downloads/thesis/data/sachs/sachs_tempered.jkl',
#     precision=6  # practical and recommended precision
# )







# def arcs_coverage(dag_ref, candidate_parents):
#     """
#     Compute the fraction of arcs in 'dag_ref' that are 'guessed' by 'candidate_parents'.
#     For each arc p->c in dag_ref, we check if 'p' appears in at least one candidate set
#     for node 'c'.
    
#     Returns a float in [0, 1]. If dag_ref has no arcs, we define coverage = 1.0
#     (no arcs => trivially covered).
#     """
#     # Collect all arcs from the reference DAG
#     # dag_ref is { child : [ list_of_parents ] }
#     arcs = []
#     for child, parents in dag_ref.items():
#         for p in parents:
#             arcs.append((p, child))
#     total_arcs = len(arcs)
    
#     # If there are no arcs at all, define coverage to be 1.0 (instead of 0.0)
#     if total_arcs == 0:
#         return 1.0

#     guessed = 0
#     for (p, c) in arcs:
#         csets = candidate_parents.get(c, None)
#         if csets is None:
#             # No candidate sets for this child => can't guess any arcs
#             continue
        
#         # Normalize to a list of sets/tuples so we can iterate
#         if isinstance(csets, (int, tuple, set)):
#             csets = [csets]

#         # If 'p' appears in at least one candidate set for child 'c', we count it as guessed
#         is_guessed = False
#         for cset in csets:
#             if isinstance(cset, int):
#                 cset = {cset}
#             else:
#                 cset = set(cset)
#             if p in cset:
#                 is_guessed = True 
#                 break
        
#         if is_guessed:
#             guessed += 1
    
#     return guessed / total_arcs
# dag_ref = {
#     0: [],
#     1: [],    # 0->1
#     2: [],    # 0->2
#     3: [],  # 0->3, 2->3
#     4: [],  # 1->4, 3->4
# }
# # So arcs are (0->1), (0->2), (0->3), (2->3), (1->4), (3->4).
# # total arcs = 6

# candidate_parents = {
#     0: [1],
#     1: [(0,),(1,)],
#     2: [(0,),(1,)],
#     3: [(0,), (2,)],  # We guess 0->3 in the first set, 2->3 in the second set
#     4: [(1,), (3,)], 
#     # That means 1->4 is guessed (by set (1,)), 
#     # and 3->4 is guessed (by set (3,)).
#     # But we do NOT have (2,) for node 4 => wait, we do have (2,) but that 
#     # doesn't guess 1->4 or 3->4. Actually we do have (1,) or (3,). That’s enough.
# }

# cov=arcs_coverage(dag_ref, candidate_parents)

# print(cov)
# import two_phase_global_coverage
# # r=two_phase_global_coverage.find_best_dag_referencer(5, 2)
# # print(r)
# from collections import Counter

# def check_parent_set_sizes(dags):
#     size_counter = Counter()
#     for dag in dags:
#         for parents in dag.values():
#             size_counter[len(parents)] += 1
#     return size_counter

# parsed = data_io.parse_dag_file("/home/gulce/Downloads/thesis/data/sachs/sachs_1000.txt")
# size_dist = check_parent_set_sizes(parsed)
# print(size_dist)

# from collections import defaultdict, Counter
    
# def most_frequent_parent_sets_k(dags, k):
#     """
#     For a list of DAGs (each DAG is {node: set_of_parents}), and an integer k,
#     determine for each node which parent set(s) of size k appear most often
#     across all DAGs.

#     Returns:
#       A dict: {node: [list_of_frozensets]}.
#     """
#     node_parent_counter = defaultdict(Counter)
    
#     # We first collect the set of all possible nodes, if not provided
#     if all_nodes is None:
#         all_nodes = set()
#         for dag in dags:
#             all_nodes.update(dag.keys())

#     for dag in dags:
#         for node, parents in dag.items():
#             if len(parents) == k:
#                 node_parent_counter[node][frozenset(parents)] += 1

#     node_most_frequent = {}
#     for node in all_nodes:  # loop over every node
#         parent_set_counts = node_parent_counter[node]  # a Counter()
#         if not parent_set_counts:
#             # no sets of size k for this node
#             node_most_frequent[node] = []
#         else:
#             max_count = max(parent_set_counts.values())
#             top_sets = [
#                 pset for pset, freq in parent_set_counts.items() if freq == max_count
#             ]
#             node_most_frequent[node] = top_sets

#     return node_most_frequent

# import data_io

# keep=most_frequent_parent_sets_k(parsed,3)
# print(keep)


import pandas as pd
import logging

# def compute_mean_coverage(csv_path):
#     """
#     Reads the CSV file, groups by ['method', 'K'], 
#     and computes mean coverage for coverage_prior and coverage_gobnilp.
#     """
#     # Read the data
#     df_out = pd.read_csv(csv_path)
    
#     # Group by method and K, then compute mean
#     df_mean = df_out.groupby(["method", "K"], as_index=False)[
#         ["coverage_prior"]
#     ].mean()
    
#     # Log the result
#     logging.info("\nMean coverage across replicates:\n%s", df_mean)
    
#     return df_mean

# if __name__ == "__main__":
#     # Set logging level to INFO so we can see the output in terminal
#     logging.basicConfig(level=logging.INFO)

#     # Update this path to the location of your CSV
#     csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"
    
#     df_mean_coverage = compute_mean_coverage(csv_file_path)
    
#     # Print or do something else with df_mean_coverage
#     print(df_mean_coverage)
# import pandas as pd
# import logging

# def debug_k9_coverage(csv_path):
#     """
#     Loads the CSV into a DataFrame, prints out all rows where K=9,
#     and shows min, max, and mean coverage.
#     """
#     # Read the data
#     df_out = pd.read_csv(csv_path)

#     # Filter rows where K=9
#     df_k9 = df_out[df_out["K"] == 9].copy()

#     # If you expect coverage=1, see if it’s actually less than 1 in some rows
#     # For coverage_prior and coverage_gobnilp, print out relevant info
#     print("Rows where K=9:")
#     print(df_k9[["method", "K", "coverage_prior", "coverage_gobnilp"]])

#     # Show statistics for coverage_prior and coverage_gobnilp
#     print("\nCoverage statistics at K=9:")
#     print(df_k9[["coverage_prior", "coverage_gobnilp"]].describe())

#     # Group by method as well, if that helps
#     print("\nCoverage at K=9 grouped by method:")
#     print(df_k9.groupby("method")[["coverage_prior", "coverage_gobnilp"]].agg(["mean", "min", "max"]))

# if __name__ == "__main__":
#     # Set up logging
#     logging.basicConfig(level=logging.INFO)
    
#     csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"
#     debug_k9_coverage(csv_file_path)
# import pandas as pd

# # Change this path to your actual CSV file
# csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"

# df_out = pd.read_csv(csv_file_path)

# # Find rows where K=9 and coverage_prior is not 1
# df_k9_not_full = df_out.loc[(df_out["K"] == 9) & (df_out["coverage_prior"] < 1)]

# # Print them out
# print("Rows where K=9 but coverage_prior < 1:\n")
# print(df_k9_not_full)

# # If you also want to check coverage_gobnilp:
# df_k9_not_full_gobnilp = df_out.loc[(df_out["K"] == 9) & (df_out["coverage_gobnilp"] < 1)]
# print("\nRows where K=9 but coverage_gobnilp < 1:\n")
# print(df_k9_not_full_gobnilp)
from collections import Counter, defaultdict
import itertools

def most_likely_parents(sampled_dags, K):
    """
    Parameters
    ----------
    sampled_dags : list[dict[int, set[int]]]
        Each element is a DAG represented as {child : set(parents)}.
    K : int
        Exact number of parents to return for every node.

    Returns
    -------
    dict[int, tuple[int]]
        node → K parents chosen by
        (i) larger edge‑frequency,
        (ii) larger joint‑frequency with already‑chosen parents,
        (iii) smaller parent index.
    """
    # ------------------------------------------------------------------
    # 0)  gather the full node set
    # ------------------------------------------------------------------
    all_nodes = set()
    for dag in sampled_dags:
        all_nodes.update(dag.keys())
        for ps in dag.values():
            all_nodes.update(ps)
    all_nodes = sorted(all_nodes)           # deterministic order

    # ------------------------------------------------------------------
    # 1)  single‑edge frequencies  p(child←parent)
    # ------------------------------------------------------------------
    edge_freq = defaultdict(Counter)        # child → Counter(parent → hits)
    for dag in sampled_dags:
        for child, parents in dag.items():
            for p in parents:
                edge_freq[child][p] += 1

    # ------------------------------------------------------------------
    # 2)  joint frequencies  p(child←{p,q})   (needed only for ties)
    # ------------------------------------------------------------------
    joint_freq = defaultdict(lambda: defaultdict(int))
    for dag in sampled_dags:
        for child, parents in dag.items():
            for p, q in itertools.combinations(parents, 2):
                joint_freq[child][frozenset((p, q))] += 1

    # ------------------------------------------------------------------
    # 3)  select K parents per child
    # ------------------------------------------------------------------
    result = {}
    for child in all_nodes:
        # ensure we have an entry even if the child never had parents
        c_freq = edge_freq[child]

        # give 0 frequency to never‑seen edges so we can still rank them
        for p in all_nodes:
            if p != child and p not in c_freq:
                c_freq[p] = 0

        # greedy tie‑aware selection
        chosen = []
        while len(chosen) < K:
            def sort_key(p):
                # primary: larger edge frequency
                primary = c_freq[p]
                # secondary: larger joint freq with already‑chosen parents
                secondary = sum(
                    joint_freq[child][frozenset((p, q))]
                    for q in chosen
                )
                # tertiary: smaller index (negate so that max() picks smallest)
                tertiary = -p
                return (primary, secondary, tertiary)

            # among not‑yet‑chosen parents, take the argmax of the key
            candidate = max(
                (p for p in all_nodes if p != child and p not in chosen),
                key=sort_key
            )
            chosen.append(candidate)

        result[child] = tuple(chosen)

    return result
import data_io
parsed_dags=data_io.parse_dag_file("/home/gulce/Downloads/thesis/data/sachs/sachs_1000.txt")
result=most_likely_parents(parsed_dags,5)
print(result)
import coverage
cov=coverage.coverage_fraction(result,parsed_dags)
print(cov)
parsed_scores=data_io.parse_gobnilp_jkl("/home/gulce/Downloads/thesis/data/sachs/sachs_1000.jkl")
scores = heuristics.GobnilpScores(parsed_scores)
n = scores.n


import heuristics
max_posterior=heuristics.maximize_true_graph_posterior(5, scores)
max_posterior_cov=coverage.coverage_fraction(max_posterior,parsed_dags)
max_posterior_acyclic=heuristics.maximize_true_graph_posterior_acyclic(5, scores)
max_posterior_cov_acyclic=coverage.coverage_fraction(max_posterior_acyclic,parsed_dags)
max_posterior_via_dags=heuristics.maximise_posterior_via_sampled_dags(5,scores,parsed_dags)
max_posterior_cov_via_dags=coverage.coverage_fraction(max_posterior_via_dags,parsed_dags)
print("post "+  " coverage :", max_posterior_cov,"  ",max_posterior) 
print("post acyclic "+  " coverage :", max_posterior_cov_acyclic,"  ",max_posterior_acyclic) 
print("post via dags  "+  " coverage :", max_posterior_cov_via_dags,"  ",max_posterior_via_dags) 
import sumu 
algo=sumu.candidate_parent_algorithm["opt"]
opt=algo(5,scores=scores,n=scores.n)
opt_coverage=coverage.coverage_fraction(opt, parsed_dags)
print("opt "+  " coverage :", opt_coverage,"  ",opt) 
import two_phase_global_coverage
two_phase_global_coverage.run_gobnilp_for_best_dag("/home/gulce/Downloads/thesis/data/sachs/sachs_1000.jkl","test")

# dot_file = f"test_gobnilp_solution.dot"
# dag_best = two_phase_global_coverage.parse_best_dag_dot(dot_file)
# print(f"[fGOBNILP] Parsed DAG from {dot_file}: {dag_best}")
# best_dict = {k: tuple(v) for k, v in dag_best.items()}
# gobnilp_cov=coverage.coverage_fraction(best_dict, parsed_dags)
# print(f"[fGOBNILP] Parsed DAG from {gobnilp_cov}: {dag_best}")
# true=coverage.get_true_parents("/home/gulce/Downloads/thesis/data/sachs/sachs.bif")
# print("true : ",true)

# def dag_log_bdeu(gob_scores, dag_dict, warn_if_missing=True):
#     """
#     Sum the local log‑BDeu scores that Gobnilp computed for the exact
#     parent‑sets in `dag_dict`.  Returns the global log‑BDeu score and the
#     per‑node breakdown.

#     Parameters
#     ----------
#     gob_scores : GobnilpScores      – the wrapper you pasted
#     dag_dict   : dict[int, tuple]   – node → parent‑tuple
#     """
#     per_node = {}
#     missing  = []
#     for v, pa in dag_dict.items():
#         s = gob_scores.local(v, pa)
#         per_node[v] = s
#         if math.isinf(s):
#             missing.append((v, pa))

#     if warn_if_missing and missing:
#         print("[warning] some parent sets were never scored by Gobnilp:")
#         for v, pa in missing:
#             print(f"  node {v}  parents {pa}")
#         print("Those terms contribute −∞, so the total is −∞.")
    
#     total = sum(per_node.values())
#     return total, per_node        # (global log‑BDeu, breakdown)


# from collections import Counter

# from collections import Counter
# from typing import Any, Dict, List, Tuple

# # ---------------------------------------------------------------------------
# # helpers: freeze / unfreeze so Counter can hash any nested structure
# # ---------------------------------------------------------------------------
# def _freeze(obj: Any) -> Any:
#     """Recursively convert obj into an immutable, hashable form."""
#     if isinstance(obj, dict):
#         return tuple(sorted((k, _freeze(v)) for k, v in obj.items()))
#     elif isinstance(obj, (list, tuple, set)):
#         return frozenset(_freeze(x) for x in obj)
#     else:
#         return obj                       # already hashable (int, str, …)

# def _unfreeze(obj: Any) -> Any:
#     """Restore the original (mutable) structure."""
#     if isinstance(obj, tuple) and obj and isinstance(obj[0], tuple):
#         return {k: _unfreeze(v) for k, v in obj}
#     elif isinstance(obj, frozenset):
#         return tuple(sorted(_unfreeze(x) for x in obj))
#     else:
#         return obj

# # ---------------------------------------------------------------------------
# # main function: returns ONE most‑frequent DAG
# # ---------------------------------------------------------------------------
# def most_frequent_dag(dag_list: List[Dict[int, Any]]
#                       ) -> Tuple[Dict[int, Any], int]:
#     """
#     Return (single_mode_dag, frequency).

#     If multiple DAGs tie, the first one encountered in dag_list is chosen.
#     """
#     if not dag_list:
#         raise ValueError("Input list is empty")

#     frozen = [_freeze(d) for d in dag_list]
#     counts = Counter(frozen)

#     # Counter.most_common(1) gives [(hashable_repr, freq)]
#     mode_hash, freq = counts.most_common(1)[0]

#     return _unfreeze(mode_hash), freq


# most_freq=most_frequent_dag(parsed_dags)
# total_log_bdeu, node_breakdown = dag_log_bdeu(scores, most_freq)

# print("Modular Sampler Global log‑BDeu score:", total_log_bdeu)
# print("Per‑node decomposition:")
# for v in sorted(node_breakdown):
#     print(f"  X{v} | Pa={ parsed_dags[0][v]} : {node_breakdown[v]}")
    
    

# total_log_bdeu, node_breakdown = dag_log_bdeu(scores, true)

# print(" True Parents Global log‑BDeu score:", total_log_bdeu)
# print("Per‑node decomposition:")
# for v in sorted(node_breakdown):
#     print(f"  X{v} | Pa={ true[v]} : {node_breakdown[v]}")
    
    
    

# total_log_bdeu, node_breakdown = dag_log_bdeu(scores, best_dict)

# print("Gobnilp Global log‑BDeu score:", total_log_bdeu)
# print("Per‑node decomposition:")
# for v in sorted(node_breakdown):
#     print(f"  X{v} | Pa={ true[v]} : {node_breakdown[v]}")