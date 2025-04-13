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







def arcs_coverage(dag_ref, candidate_parents):
    """
    Compute the fraction of arcs in 'dag_ref' that are 'guessed' by 'candidate_parents'.
    For each arc p->c in dag_ref, we check if 'p' appears in at least one candidate set
    for node 'c'.
    
    Returns a float in [0, 1]. If dag_ref has no arcs, we define coverage = 1.0
    (no arcs => trivially covered).
    """
    # Collect all arcs from the reference DAG
    # dag_ref is { child : [ list_of_parents ] }
    arcs = []
    for child, parents in dag_ref.items():
        for p in parents:
            arcs.append((p, child))
    total_arcs = len(arcs)
    
    # If there are no arcs at all, define coverage to be 1.0 (instead of 0.0)
    if total_arcs == 0:
        return 1.0

    guessed = 0
    for (p, c) in arcs:
        csets = candidate_parents.get(c, None)
        if csets is None:
            # No candidate sets for this child => can't guess any arcs
            continue
        
        # Normalize to a list of sets/tuples so we can iterate
        if isinstance(csets, (int, tuple, set)):
            csets = [csets]

        # If 'p' appears in at least one candidate set for child 'c', we count it as guessed
        is_guessed = False
        for cset in csets:
            if isinstance(cset, int):
                cset = {cset}
            else:
                cset = set(cset)
            if p in cset:
                is_guessed = True 
                break
        
        if is_guessed:
            guessed += 1
    
    return guessed / total_arcs
dag_ref = {
    0: [],
    1: [],    # 0->1
    2: [],    # 0->2
    3: [],  # 0->3, 2->3
    4: [],  # 1->4, 3->4
}
# So arcs are (0->1), (0->2), (0->3), (2->3), (1->4), (3->4).
# total arcs = 6

candidate_parents = {
    0: [1],
    1: [(0,),(1,)],
    2: [(0,),(1,)],
    3: [(0,), (2,)],  # We guess 0->3 in the first set, 2->3 in the second set
    4: [(1,), (3,)], 
    # That means 1->4 is guessed (by set (1,)), 
    # and 3->4 is guessed (by set (3,)).
    # But we do NOT have (2,) for node 4 => wait, we do have (2,) but that 
    # doesn't guess 1->4 or 3->4. Actually we do have (1,) or (3,). That’s enough.
}

cov=arcs_coverage(dag_ref, candidate_parents)

print(cov)
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

def compute_mean_coverage(csv_path):
    """
    Reads the CSV file, groups by ['method', 'K'], 
    and computes mean coverage for coverage_prior and coverage_gobnilp.
    """
    # Read the data
    df_out = pd.read_csv(csv_path)
    
    # Group by method and K, then compute mean
    df_mean = df_out.groupby(["method", "K"], as_index=False)[
        ["coverage_prior"]
    ].mean()
    
    # Log the result
    logging.info("\nMean coverage across replicates:\n%s", df_mean)
    
    return df_mean

if __name__ == "__main__":
    # Set logging level to INFO so we can see the output in terminal
    logging.basicConfig(level=logging.INFO)

    # Update this path to the location of your CSV
    csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"
    
    df_mean_coverage = compute_mean_coverage(csv_file_path)
    
    # Print or do something else with df_mean_coverage
    print(df_mean_coverage)
import pandas as pd
import logging

def debug_k9_coverage(csv_path):
    """
    Loads the CSV into a DataFrame, prints out all rows where K=9,
    and shows min, max, and mean coverage.
    """
    # Read the data
    df_out = pd.read_csv(csv_path)

    # Filter rows where K=9
    df_k9 = df_out[df_out["K"] == 9].copy()

    # If you expect coverage=1, see if it’s actually less than 1 in some rows
    # For coverage_prior and coverage_gobnilp, print out relevant info
    print("Rows where K=9:")
    print(df_k9[["method", "K", "coverage_prior", "coverage_gobnilp"]])

    # Show statistics for coverage_prior and coverage_gobnilp
    print("\nCoverage statistics at K=9:")
    print(df_k9[["coverage_prior", "coverage_gobnilp"]].describe())

    # Group by method as well, if that helps
    print("\nCoverage at K=9 grouped by method:")
    print(df_k9.groupby("method")[["coverage_prior", "coverage_gobnilp"]].agg(["mean", "min", "max"]))

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"
    debug_k9_coverage(csv_file_path)
import pandas as pd

# Change this path to your actual CSV file
csv_file_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000.csv"

df_out = pd.read_csv(csv_file_path)

# Find rows where K=9 and coverage_prior is not 1
df_k9_not_full = df_out.loc[(df_out["K"] == 9) & (df_out["coverage_prior"] < 1)]

# Print them out
print("Rows where K=9 but coverage_prior < 1:\n")
print(df_k9_not_full)

# If you also want to check coverage_gobnilp:
df_k9_not_full_gobnilp = df_out.loc[(df_out["K"] == 9) & (df_out["coverage_gobnilp"] < 1)]
print("\nRows where K=9 but coverage_gobnilp < 1:\n")
print(df_k9_not_full_gobnilp)