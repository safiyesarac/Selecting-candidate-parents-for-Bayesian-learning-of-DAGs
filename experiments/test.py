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




def auto_tempered_gobnilp(input_file, output_file, target_max_diff=Decimal('-500'), precision=6):
    with open(input_file, 'r') as fin:
        original_lines = fin.readlines()

    idx = 0
    num_nodes = int(original_lines[idx].strip())
    idx += 1

    all_log_scores = []
    positions = []  # Track the exact positions to replace scores later

    # Read original scores and positions
    for _ in range(num_nodes):
        while idx < len(original_lines) and original_lines[idx].strip() == "":
            idx += 1  # Skip empty lines

        node_info = original_lines[idx].strip().split()
        node_name, num_parent_sets = node_info[0], int(node_info[1])
        idx += 1

        parent_sets = []
        for _ in range(num_parent_sets):
            while idx < len(original_lines) and original_lines[idx].strip() == "":
                idx += 1  # Skip empty lines inside parent sets

            line_parts = original_lines[idx].strip().split()
            score = Decimal(line_parts[0])
            parent_sets.append(score)
            all_log_scores.append(score)
            positions.append(idx)  # Remember the exact line to replace
            idx += 1

    # Automatic temperature selection
    max_score, min_score = max(all_log_scores), min(all_log_scores)
    max_diff = min_score - max_score
    if max_diff == 0:
        temperature = Decimal('1')
    else:
        temperature = max_diff / target_max_diff
        if temperature < 1:
            temperature = Decimal('1')

    print(f"Automatically chosen temperature: {temperature}")

    # Tempered normalization
    tempered_scores = [lw / temperature for lw in all_log_scores]
    max_tempered = max(tempered_scores)

    sum_exp = sum(
        (score - max_tempered).exp()
        for score in tempered_scores
        if (score - max_tempered) > Decimal('-700')
    )
    log_partition = max_tempered + sum_exp.ln()

    # Final normalized and formatted scores
    normalized_scores = [
        f"{(score - log_partition):.{precision}f}" for score in tempered_scores
    ]

    # Write back the original file, replacing scores EXACTLY where they appeared
    with open(output_file, 'w') as fout:
        norm_idx = 0
        for i, line in enumerate(original_lines):
            if i in positions:
                parts = line.strip().split()
                # Replace only the score, keep the rest exactly as-is
                replaced_line = f"{normalized_scores[norm_idx]} {' '.join(parts[1:])}\n"
                fout.write(replaced_line)
                norm_idx += 1
            else:
                fout.write(line)  # keep the line exactly as original


# Example usage clearly and practically demonstrated:
auto_tempered_gobnilp(
    '/home/gulce/Downloads/thesis/data/enginefuel/enginefuel_10000.jkl',
    '/home/gulce/Downloads/thesis/data/enginefuel/enginefuel_10000_tempered.jkl',
    precision=6  # practical and recommended precision
)







