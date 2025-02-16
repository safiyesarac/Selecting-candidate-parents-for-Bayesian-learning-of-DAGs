# takes jkl file k not gıven all k gıven then a txt fıle to measure coverage fractıon saves logs in a log fıle the results 
#!/usr/bin/env python3

"""
experiment_heuristics_with_timeout.py

This script:
  1) Reads a Gobnilp .jkl file (local scores) and parses it into a sumu-compatible Scores object.
  2) Reads a text file of sampled DAGs to compute coverage fraction against.
  3) Optionally reads a dataset CSV if needed by certain sumu algorithms (like 'mb', 'pc', 'ges').
  4) Runs several candidate-parent heuristics from sumu (and a custom beam search example) 
     for K in [K_min..K_max].
  5) Computes coverage fraction and logs results to a CSV file and prints to stdout.
  6) Skips any heuristic call that exceeds --skip_after_seconds (default 300s = 5 min).

Usage:
  python experiment_heuristics_with_timeout.py \
      --jkl_file data/hailfinder_scores.jkl \
      --sampled_dags data/hailfinder_sampled_dags.txt \
      --data_file data/hailfinder_dataset.csv \
      --K_min 1 \
      --K_max 30 \
      --skip_after_seconds 300 \
      --output_csv coverage_log.csv


"""

import argparse
import time
import numpy as np
import pandas as pd
import os
import data_io
import heuristics
import coverage 

# sumu and related modules
import sumu
from sumu.candidates import candidate_parent_algorithm as cpa


#(base) gulce@gulce-HP-Laptop:~/Downloads/thesis$ python experiments/heuristic_performance_experiments.py     --jkl_file data/asia_scores.jkl     --sampled_dags data/asia_sampled.txt     --data_file data/asia_dataset.csv     --K_min 1     --K_max 7    --skip_after_seconds 300     --output_csv data/coverage/asia_coverage_results.csv
#
##############################################################################
# 6. main()
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jkl_file", type=str, required=True,
                        help="Path to the Gobnilp .jkl scores file.")
    parser.add_argument("--sampled_dags", type=str, required=True,
                        help="Path to the text file containing sampled DAGs.")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to a CSV data file (optional), needed for certain sumu heuristics (mb, pc, ges, etc.).")
    parser.add_argument("--K_min", type=int, default=1,
                        help="Minimum K for candidate parents.")
    parser.add_argument("--K_max", type=int, default=10,
                        help="Maximum K for candidate parents.")
    parser.add_argument("--output_csv", type=str, default="coverage_log.csv",
                        help="Path to output CSV with coverage fraction results.")
    parser.add_argument("--sample_mode", type=str, default="exact modular",
                        help="exact sampling or mcmc")
    parser.add_argument("--skip_after_seconds", type=int, default=300,
                        help="Time limit (in seconds) for each heuristic call. If exceeded, skip.")
    args = parser.parse_args()

    # 1. Parse JKL -> GobnilpScores
    parsed_scores = data_io.parse_gobnilp_jkl(args.jkl_file)
    scores = heuristics.GobnilpScores(parsed_scores)
    n = scores.n

    # 2. Parse sampled DAGs
    sampled_dags = data_io.parse_dag_file(args.sampled_dags)

    # 3. Optionally load data for sumu algorithms that require the original data
    #    We'll also store the number of data rows in `num_data_rows`.
    mydata = None
    num_data_rows = 0
    if args.data_file and os.path.exists(args.data_file):
        # If the second row has e.g. arity or an extraneous row, adapt skiprows if needed:
        # df = pd.read_csv(args.data_file, skiprows=[1])
   
        df = pd.read_csv(args.data_file, skiprows=[1])

        mydata = sumu.Data(df.values, discrete=True)  
        num_data_rows = df.shape[0]
        print(f"[INFO] Loaded data file {args.data_file} with {num_data_rows} rows.")
    else:
        print("No data file given or file does not exist; 'mb', 'pc', 'ges' etc. might fail if used.")

    # 4. Define the candidate algorithms we want to test
    candidate_algos = {
        #"top":         (cpa["top"],         {"scores": scores, "n": n}),
        #  "opt":         (cpa["opt"],         {"scores": scores, "n": n}),
        # "mb":          (cpa["mb"],          {"data": mydata, "fill": "random"}),
        # "pc":          (cpa["pc"],          {"data": mydata, "fill": "random"}),
        # "ges":         (cpa["ges"],         {"scores": scores, "data": mydata, "fill": "top"}),
        # "greedy":      (cpa["greedy"],      {"scores": scores}),
        # "greedy-lite": (cpa["greedy-lite"], {"scores": scores}),
        # "back-forth":  (cpa["back-forth"],  {"scores": scores, "data": scores.data}),
        # "beam":        (heuristics.beam_bdeu,          {"scores": scores, "beam_size": 5}),
        # "marginal_bdeu_parents":        (heuristics.marginal_bdeu_parents,            {"scores": scores, "n": n})
        
         "voting_bdeu_parents":        (heuristics.bdeu_score_based_voting,            {"scores": scores}),
         "synergy": (heuristics.synergy_based_parent_selection,  {"scores": scores}),
        "stability":(heuristics.stability_bdeu, {"scores": scores, "data": mydata})
        
        
        
        
        
    }

    # 5. Loop over each algorithm, vary K, measure coverage, respect time limit
    #    We'll now store the number of data rows in the results, too.
    results = []  # will store tuples of (algorithm, K, coverage_fraction, num_data_rows)

    for algo_name, (algo_func, algo_kwargs) in candidate_algos.items():
        print(f"\n*** Running algorithm: {algo_name} ***")
        for K in range(args.K_min, args.K_max + 1):
            print(f"   [K={K}] ...", end="", flush=True)
            start_time = time.time()
            candidate_parents = None

            # Attempt to run the heuristic
            try:
                tmp_result = algo_func(K, **algo_kwargs)
                # Some sumu CPAs return (C, None) or (C, extra).
                if isinstance(tmp_result, tuple) and len(tmp_result) >= 1:
                    candidate_parents = tmp_result[0]
                else:
                    candidate_parents = tmp_result
            except Exception as e:
                # Could be an error if data wasn't provided for 'mb' or 'pc', or other issues
                print(f"  [ERROR] {e}")
                results.append((algo_name, K, None, num_data_rows))
                continue

            # Check elapsed time
            elapsed = time.time() - start_time
            if elapsed > args.skip_after_seconds:
                # If it took more than skip_after_seconds, skip it
                print(f"  [SKIPPED: took {elapsed:.1f}s > {args.skip_after_seconds}s]")
                results.append((algo_name, K, None, num_data_rows))
                break

            # Otherwise, measure coverage
            cf = coverage.coverage_fraction(candidate_parents, sampled_dags)
            print(f"  coverage={cf}, time={elapsed:.1f}s")
            results.append((algo_name, K, cf, num_data_rows))

    # 6. Save results to CSV
    df_res = pd.DataFrame(results, columns=["Algorithm", "K", "CoverageFraction", "NumDataRows"])
    df_res.to_csv(args.output_csv, index=False)
    print(f"\nCoverage results saved to: {args.output_csv}")
    if num_data_rows > 0:
        print(f"All coverage fractions above used dataset with {num_data_rows} rows.")
    else:
        print("No dataset was loaded (NumDataRows=0).")


if __name__ == "__main__":
    main()

