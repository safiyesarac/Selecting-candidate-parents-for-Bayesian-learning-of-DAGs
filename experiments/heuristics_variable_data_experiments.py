    
def get_true_parents(model_file):  
    import bnlearn as bn

# Load a predefined Bayesian network structure
    model = bn.import_DAG(model_file)
    adjmat = model["adjmat"]

    # Get the list of node names (assumed to be the same in the DataFrame index and columns)
    nodes = list(adjmat.columns)



    # Build a dictionary mapping each node to its list of parent nodes.
    model_dict = {}
    for node in nodes:
        # For each node, find all source nodes for which the value is True.
        # The expression `adjmat[node] == True` returns a boolean Series;
        # its index are the source node names.
        parent_nodes = list(adjmat.index[adjmat[node] == True])
        model_dict[node] = parent_nodes

    # 3. Define a conversion function that works without needing a desired order.
    def convert_bn_to_indices(model_dict, ordered_node_list):
        """
        Convert a BN to a dictionary mapping node indices to parent indices,
        respecting the user-supplied 'ordered_node_list'.
        """
        node_names = ordered_node_list  # Use the given order
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        bn_dict = {}
        for name in node_names:
            parents = model_dict.get(name, [])
            parent_indices = tuple(name_to_idx[p] for p in parents)
            bn_dict[name_to_idx[name]] = parent_indices

        return bn_dict

    ordered_nodes = list(adjmat.columns)
    return convert_bn_to_indices(model_dict, ordered_nodes)
# sumu and related modules

def compute_coverage_and_efficiency(true_parent_model, candidate_parent_model):
    """
    Computes coverage and efficiency for each node based on the true parent sets
    and the candidate parent sets.
    
    Parameters:
    - true_parent_model (dict): Dictionary where keys are node indices and values are sets of true parent nodes.
    - candidate_parent_model (dict): Dictionary where keys are node indices and values are sets of candidate parent nodes.
    
    Returns:
    - coverage_metrics (list): A list of coverage values for each node.
    - efficiency_metrics (list): A list of efficiency values for each node.
    - average_coverage (float): The average coverage across all nodes.
    - average_efficiency (float): The average efficiency across all nodes.
    """
    # Initialize lists to store per-node coverage and efficiency
    coverages = []
    efficiencies = []
    
    # Loop over all nodes in the true parent model
    for node in true_parent_model:
        # Get the true parents and candidate parents for this node
        true_parents = set(true_parent_model[node])
        candidate_parents = set(candidate_parent_model[node])
        
        # Calculate Coverage: how many true parents are in the candidate parent set
        if len(true_parents) > 0:
            coverage = len(true_parents.intersection(candidate_parents)) / len(true_parents)
        else:
            coverage = 1  # If no true parents, coverage is trivially 1 (nothing to cover)
        
        # Calculate Efficiency: how many candidate parents are true parents
        if len(candidate_parents) > 0:
            efficiency = len(true_parents.intersection(candidate_parents)) / len(candidate_parents)
        else:
            efficiency = 0  # If no candidate parents, efficiency is 0
        
        # Store the results for this node
        coverages.append(coverage)
        efficiencies.append(efficiency)
    
    # Compute aggregated metrics (average coverage and efficiency)
    average_coverage = sum(coverages) / len(coverages)
    average_efficiency = sum(efficiencies) / len(efficiencies)
    
    return coverages, efficiencies, average_coverage, average_efficiency


# Example usage:

# Define the true parent sets (converted_asia) and candidate parent sets (comparison_model)

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
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the bn model file.")
    parser.add_argument("--jkl_file", type=str, required=True,
                        help="Path to the Gobnilp .jkl scores file.")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to a CSV data file (optional), needed for certain sumu heuristics (mb, pc, ges, etc.).")
    parser.add_argument("--K_min", type=int, default=1,
                        help="Minimum K for candidate parents.")
    parser.add_argument("--K_max", type=int, default=10,
                        help="Maximum K for candidate parents.")
    parser.add_argument("--output_csv", type=str, default="efficiency_log.csv",
                        help="Path to output CSV with coverage fraction results.")
    parser.add_argument("--skip_after_seconds", type=int, default=300,
                        help="Time limit (in seconds) for each heuristic call. If exceeded, skip.")
    args = parser.parse_args()
    
    # 1. Parse JKL -> GobnilpScores
    parsed_scores = data_io.parse_gobnilp_jkl(args.jkl_file)
    scores = heuristics.GobnilpScores(parsed_scores)
    n = scores.n

    

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
        "top":         (cpa["top"],         {"scores": scores, "n": n}),
         "opt":         (cpa["opt"],         {"scores": scores, "n": n}),
        "mb":          (cpa["mb"],          {"data": mydata, "fill": "random"}),
        "pc":          (cpa["pc"],          {"data": mydata, "fill": "random"}),
        "ges":         (cpa["ges"],         {"scores": scores, "data": mydata, "fill": "top"}),
        "greedy":      (cpa["greedy"],      {"scores": scores}),
        "greedy-lite": (cpa["greedy-lite"], {"scores": scores}),
        "back-forth":  (cpa["back-forth"],  {"scores": scores, "data": scores.data}),
        "beam":        (heuristics.beam_bdeu,          {"scores": scores, "beam_size": 5}),
    }

    # 5. Loop over each algorithm, vary K, measure coverage, respect time limit
    #    We'll now store the number of data rows in the results, too.
    results = []  # will store tuples of (algorithm, K, coverage_fraction, num_data_rows)
    true_parents=get_true_parents(args.model_file)
    results.append(('True Parents :', true_parents ))
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
                continue

            
            coverages, efficiencies, avg_coverage, avg_efficiency = compute_coverage_and_efficiency(
               true_parents, candidate_parents
            )



            print("Aggregated Metrics:")
        
            print(f"  coverage={avg_coverage}, efficiency= {avg_efficiency} time={elapsed:.1f}s")
            results.append((algo_name, K, avg_coverage, avg_efficiency , candidate_parents))

    # 6. Save results to CSV
    df_res = pd.DataFrame(results, columns=["Algorithm", "K", "Coverage True Parents", "Efficiency True Parents", 'ParentSet'])
    df_res.to_csv(args.output_csv, index=False)
    print(f"\nCoverage results saved to: {args.output_csv}")
    if num_data_rows > 0:
        print(f"All coverage fractions above used dataset with {num_data_rows} rows.")
    else:
        print("No dataset was loaded (NumDataRows=0).")
    
    


if __name__ == "__main__":
    main()

