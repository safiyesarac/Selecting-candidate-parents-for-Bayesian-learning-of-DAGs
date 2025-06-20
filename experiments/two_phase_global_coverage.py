

"""
two_phase_nodelevel_coverage.py (Debug Version)

Implements a 2-phase approach to measure coverage *node-by-node*:

  For each replicate r in 1..M:
    1) Sample (G*, cpts*) from a 'fair' prior
    2) Generate a small phantom dataset => pass to heuristic => for each node, returns candidate parent sets
    3) Generate a large reference dataset => run Gobnilp solver => obtains best DAG_ref
    4) coverage_r = 1 if for EVERY node i, the DAG_ref[i] is in heuristic's sets[i], else 0

At the end, we average coverage_r to get the final coverage fraction.
"""

import argparse
import numpy as np
import pandas as pd
import os
import random
import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def save_data(df,csv_file, dat_file):
    columns = [str(i) for i in range(len(df.columns))]
    print(columns)
    
    data = [[2] *( len(columns)+0)]
    
    df_arity = pd.DataFrame(data, columns=columns)

    column_mapping = {old: new for old, new in zip(df.columns, df_arity.columns)}
    df = df.rename(columns=column_mapping)

    
    df_combined_correct = pd.concat([df_arity, df], ignore_index=True)
    df_combined_correct.to_csv(csv_file, index=False)


    
    df_combined_correct = pd.concat([df_arity, df])
    
    
    
    df_combined = pd.concat([df_arity, df], ignore_index=True)

    
    df_combined.to_csv(dat_file, index=False, sep=' ')    
    




def sample_dag_fair_prior(n_vars, max_in_degree):
    """
    Sample exactly one random DAG on n_vars nodes
    from a 'fair' prior, ensuring no node has in-degree
    exceeding max_in_degree.
    """
    import sampling
    import data_io

    tmp_file = f"_tmp_dag_{random.randint(0,999999)}.txt"
    logging.debug(f"[sample_dag_fair_prior] About to sample DAG; writing to {tmp_file}")

    
    n_dags_to_sample = 1
    sampling.sample_from_exact_modular_fair_sampler(
        n_vars, max_in_degree, n_dags_to_sample, tmp_file
    )

    dag_list = data_io.parse_dag_file(tmp_file)
    logging.debug(f"[sample_dag_fair_prior] Parsed {len(dag_list)} DAG(s) from {tmp_file}")

    
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    if not dag_list or len(dag_list) == 0:
        raise ValueError("No DAG was sampled from the fair prior.")
    dag = dag_list[0]
    logging.debug(f"[sample_dag_fair_prior] Final DAG: {dag}")

    return dag


def sample_cpts_for_dag(dag, alpha=1.0, num_states=2):
    """
    Sample random CPTs for each node given its parents,
    using a Dirichlet( alpha ) prior for each configuration.
    """
    import numpy as np
    cpts = {}
    for node, parents in dag.items():
        k = len(parents)
        n_config = num_states ** k
        table = np.zeros((n_config, num_states))
        for cfg in range(n_config):
            theta = np.random.gamma(alpha, size=num_states)
            theta /= theta.sum()
            table[cfg, :] = theta
        cpts[node] = table

    logging.debug("[sample_cpts_for_dag] Sampled CPTs for each node.")
    return cpts

def get_topological_order(dag):
    """
    Return a valid topological ordering of the nodes in 'dag'.
    dag = { child : [ list_of_parents ] }.
    """
    d = len(dag)
    in_deg = {n: 0 for n in dag}
    for child, pars in dag.items():
        for p in pars:
            in_deg[child] += 1

    queue = [x for x in in_deg if in_deg[x] == 0]
    order = []
    while queue:
        cur = queue.pop()
        order.append(cur)
        for child, ps in dag.items():
            if cur in ps:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

    if len(order) < d:
        raise ValueError("[get_topological_order] Graph is not a DAG or has a cycle.")

    logging.debug(f"[get_topological_order] Topological order = {order}")
    return order

def generate_data_from_dag(dag, cpts, sample_size, num_states=2):
    """
    Generate synthetic data from 'dag' and 'cpts',
    returning a numpy array of shape (sample_size, len(dag)).
    """
    import numpy as np
    d = len(dag)
    data = np.zeros((sample_size, d), dtype=int)
    topo = get_topological_order(dag)
    for s in range(sample_size):
        row = [None] * d
        for node in topo:
            par = dag[node]
            idx = 0
            for p in sorted(par):
                idx = idx * num_states + row[p]
            pvals = cpts[node][idx, :]
            val = np.random.choice(num_states, p=pvals)
            row[node] = val
        data[s, :] = row

    logging.debug(f"[generate_data_from_dag] Generated data of shape {data.shape} for sample_size={sample_size}.")
    return data
def get_sumu_scores_for_small_data(data_small, replicate_idx=0):
    """
    Generate local BDeu .jkl file for data_small, parse it with sumu.
    Return a sumu.scores object we can reuse for multiple heuristics.
    """
    import pandas as pd
    import data_preparation
    import data_io
    import heuristics
    prefix = f"smalldata_rep{replicate_idx}"
    csv_file = f"{prefix}.csv"
    dat_file = f"{prefix}.dat"
    jkl_file = f"{prefix}.jkl"

    df_small = pd.DataFrame(data_small)
    df_small.to_csv(csv_file, index=False)
    data_preparation.save_data(df_small, csv_file, dat_file)
    data_preparation.compute_bdeu_scores(dat_file, jkl_file)

    parsed_scores = data_io.parse_gobnilp_jkl(jkl_file)

    scores = heuristics.GobnilpScores(parsed_scores)  

    for f in [csv_file, dat_file, jkl_file]:
        if os.path.exists(f):
            os.remove(f)

    return scores




def run_heuristic_on_small_data(data_small, replicate_idx=0):
    """
    We'll produce node->list_of_parent_sets for each node
    by using sumu with "greedy" (K=n).
    Then we won't unify them into a single DAG,
    but store them as candidate_parents[node] = [that single set or sets].
    """
    import pandas as pd
    import data_preparation
    import data_io
    import heuristics
    import sumu
    from sumu.candidates import candidate_parent_algorithm as cpa

    prefix = f"smalldata_rep{replicate_idx}"
    csv_file = f"{prefix}.csv"
    dat_file = f"{prefix}.dat"
    jkl_file = f"{prefix}.jkl"

    
    df_small = pd.DataFrame(data_small)
    df_small.to_csv(csv_file, index=False)
    data_preparation.save_data(df_small, csv_file, dat_file)
    data_preparation.compute_bdeu_scores(dat_file, jkl_file)

    
    parsed_scores = data_io.parse_gobnilp_jkl(jkl_file)
    scores = heuristics.GobnilpScores(parsed_scores)
    n = scores.n
    logging.debug(f"[run_heuristic_on_small_data] JKL parsed -> scores for n={n} nodes")

    algo_func = cpa["greedy"]
    algo_kwargs = {"scores": scores}
    
    tmp_result = algo_func(n-4, **algo_kwargs)

    if isinstance(tmp_result, tuple):
        candidate_parents = tmp_result[0]
    else:
        candidate_parents = tmp_result

    logging.debug(f"[run_heuristic_on_small_data] Candidate parents for each node: {candidate_parents}")

    
    for f in [csv_file, dat_file, jkl_file]:
        if os.path.exists(f):
            os.remove(f)

    return candidate_parents




def find_best_dag_reference(data_ref, replicate_idx=0):
    """
    Use rungobnilp.py on local BDeu scores from data_ref
    to get the single best DAG in .dot form, then parse.
    """
    import pandas as pd
    import data_preparation
    import data_io
    import heuristics
    prefix = f"refdata_rep{replicate_idx}"
    csv_file = f"{prefix}.csv"
    dat_file = f"{prefix}.dat"
    jkl_file = f"{prefix}.jkl"

    df_ref = pd.DataFrame(data_ref)
    df_ref.to_csv(csv_file, index=False)
    data_preparation.save_data(df_ref, csv_file, dat_file)
    data_preparation.compute_bdeu_scores(dat_file, jkl_file)

    logging.debug(f"[find_best_dag_reference] Created data/scores for replicate={replicate_idx}, now running rungobnilp.")
    run_gobnilp_for_best_dag(jkl_file, prefix)

    dot_file = f"{prefix}_gobnilp_solution.dot"
    dag_best = parse_best_dag_dot(dot_file)
    logging.debug(f"[find_best_dag_reference] Parsed DAG from {dot_file}: {dag_best}")

    
    for f in [csv_file, dat_file, jkl_file, dot_file]:
        if os.path.exists(f):
            os.remove(f)

    return dag_best


def run_gobnilp_for_best_dag(jkl_file, prefix):
    """
    Calls rungobnilp.py in 'scores' mode to find a best BN,
    producing a .dot file with the result.
    """
    import subprocess
    cmd = [
        "python3",
        "/home/gulce/Downloads/thesis/pygobnilp-1.0/rungobnilp.py",
        jkl_file,        
        "--scores",
        "--nsols=1",
        "--nopruning",
        f"--output_stem={prefix}_gobnilp_solution",
        "--output_ext=dot",
        "--nooutput_cpdag",
        "--noabbrev",
        "--noplot"
    ]
    logging.info("[run_gobnilp_for_best_dag] Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("[run_gobnilp_for_best_dag] Gobnilp call failed with return code %s", e.returncode)
        raise
import pydot
def parse_best_dag_dot(dot_file):
    """
    Parse adjacency from .dot => returns {node: [parents]}.
    If the file is missing or no edges, returns an empty dict.
    """
    dag_out = {}
    if not os.path.exists(dot_file):
        logging.warning(f"[parse_best_dag_dot] Solution .dot not found: {dot_file}")
        return dag_out

    
    graphs = pydot.graph_from_dot_file(dot_file)
    if not graphs:
        logging.warning(f"[parse_best_dag_dot] No graphs found in {dot_file}.")
        return dag_out
    
    graph = graphs[0]  

    
    for edge in graph.get_edges():
        src = edge.get_source()      
        dst = edge.get_destination() 
        try:
            parent = int(src)
            child = int(dst)
        except ValueError:
            logging.warning(f"[parse_best_dag_dot] Non-integer node labels: {src} -> {dst}. Skipping.")
            continue

        if child not in dag_out:
            dag_out[child] = []
        dag_out[child].append(parent)

    
    all_nodes = set()
    for c, ps in dag_out.items():
        all_nodes.add(c)
        all_nodes.update(ps)

    if all_nodes:
        max_node = max(all_nodes)
        for n in range(max_node + 1):
            if n not in dag_out:
                dag_out[n] = []

    return dag_out




def dag_nodes_covered(dag_ref, candidate_parents):
    """
    Check if DAG 'dag_ref' is "covered" by the candidate parents in 'candidate_parents'.
    For each node i, we require that the true parents of i are a *subset* of
    at least one candidate set. That is, Pa_G(X_i) ⊆ C_i for at least one C_i in candidate_parents[i].

    Parameters
    ----------
    dag_ref : dict
        Dictionary: node -> list_of_parents (the reference DAG).
    candidate_parents : dict
        Dictionary: node -> candidate set(s). Each entry can be:
          * a single tuple/list of parents
          * multiple tuples/lists (e.g., a list of possible parents sets)
          * None if no candidates found

    Returns
    -------
    bool
        True if for every node, there is at least one candidate set that contains all of its true parents.
        False otherwise.
    """
    print("COMPARE  : ",dag_ref,"--",candidate_parents)
    for node, ref_pa_list in dag_ref.items():
        ref_set = set(ref_pa_list)

        
        csets = candidate_parents.get(node, None)
        if csets is None:
            
            return False

        
        if isinstance(csets, tuple):
            csets = [csets]
        elif isinstance(csets, int):
            csets = [[csets]]

        
        
        covered_this_node = False
        for cset in csets:
            cset_as_set = set(cset)
            if ref_set.issubset(cset_as_set):
                covered_this_node = True
                break

        if not covered_this_node:
            
            return False

    
    return True


def arcs_coverage(dag_ref, candidate_parents):
    """
    Compute the fraction of arcs in 'dag_ref' that are 'guessed' by 'candidate_parents'.
    For each arc p->c in dag_ref, we check if 'p' appears in at least one candidate set
    for node 'c'.
    
    Returns a float in [0, 1], or 0.0 if dag_ref has no arcs.
    """
    
    
    arcs = []
    for child, parents in dag_ref.items():
        for p in parents:
            arcs.append((p, child))
    total_arcs = len(arcs)
    if total_arcs == 0:
        return 1.0  

    guessed = 0
    for (p, c) in arcs:
        csets = candidate_parents.get(c, None)
        if csets is None:
            
            continue
        
        
        if isinstance(csets, (int, tuple, set)):
            csets = [csets]

        
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

from sumu.candidates import candidate_parent_algorithm as cpa



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vars", type=int, default=10, help="Number of nodes (variables)")
    parser.add_argument("--max_in_degree", type=int, default=3, help="Max in-degree for the DAG prior")
    parser.add_argument("--num_states", type=int, default=2, help="Number of discrete states for each variable")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet prior hyperparameter")
    parser.add_argument("--M", type=int, default=5, help="Number of replicates.")
    parser.add_argument("--small_data_size", type=int, default=500)
    parser.add_argument("--ref_data_size", type=int, default=10000)
    parser.add_argument("--output_csv", type=str, default="two_phase_nodelevel_coverage.csv")
    args = parser.parse_args()
    logging.debug(f"[main] Starting with arguments: {args}")

    
    param_str = (
        f"nVars={args.n_vars}_maxInDeg={args.max_in_degree}_"
        f"alpha={args.alpha}_smallData={args.small_data_size}_refData={args.ref_data_size}"
    )

    
    coverage_csv = f"coverage_{param_str}.csv"
    means_csv    = f"coverage_means_{param_str}.csv"
    log_file     = f"coverage_log_{param_str}.log"

    coverage_rows = []
    with open(log_file, "w") as lf:
        lf.write(f"
        lf.write(f"

        for r in range(args.M):
            logging.info(f"=== Replicate {r+1}/{args.M} ===")

            
            G_star = sample_dag_fair_prior(args.n_vars, args.max_in_degree)
            cpts_star = sample_cpts_for_dag(G_star, alpha=args.alpha, num_states=args.num_states)
            logging.debug(f"[main] Rep={r}, Sampled DAG (G_star) = {G_star}")

            
            data_small = generate_data_from_dag(G_star, cpts_star, args.small_data_size, args.num_states)
            scores_small = get_sumu_scores_for_small_data(data_small, replicate_idx=r)
            n = scores_small.n

            
            import sumu
            mydata_small = sumu.Data(data_small)

            
            data_ref = generate_data_from_dag(G_star, cpts_star, args.ref_data_size, args.num_states)
            DAG_ref = find_best_dag_reference(data_ref, replicate_idx=r)
            logging.debug(f"[main] Rep={r}, DAG_ref => {DAG_ref}")

            
            from sumu.candidates import candidate_parent_algorithm as cpa
            import heuristics  

            candidate_algos = {
                "opt":            (cpa["opt"],            {"scores": scores_small, "n": n}),
                "top":            (cpa["top"],            {"scores": scores_small, "n": n}),
                "mb":             (cpa["mb"],             {"data": mydata_small, "fill": "random"}),
                "pc":             (cpa["pc"],             {"data": mydata_small, "fill": "random"}),
                "ges":            (cpa["ges"],            {"scores": scores_small, "data": mydata_small, "fill": "top"}),
                "greedy":         (cpa["greedy"],         {"scores": scores_small}),
                "greedy-lite":    (cpa["greedy-lite"],    {"scores": scores_small}),
                "back-forth":     (cpa["back-forth"],     {"scores": scores_small, "data": scores_small.data}),
                "beam":           (heuristics.beam_bdeu,  {"scores": scores_small, "beam_size": 5}),
                "marginal_bdeu_parents": (heuristics.marginal_bdeu_parents, {"scores": scores_small, "n": n}),
                
                "synergy":        (heuristics.synergy_based_parent_selection, {"scores": scores_small}),
                "stability":      (heuristics.stability_bdeu, {"scores": scores_small, "data": mydata_small}),
                "post":           (heuristics.maximize_true_graph_posterior, {"scores": scores_small}),
            }

            for algo_name, (algo_func, algo_kwargs) in candidate_algos.items():
                
                for K in range(1, n):
                    try:
                        tmp_res = algo_func(K, **algo_kwargs)
                        if isinstance(tmp_res, tuple):
                            candidate_pars = tmp_res[0]
                        else:
                            candidate_pars = tmp_res
                    except Exception as e:
                        logging.warning(f"[Rep={r}] algo={algo_name}, K={K} error: {e}")
                        
                        coverage_rows.append((r, algo_name, K, 0, 0))
                        continue

                    cov_star = arcs_coverage(G_star, candidate_pars)
                    cov_ref  = arcs_coverage(DAG_ref, candidate_pars)
                   
                    
                    coverage_rows.append((r, algo_name, K, cov_star, cov_ref))
                    
                    
                    lf.write("\n--------------------------------------------------\n")
                    lf.write(f"Rep={r}, ALGORITHM={algo_name}, K={K}\n\n")
                    lf.write(f"Candidate parents by heuristic:\n{candidate_pars}\n\n")
                    lf.write(f"PRIOR parents (G_star):\n{G_star}\n\n")
                    lf.write(f"GOBNILP parents (DAG_ref):\n{DAG_ref}\n\n")
                    lf.write(f"Coverage with PRIOR = {cov_star}, Coverage with GOBNILP = {cov_ref}\n")
                    lf.write("--------------------------------------------------\n\n")

    
    df_out = pd.DataFrame(
        coverage_rows,
        columns=["replicate", "method", "K", "coverage_prior", "coverage_gobnilp"]
    )
    
    df_out.to_csv(args.output_csv, index=False)
    logging.info(f"Saved raw results to {args.output_csv}")

    
    df_mean = df_out.groupby(["method", "K"], as_index=False)[["coverage_prior", "coverage_gobnilp"]].mean()
    logging.info("\nMean coverage across replicates:\n%s", df_mean)
    
    
    df_mean.to_csv(means_csv, index=False)
    logging.info(f"Saved coverage means to {means_csv}")


if __name__ == "__main__":
    main()
