# mybnexp/coverage.py
def get_true_parents(model_file):  
    import bnlearn as bn

# Load a predefined Bayesian network structure
    model = bn.import_DAG(model_file)
    adjmat = model["adjmat"]

    # Get the list of node names (assumed to be the same in the DataFrame index and columns)
    nodes = list(adjmat.columns)
    print(adjmat.columns)



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
def coverage_fraction(candidate_parents, sampled_dags):
    """
    Compute fraction of sampled_dags that are covered by candidate_parents.

    Coverage criterion:
        For every node v, the DAG's parents for v are a subset of candidate_parents[v].
    """
    if not sampled_dags:
        return None
    count_covered = 0
    total = len(sampled_dags)

    for dag in sampled_dags:
        covered = True
        for node, parents in dag.items():
            if not parents.issubset(candidate_parents[node]):
                covered = False
                break
        if covered:
            count_covered += 1
    
    return count_covered / total

