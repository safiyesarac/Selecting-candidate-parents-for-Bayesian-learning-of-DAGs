
def dag_nodes_covered(dag_ref, candidate_parents):
    """
    node-by-node coverage check:
    For each node i in dag_ref, let ref_par = set(dag_ref[i]).
    We see if candidate_parents[i] has a set that equals ref_par.
    If yes => node i is covered, else not.
    coverage => True only if every node i is covered.
    """
    for node, ref_pa_list in dag_ref.items():
        ref_set = set(ref_pa_list)

        # Grab what the heuristic returned for this node
        csets = candidate_parents.get(node, None)
        if csets is None:
            # No candidate parents at all
            return False

        # If the heuristic returns a *single* tuple (e.g. (3,4,5)),
        # we make it a list of one element => [(3,4,5)]
        if isinstance(csets, tuple):
            csets = [csets]

        # If it might be a single int instead (e.g. 4), wrap that as well
        elif isinstance(csets, int):
            csets = [[csets]]

        # Now csets should be a list of "parent sets."
        # (e.g. [ (3,4,5) ] or [ [4,5], [6,7] ], etc.)

        found_match = False
        for cset in csets:
            # cset might be a tuple or list; convert to a set of integers
            if not isinstance(cset, set):
                cset = set(cset)

            if cset == ref_set:
                found_match = True
                break

        if not found_match:
            return False

    # If we never returned False, all nodes had a matching parent set
    return True

dag_nodes_covered({1: [8, 2, 3], 7: [6], 0: [], 2: [], 3: [], 4: [], 5: [], 6: [], 8: []},{1: [8, 2, 3,7], 7: [6], 0: [], 2: [], 3: [], 4: [], 5: [], 6: [], 8: []})
print(dag_nodes_covered)