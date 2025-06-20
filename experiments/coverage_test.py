
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

        
        csets = candidate_parents.get(node, None)
        if csets is None:
            
            return False

        
        
        if isinstance(csets, tuple):
            csets = [csets]

        
        elif isinstance(csets, int):
            csets = [[csets]]

        
        

        found_match = False
        for cset in csets:
            
            if not isinstance(cset, set):
                cset = set(cset)

            if cset == ref_set:
                found_match = True
                break

        if not found_match:
            return False

    
    return True

dag_nodes_covered({1: [8, 2, 3], 7: [6], 0: [], 2: [], 3: [], 4: [], 5: [], 6: [], 8: []},{1: [8, 2, 3,7], 7: [6], 0: [], 2: [], 3: [], 4: [], 5: [], 6: [], 8: []})
print(dag_nodes_covered)