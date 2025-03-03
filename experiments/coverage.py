# mybnexp/coverage.py

def coverage_fraction(candidate_parents, sampled_dags):
    """
    Compute fraction of sampled_dags that are covered by candidate_parents.

    Coverage criterion:
        For every node v, the DAG's parents for v are a subset of candidate_parents[v].
    """
    if not sampled_dags:
        return 0.0
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

