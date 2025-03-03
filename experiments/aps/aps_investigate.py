


from itertools import chain, combinations

import numpy as np
import data_io
import heuristics as heuristics
import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import _aps
_aps.enable_aps_debug(True)

def subsets(iterable, fromsize, tosize):
    s = list(iterable)
    step = 1 + (fromsize > tosize) * -2
    return chain.from_iterable(
        combinations(s, i) for i in range(fromsize, tosize + step, step)
    )



def opt(K, **kwargs):

    scores = kwargs.get("scores")
    n = kwargs.get("n")

    C = np.array([[v for v in range(n) if v != u] for u in range(n)], dtype=np.int32)
    print("BDEU ")
    print(parsed_scores)
    print("BDEU Candiddate restricted")
    print(scores.all_candidate_restricted_scores(C))
    print("")
    pset_posteriors = _aps.aps(scores.all_candidate_restricted_scores(C),
                          as_dict=True, normalize=True)
    
    print(pset_posteriors)
    C = dict()
    for v in pset_posteriors:
        postsums = dict()
        for candidate_set in subsets(set(pset_posteriors).difference({v}), K, K):
            postsums[candidate_set] = np.logaddexp.reduce([pset_posteriors[v][pset]
                                                           for pset in subsets(candidate_set, 0, K)])
        C[v] = max(postsums, key=lambda candidate_set: postsums[candidate_set])
    return C



def top(K, **kwargs):

    scores = kwargs["scores"]
    assert scores is not None, "scorepath (-s) required for algo == top"

    C = dict()
    for v in range(scores.n):
        top_candidates = sorted([(parent, scores.local(v, np.array([parent])))
                                 for parent in range(scores.n) if parent != v],
                                key=lambda item: item[1], reverse=True)[:K]
        top_candidates = tuple(sorted(c[0] for c in top_candidates))
        C[v] = top_candidates
    return C


parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/asia/asia_scores.jkl')
scores = heuristics.GobnilpScores(parsed_scores)
n = scores.n
print("-----------------------------------",flush=True)
C = np.array([[v for v in range(n) if v != u] for u in range(n)], dtype=np.int32)
print(scores.all_candidate_restricted_scores(C))
for K in range(2,3, 1):


    candidate_parents= opt(
    K,
    scores=scores,    # Must pass in your scoring object
    n=scores.data.n,  # 'opt' also expects 'n' in kwargs
)
    print(candidate_parents)
    
print("----------------------------")
# for K in range(1,8, 1):


#     candidate_parents= top(
#     K,
#     scores=scores,    # Must pass in your scoring object
#     n=scores.data.n,  # 'opt' also expects 'n' in kwargs
# )
#     print(candidate_parents)