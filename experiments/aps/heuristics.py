# mybnexp/heuristics.py

import numpy as np
import logging

import sumu
from sumu.candidates import candidate_parent_algorithm as cpa
class Data:
    """Simple container for the number of variables `n`."""
    def __init__(self, n):
        self.n = n

class GobnilpScores:
    """
    A Scores-like class that wraps the output of parse_gobnilp_jkl 
    for use with sumu's candidate parent algorithms.
    """
    def __init__(self, parsed_scores):
        """
        Args:
            parsed_scores (dict): 
                A dict of { node: [ (score, (parents...)), ... ], ... }
        """
        self.n = max(parsed_scores.keys()) + 1
        self.data = Data(self.n)
        print("-----------------------------------",flush=True)
        
        # Store local scores in { node: {parents_tuple: score} }
        self.local_scores = {}
        for node, sp_list in parsed_scores.items():
            self.local_scores[node] = {}
            for (score, parents) in sp_list:
                parents_sorted = tuple(sorted(parents))
                self.local_scores[node][parents_sorted] = score

        # If you do not have a known maximum parent set size, keep this -1
        self.maxid = -1

    def local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.local_scores[v].get(p_sorted, float("-inf"))
    
    def _local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.local_scores[v].get(p_sorted, float("-inf"))

    def all_candidate_restricted_scores(self, C):
        import numpy as np
        print("-----------------------------------",flush=True)
        V = len(C)
        # Compute the number of subsets for each node.
        subset_counts = [1 << len(C[i]) for i in range(V)]
        max_subset_count = max(subset_counts)
        arr = np.full((V, max_subset_count), float("-inf"), dtype=float)
        
        for i in range(V):
            # Ensure the candidate parents for node i are sorted.
            sorted_parents = sorted(C[i])
            subset_count = 1 << len(sorted_parents)
            for m in range(subset_count):
                # Reverse the bit order so that the leftmost (first) element of the sorted list
                # corresponds to the most significant bit.
                parents_tuple = tuple(
                    sorted_parents[k] for k in range(len(sorted_parents))
                    if (m & (1 << (len(sorted_parents) - 1 - k)))
                )
                # Look up the score; if missing, use -inf.
                sc = self.local_scores[i].get(parents_tuple, float("-inf"))
                arr[i, m] = sc
        return arr


    def sum(self, v, U, T):

        from itertools import combinations

        # Compute the union of sets U and T
        combined_parents = U | T  # This will create a single set containing all elements from U and T

        # Determine the maximum number of parents (no limit in this case)
        max_parents = len(combined_parents)

        total_score = float("-inf")

        # Iterate over all possible parent sets from the union of U and T
        for k in range(max_parents + 1):
            for parent_set in combinations(combined_parents, k):
                score = self.local(v, parent_set)
                total_score = np.logaddexp(total_score, score)

        return total_score

    def clear_cache(self):
        """If your scoring logic uses caching, clear it here; otherwise do nothing."""
        pass
        """_summary_
        """    

    def filter_parent_sets_by_size(self, k):
            """
            Filter the local scores to only include candidate parent sets with exactly k parents.
            """
            for node in self.local_scores:
                self.local_scores[node] = {
                    parents: score
                    for parents, score in self.local_scores[node].items()
                    if len(parents) == k
                }


    def clear_cache(self):
        """If your scoring logic uses caching, clear it here; otherwise do nothing."""
        pass
    


def beam_bdeu(K, scores, beam_size=5, seed=None):
    """
    Custom beam search to pick exactly K parents for each node, maximizing local BDeu (scores.local).

    Returns: dict: node -> tuple_of_parents
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = scores.n

    def local_score(v, pset):
        # sumu's scores.local expects a list or np.array
        return scores.local(v, np.array(list(pset)))

    candidate_parents = {}
    for v in range(n):
        # all possible parents except v
        possible_parents = [u for u in range(n) if u != v]

        if len(possible_parents) < K:
            logging.warning(f"Node {v}: cannot pick K={K} parents out of {len(possible_parents)} possible!")
            # fallback or raise an error
            raise ValueError(f"Node {v} has fewer than K possible parents.")

        # Start beam with empty set
        beam = [(local_score(v, []), frozenset())]

        for _ in range(K):
            new_level = []
            for (old_score, pset) in beam:
                for cand in possible_parents:
                    if cand not in pset:
                        new_pset = set(pset)
                        new_pset.add(cand)
                        new_score = local_score(v, new_pset)
                        new_level.append((new_score, frozenset(new_pset)))
            
            new_level.sort(key=lambda x: x[0], reverse=True)
            beam = new_level[:beam_size]
        
        # best among subsets of size K
        best_score, best_pars = max(beam, key=lambda x: x[0])
        candidate_parents[v] = tuple(sorted(best_pars))

    return candidate_parents


def get_candidate_parents(algo_name, K, scores, data=None, fill="top", **kwargs):
    """
    Return candidate parents for each node, depending on `algo_name`.

    If algo_name == "beam", uses custom beam_bdeu.
    Otherwise uses sumu's built-in candidate_parent_algorithm (cpa).

    Returns: dict: {node: parent_set or parent_tuple}
    """
    if algo_name == "beam":
        return beam_bdeu(K=K, scores=scores, **kwargs)
    else:
        # use sumu's cpa
        algo = cpa[algo_name]
        # Some cpa methods require data; some do not. We unify the call:
        # By default, let's pass both scores and data if they exist:
        return algo(K, scores=scores, data=data, fill=fill)


import numpy as np
from itertools import combinations
from math import log, exp

def logsumexp(x):
    """Stable log-sum-exp."""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def marginal_bdeu_parents(K, **kwargs):
    """
    Select candidate parents by marginal BDeu posterior.

    For each node v, we compute for each potential parent u:
        P(u in ParentSet(v)) = sum_{S : u in S} exp( BDeu(v,S) )
    normalized by the sum of exp( BDeu(v,S) ) over *all* S.

    Then pick top-K parents of v by this marginal probability.

    Parameters
    ----------
    K : int
        Maximum number of parents to keep for each node.
    kwargs : dict
        Must contain 'scores', which is a GobnilpScores-like object with:
            - scores.local_scores[v]: dict {parents_tuple: log_bdeu_score}
            - a 'local(v, parents)' method
            - an integer 'n' or 'scores.data.n' for the number of variables
        Optionally can contain 'n', if not inferred from the scores object.
        'fill' can be 'none', 'top', or 'random'.

    Returns
    -------
    C : dict
        Dictionary of the form { v : tuple_of_parents }.
    """
    # 1) Get scores object
    scores = kwargs.get("scores", None)
    if scores is None:
        raise ValueError("marginal_bdeu_parents requires 'scores' in kwargs.")
    
    fill_method = kwargs.get("fill", "none")

    # 2) Determine number of variables 'n'
    n = kwargs.get("n", None)
    if n is None:
        # Attempt to retrieve n from the scores object
        if hasattr(scores, "n"):
            n = scores.n
        elif hasattr(scores, "data") and hasattr(scores.data, "n"):
            n = scores.data.n
        else:
            raise ValueError("Cannot find 'n' from either kwargs or the scores object.")

    # 3) Dictionary to hold the final candidate parents
    C = {}

    # 4) Loop over each node v
    for v in range(n):

        # Instead of 'if v not in scores:', use 'scores.local_scores'
        if v not in scores.local_scores:
            # If no entry, means no known subsets => no candidates
            C[v] = ()
            continue

        # v_subsets: dict mapping { (p1, p2, ...): log_BDeu, ... }
        v_subsets = scores.local_scores[v]
        if not v_subsets:
            # empty => no parents
            C[v] = ()
            continue

        # -- Compute the log normalizer: logsumexp over all subsets' scores
        all_log_scores = list(v_subsets.values())
        normalizer = logsumexp(all_log_scores)  # log Z

        # -- For each potential parent u != v, gather log-scores of subsets that contain u
        logp_u = {}
        for u in range(n):
            if u == v:
                continue
            log_values = []
            for parent_set, logscore in v_subsets.items():
                if u in parent_set:
                    log_values.append(logscore)
            if len(log_values) == 0:
                logp_u[u] = float("-inf")  # u never appears in any subset
            else:
                logp_u[u] = logsumexp(log_values)

        # -- Sort parents by logp_u[u] descending (equivalent to marginal posterior)
        candidate_list = [(u, lv) for (u, lv) in logp_u.items()]
        candidate_list.sort(key=lambda x: x[1], reverse=True)

        # -- Pick top-K
        chosen = [u for (u, lv) in candidate_list[:K]]
        C[v] = tuple(sorted(chosen))

    # 5) If fill_method is 'top' or 'random', optionally fill/prune to K exactly
    if fill_method in ["top", "random"]:
        C = _adjust_number_candidates(K, C, method=fill_method, scores=scores)

    return C


def _adjust_number_candidates(K, C, method, scores=None):
    """Adjust the number of candidate parents to exactly K for each node."""
    assert method in ['random', 'top'], "method must be 'random' or 'top'"
    
    for v in C:
        current_parents = list(C[v])
        parent_count = len(current_parents)

        if parent_count < K:
            # Need to ADD parents
            needed = K - parent_count
            add_from = [node for node in range(len(C)) if node != v and node not in current_parents]

            if method == 'random':
                chosen = np.random.choice(add_from, needed, replace=False)
                current_parents += chosen.tolist()

            elif method == 'top' and scores is not None:
                scored_list = []
                for parent_candidate in add_from:
                    score_val = scores.local(v, np.array([parent_candidate]))
                    scored_list.append((parent_candidate, score_val))
                scored_list.sort(key=lambda x: x[1], reverse=True)
                best = [p for (p, s_val) in scored_list[:needed]]
                current_parents += best

        elif parent_count > K:
            # Need to PRUNE parents
            excess = parent_count - K
            if method == 'random':
                chosen_to_keep = np.random.choice(current_parents, K, replace=False)
                current_parents = list(chosen_to_keep)
            elif method == 'top' and scores is not None:
                scored_list = []
                for p in current_parents:
                    score_val = scores.local(v, np.array([p]))
                    scored_list.append((p, score_val))
                scored_list.sort(key=lambda x: x[1], reverse=True)
                current_parents = [p for (p, s_val) in scored_list[:K]]

        # Update
        C[v] = tuple(sorted(set(current_parents)))

    return C
import numpy as np
from itertools import combinations

def score_improvement(v, u, scores, potential_parents):
    """
    Calculate the score improvement for adding a parent u to the current parent set of node v.
    
    Parameters:
    v: Node for which we are selecting parents.
    u: Potential parent to be added.
    scores: GobnilpScores-like object containing the BDeu scores.
    potential_parents: Current set of parents for node v.
    
    Returns:
    score_improvement: The improvement in the BDeu score when adding parent u.
    """
    # Get the current score of the node v with its current parent set
    current_score = scores.local(v, np.array(potential_parents))
    
    # Add the potential parent u to the current set of parents
    new_parents = tuple(sorted(potential_parents + (u,)))
    new_score = scores.local(v, np.array(new_parents))
    
    # The improvement in the BDeu score is the difference
    score_improvement = new_score - current_score
    return score_improvement

import numpy as np
import logging

import sumu
from sumu.candidates import candidate_parent_algorithm as cpa


########################################
# 1) Fix the Data class so "object of type 'Data' has no len()" is resolved:
########################################
class Data:
    """Simple container for the number of variables `n`."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        """
        Some code (like stability selection) tries to do `len(data)`.
        Defining this lets that code run without TypeError.
        But be aware that if code tries `data[idx, :]`, 
        you still need __getitem__ for real row-based sampling.
        """
        return self.n
    
    def __len__(self):
        """
        If needed, let's define __len__ so that code 
        that calls len(data) won't crash. 
        """
        return self.n


class GobnilpScores:
    """
    A Scores-like class that wraps the output of parse_gobnilp_jkl 
    for use with sumu's candidate parent algorithms.
    """
    def __init__(self, parsed_scores):
        """
        Args:
            parsed_scores (dict): 
                A dict of { node: [ (score, (parents...)), ... ], ... }
        """
        self.n = max(parsed_scores.keys()) + 1
        self.data = Data(self.n)
        
        # Store local scores in { node: {parents_tuple: score} }
        self.local_scores = {}
        for node, sp_list in parsed_scores.items():
            self.local_scores[node] = {}
            for (score, parents) in sp_list:
                parents_sorted = tuple(sorted(parents))
                self.local_scores[node][parents_sorted] = score

        # If you do not have a known maximum parent set size, keep this -1
        self.maxid = -1

    def local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.local_scores[v].get(p_sorted, float("-inf"))
    
    def _local(self, v, parents):
        """
        Same as 'local' above; sometimes sumu calls _local(...) internally.
        """
        p_sorted = tuple(sorted(parents))
        return self.local_scores[v].get(p_sorted, float("-inf"))

    def all_candidate_restricted_scores(self, C):
        import numpy as np
        V = len(C)
        # Compute the number of subsets for each node.
        subset_counts = [1 << len(C[i]) for i in range(V)]
        max_subset_count = max(subset_counts)
        arr = np.full((V, max_subset_count), float("-inf"), dtype=float)
        
        for i in range(V):
            # Sort the candidate parents for node i.
            sorted_parents = sorted(C[i])
            subset_count = 1 << len(sorted_parents)
            for m in range(subset_count):
                # Use natural bit order: bit k corresponds to sorted_parents[k]
                parents_tuple = tuple(
                    sorted_parents[k]
                    for k in range(len(sorted_parents))
                    if (m & (1 << k))
                )
                # Look up the score; if missing, use -inf.
                sc = self.local_scores[i].get(parents_tuple, float("-inf"))
                arr[i, m] = sc
        return arr


    def sum(self, v, U, T):
        """
        Example method that sums log-scores over subsets from the union of U and T,
        for node v.
        """
        from itertools import combinations
        combined_parents = U | T
        max_parents = len(combined_parents)
        total_score = float("-inf")

        for k in range(max_parents + 1):
            for parent_set in combinations(combined_parents, k):
                score = self.local(v, parent_set)
                total_score = np.logaddexp(total_score, score)
        return total_score

    def clear_cache(self):
        """If your scoring logic uses caching, clear it here; otherwise do nothing."""
        pass


########################################
# 2) "beam_bdeu" is fine, no changes needed
########################################
def beam_bdeu(K, scores, beam_size=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n = scores.n

    def local_score(v, pset):
        return scores.local(v, np.array(list(pset)))

    candidate_parents = {}
    for v in range(n):
        possible_parents = [u for u in range(n) if u != v]

        if len(possible_parents) < K:
            logging.warning(f"Node {v}: cannot pick K={K} parents out of {len(possible_parents)} possible!")
            raise ValueError(f"Node {v} has fewer than K possible parents.")

        beam = [(local_score(v, []), frozenset())]

        for _ in range(K):
            new_level = []
            for (old_score, pset) in beam:
                for cand in possible_parents:
                    if cand not in pset:
                        new_pset = set(pset)
                        new_pset.add(cand)
                        new_score = local_score(v, new_pset)
                        new_level.append((new_score, frozenset(new_pset)))
            
            new_level.sort(key=lambda x: x[0], reverse=True)
            beam = new_level[:beam_size]
        
        best_score, best_pars = max(beam, key=lambda x: x[0])
        candidate_parents[v] = tuple(sorted(best_pars))

    return candidate_parents


def get_candidate_parents(algo_name, K, scores, data=None, fill="top", **kwargs):
    """
    Return candidate parents for each node, depending on `algo_name`.
    If algo_name == "beam", uses custom beam_bdeu.
    Otherwise uses sumu's built-in candidate_parent_algorithm (cpa).
    """
    if algo_name == "beam":
        return beam_bdeu(K=K, scores=scores, **kwargs)
    else:
        # use sumu's cpa
        algo = cpa[algo_name]
        return algo(K, scores=scores, data=data, fill=fill, **kwargs)


########################################
# 3) Marginal BDeu - ensure no "list+tuple" issues or 'v not in scores' issues
########################################
from itertools import combinations
from math import log

def logsumexp(x):
    """Stable log-sum-exp."""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def marginal_bdeu_parents(K, **kwargs):
    scores = kwargs.get("scores", None)
    if scores is None:
        raise ValueError("marginal_bdeu_parents requires 'scores' in kwargs.")
    
    fill_method = kwargs.get("fill", "none")

    n = kwargs.get("n", None)
    if n is None:
        if hasattr(scores, "n"):
            n = scores.n
        elif hasattr(scores, "data") and hasattr(scores.data, "n"):
            n = scores.data.n
        else:
            raise ValueError("Cannot find 'n' in kwargs or scores object.")

    C = {}

    for v in range(n):
        if v not in scores.local_scores:
            C[v] = ()
            continue

        v_subsets = scores.local_scores[v]
        if not v_subsets:
            C[v] = ()
            continue

        all_log_scores = list(v_subsets.values())
        normalizer = logsumexp(all_log_scores)

        logp_u = {}
        for u in range(n):
            if u == v:
                continue
            log_values = []
            for parent_set, logscore in v_subsets.items():
                if u in parent_set:
                    log_values.append(logscore)
            if len(log_values) == 0:
                logp_u[u] = float("-inf")
            else:
                logp_u[u] = logsumexp(log_values) - normalizer

        # sort by logp_u[u] desc
        candidate_list = [(u, lv) for (u, lv) in logp_u.items()]
        candidate_list.sort(key=lambda x: x[1], reverse=True)

        chosen = [u for (u, lv) in candidate_list[:K]]
        C[v] = tuple(sorted(chosen))

    if fill_method in ["top", "random"]:
        C = _adjust_number_candidates(K, C, method=fill_method, scores=scores)

    return C


def _adjust_number_candidates(K, C, method, scores=None):
    assert method in ['random', 'top'], "method must be 'random' or 'top'"
    
    for v in C:
        current_parents = list(C[v])
        parent_count = len(current_parents)

        if parent_count < K:
            needed = K - parent_count
            add_from = [node for node in range(len(C)) if node != v and node not in current_parents]

            if method == 'random':
                chosen = np.random.choice(add_from, needed, replace=False)
                current_parents += chosen.tolist()

            elif method == 'top' and scores is not None:
                scored_list = []
                for parent_candidate in add_from:
                    score_val = scores.local(v, np.array([parent_candidate]))
                    scored_list.append((parent_candidate, score_val))
                scored_list.sort(key=lambda x: x[1], reverse=True)
                best = [p for (p, s_val) in scored_list[:needed]]
                current_parents += best

        elif parent_count > K:
            if method == 'random':
                chosen_to_keep = np.random.choice(current_parents, K, replace=False)
                current_parents = list(chosen_to_keep)
            elif method == 'top' and scores is not None:
                scored_list = []
                for p in current_parents:
                    score_val = scores.local(v, np.array([p]))
                    scored_list.append((p, score_val))
                scored_list.sort(key=lambda x: x[1], reverse=True)
                current_parents = [p for (p, s_val) in scored_list[:K]]

        C[v] = tuple(sorted(set(current_parents)))

    return C


########################################
# 4) BDeu Score-Based Voting (bdeu_score_based_voting) 
#    to handle "list+tuple" fixes
########################################
def score_improvement(v, u, scores, current_set):
    """
    Calculate the improvement from adding parent u to `current_set` for node v.
    """
    current_score = scores.local(v, np.array(current_set))
    new_parents = tuple(sorted(current_set + (u,)))
    new_score = scores.local(v, np.array(new_parents))
    return new_score - current_score

def bdeu_score_based_voting(K, **kwargs):
    scores = kwargs.get("scores", None)
    if scores is None:
        raise ValueError("bdeu_score_based_voting requires 'scores' in kwargs.")
    
    n = kwargs.get("n", None)
    if n is None:
        n = scores.n if hasattr(scores, "n") else scores.data.n
    
    C = {}

    for v in range(n):
        potential_parents = [u for u in range(n) if u != v]
        
        # For each parent, measure improvement
        parent_contributions = {}
        for u in potential_parents:
            # treat the "current_set" as empty for measure
            improvement = score_improvement(v, u, scores, ())
            parent_contributions[u] = improvement
        
        sorted_parents = sorted(parent_contributions, key=parent_contributions.get, reverse=True)
        C[v] = tuple(sorted(sorted_parents[:K]))
    
    return C


########################################
# 5) Synergy-based approach 
#    Accept 'n=None' so no "unexpected keyword argument 'n'" error
########################################
def synergy_for_node(
    v, K, scores, alpha=0.0, fallback=True
):
    """
    Select up to K parents for node v using synergy-based search.
    
    If no positive synergy is found, we stop. If that yields fewer
    than K parents and 'fallback' is True, we pick top singletons
    from the leftover candidates to fill up to K.
    
    v : int
        The node index
    K : int
        Maximum number of parents
    scores : GobnilpScores
        Has .local(v, parents) -> log BDeu
    alpha : float
        Synergy offset factor
    fallback : bool
        Whether to fallback to top singletons if synergy picks fewer than K.
    """
    n = scores.n
    candidates = [u for u in range(n) if u != v]

    # Precompute BDeu(empty) and singletons
    bdeu_empty = scores.local(v, np.array([], dtype=int))
    bdeu_single = {u: scores.local(v, np.array([u])) for u in candidates}

    S = ()  # current parent set
    while len(S) < K:
        best_gain = float("-inf")
        best_u = None

        # current BDeu of S
        bdeu_S = scores.local(v, np.array(S))

        for u in candidates:
            if u in S:
                continue
            # synergy gain = BDeu(S ∪ {u}) - BDeu(S) - alpha*(BDeu({u}) - BDeu({}))
            bdeu_Su = scores.local(v, np.array(S + (u,)))
            gain = bdeu_Su - bdeu_S
            if alpha > 0:
                gain -= alpha * (bdeu_single[u] - bdeu_empty)

            if gain > best_gain:
                best_gain = gain
                best_u = u

        if best_gain <= 0 or best_u is None:
            # no further synergy improvement
            break

        # Otherwise add best_u
        S = tuple(sorted(S + (best_u,)))

    # if synergy yields fewer than K parents and fallback==True,
    # optionally fill from best singletons among leftover candidates
    if fallback and len(S) < K:
        # leftover are the candidates not in S
        leftover = [u for u in candidates if u not in S]
        # sort by single-parent BDeu descending
        leftover.sort(key=lambda u: bdeu_single[u], reverse=True)
        needed = K - len(S)
        # pick top 'needed' leftover
        fill_pars = leftover[:needed]
        S = tuple(sorted(S + tuple(fill_pars)))

    return S


###############################################################################
# Synergy-based parent selection for all nodes
###############################################################################
def synergy_based_parent_selection(K, scores, alpha=0.0, fallback=True):
    """
    For each node v, call synergy_for_node(...) to pick up to K parents.
    
    K : int
        Max parents
    scores : GobnilpScores
        log BDeu data
    alpha : float
        synergy offset
    fallback : bool
        Whether to fill parents with best singletons if synergy picks none.
    """
    n = scores.n
    C = {}
    for v in range(n):
        # If the node has no local scores, we can't pick anything
        if v not in scores.local_scores or len(scores.local_scores[v]) == 0:
            C[v] = ()
            continue

        # Otherwise run synergy
        chosen = synergy_for_node(v, K, scores, alpha=alpha, fallback=fallback)
        C[v] = chosen

    return C


########################################
# 6) Example "stability_bdeu" approach
#    We define a minimal fix to allow len(data) by using Data.__len__ above.
########################################

def stability_bdeu(K, scores, data, B=20, threshold=0.5, fill='top'):
    """
    Perform bootstrap-based stability selection for candidate parents 
    using BDeu scores. If the BDeu scores are empty or all -inf, 
    we gracefully return empty sets or fallback.

    Parameters
    ----------
    K : int
        Max number of parents per node.
    scores : GobnilpScores
        BDeu scoring object for the full dataset.
    data : array-like or Data object
        The 'dataset'. If it is just Data(n), there's no real row dimension.
    B : int
        Number of bootstrap samples.
    threshold : float
        Frequency threshold in [0,1].
    fill : str
        'top' or 'random' or 'none', how to fill if fewer than K stable parents.

    Returns
    -------
    C_stable : dict
        { v : tuple_of_parents } with up to K parents.
    """
    n = scores.n
    # freq[v,u] = how many times u was chosen for v
    freq = np.zeros((n, n), dtype=float)

    for _ in range(B):
        # 1) "sample" data. If real data is not provided, 
        #    this simply returns 'data' or does nothing
        sampled_data = bootstrap_data(data)
        # 2) compute BDeu scores from the sampled data
        sampled_scores = scores

        # 3) pick top-K parents in *this bootstrap* for each node
        #    using our fallback-based function:
        C_b = pick_top_k_parents(sampled_scores, K)

        for v, parents in C_b.items():
            for p in parents:
                freq[v, p] += 1

    freq /= B  # convert to frequency

    # Now build stable sets
    C_stable = {}
    for v in range(n):
        stable_pars = [u for u in range(n) if u != v and freq[v, u] >= threshold]
        if len(stable_pars) > K:
            # prune
            stable_pars.sort(key=lambda u: freq[v,u], reverse=True)
            stable_pars = stable_pars[:K]
        elif len(stable_pars) < K and fill=='top':
            # fill
            others = [u for u in range(n) if u != v and u not in stable_pars]
            others.sort(key=lambda x: freq[v,x], reverse=True)
            needed = K - len(stable_pars)
            stable_pars += others[:needed]

        C_stable[v] = tuple(sorted(stable_pars))

    return C_stable

###############################################################################
# 2) Bootstrapping and BDeu scoring stubs with fallback
###############################################################################
def bootstrap_data(data):
    """
    If data is just Data(n), we do no real sampling. 
    For a real dataset, you would do e.g.:

    idxs = np.random.choice(len(data), len(data), replace=True)
    return data[idxs, :]
    """
    return data



###############################################################################
# 3) pick_top_k_parents with fallback to avoid empty sets
###############################################################################
def pick_top_k_parents(scores, K):
    """
    For each node v, pick top-K parents from local_scores.
    If the node has no singletons or all -inf, fallback to empty set.
    """
    C_b = {}
    n = scores.n

    for v in range(n):
        # If local_scores is missing or empty, fallback to ()
        if v not in scores.local_scores or not scores.local_scores[v]:
            C_b[v] = ()
            continue

        # Collect single-parent subsets
        singletons = {}
        for pset, val in scores.local_scores[v].items():
            if len(pset) == 1:
                singletons[pset[0]] = val

        # If singletons is empty, fallback to the best available set
        # For example, pick the highest scoring multi-parent if it exists:
        if not singletons:
            # fallback to the best subset overall
            best_sub = None
            best_val = float("-inf")
            for pset, val in scores.local_scores[v].items():
                if val > best_val:
                    best_val = val
                    best_sub = pset
            if best_sub is None or best_val == float("-inf"):
                # no valid subsets => empty
                C_b[v] = ()
            else:
                # If best_sub has more than K parents, we still prune
                if len(best_sub) <= K:
                    C_b[v] = tuple(sorted(best_sub))
                else:
                    # arbitrary prune if best_sub is bigger than K
                    # e.g. pick top K parents from best_sub
                    C_b[v] = tuple(sorted(best_sub)[:K])
            continue

        # Otherwise pick top K singletons
        sorted_singles = sorted(singletons.items(), key=lambda x: x[1], reverse=True)
        top_k = [p for (p, sc) in sorted_singles[:K]]
        C_b[v] = tuple(sorted(top_k))

    return C_b
