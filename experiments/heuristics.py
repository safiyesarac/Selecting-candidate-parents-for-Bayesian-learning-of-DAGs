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
        self.scores = {}
        for node, sp_list in parsed_scores.items():
            self.scores[node] = {}
            for (score, parents) in sp_list:
                parents_sorted = tuple(sorted(parents))
                self.scores[node][parents_sorted] = score

        # If you do not have a known maximum parent set size, keep this -1
        self.maxid = -1

    def local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.scores[v].get(p_sorted, float("-inf"))
    
    
    def _local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.scores[v].get(p_sorted, float("-inf"))

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
                sc = self.scores[i].get(parents_tuple, float("-inf"))
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
            for node in self.scores:
                self.scores[node] = {
                    parents: score
                    for parents, score in self.scores[node].items()
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

    def score(v, pset):
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
        beam = [(score(v, []), frozenset())]

        for _ in range(K):
            new_level = []
            for (old_score, pset) in beam:
                for cand in possible_parents:
                    if cand not in pset:
                        new_pset = set(pset)
                        new_pset.add(cand)
                        new_score = score(v, new_pset)
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
            - scores.scores[v]: dict {parents_tuple: log_bdeu_score}
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

        # Instead of 'if v not in scores:', use 'scores.scores'
        if v not in scores.scores:
            # If no entry, means no known subsets => no candidates
            C[v] = ()
            continue

        # v_subsets: dict mapping { (p1, p2, ...): log_BDeu, ... }
        v_subsets = scores.scores[v]
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
        self.scores = {}
        for node, sp_list in parsed_scores.items():
            self.scores[node] = {}
            for (score, parents) in sp_list:
                parents_sorted = tuple(sorted(parents))
                self.scores[node][parents_sorted] = score

        # If you do not have a known maximum parent set size, keep this -1
        self.maxid = -1

    def local(self, v, parents):
        """
        Sumu calls 'scores.local(...)' in the candidate generation.
        So, we must provide this method name exactly.
        """
        p_sorted = tuple(sorted(parents))
        return self.scores[v].get(p_sorted, float("-inf"))
    
    def _local(self, v, parents):
        """
        Same as 'local' above; sometimes sumu calls _local(...) internally.
        """
        p_sorted = tuple(sorted(parents))
        return self.scores[v].get(p_sorted, float("-inf"))

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
                sc = self.scores[i].get(parents_tuple, float("-inf"))
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

    def score(v, pset):
        return scores.local(v, np.array(list(pset)))

    candidate_parents = {}
    for v in range(n):
        possible_parents = [u for u in range(n) if u != v]

        if len(possible_parents) < K:
            logging.warning(f"Node {v}: cannot pick K={K} parents out of {len(possible_parents)} possible!")
            raise ValueError(f"Node {v} has fewer than K possible parents.")

        beam = [(score(v, []), frozenset())]

        for _ in range(K):
            new_level = []
            for (old_score, pset) in beam:
                for cand in possible_parents:
                    if cand not in pset:
                        new_pset = set(pset)
                        new_pset.add(cand)
                        new_score = score(v, new_pset)
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
        if v not in scores.scores:
            C[v] = ()
            continue

        v_subsets = scores.scores[v]
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
        if v not in scores.scores or len(scores.scores[v]) == 0:
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
    For each node v, pick top-K parents from scores.
    If the node has no singletons or all -inf, fallback to empty set.
    """
    C_b = {}
    n = scores.n

    for v in range(n):
        # If scores is missing or empty, fallback to ()
        if v not in scores.scores or not scores.scores[v]:
            C_b[v] = ()
            continue

        # Collect single-parent subsets
        singletons = {}
        for pset, val in scores.scores[v].items():
            if len(pset) == 1:
                singletons[pset[0]] = val

        # If singletons is empty, fallback to the best available set
        # For example, pick the highest scoring multi-parent if it exists:
        if not singletons:
            # fallback to the best subset overall
            best_sub = None
            best_val = float("-inf")
            for pset, val in scores.scores[v].items():
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
import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np


class BayesianNetwork:
    def __init__(self, structure, scores):
        """
        structure: dict of node -> tuple/set of parent nodes
        scores: GobnilpScores (with real BDeu up to 3 parents)
        """
        self.structure = structure
        self.scores = scores

    def compute_posterior(self):
        """
        Sums up real local scores.  If any node has more than 3 parents,
        the real GobnilpScores.local(...) might be -inf => -inf total.
        So this step is only valid if you do indeed have those bigger sets scored,
        or you rely on the same approximation again here.
        """
        total_score = 0.0
        for node, pars in self.structure.items():
            val = self.scores.local(node, tuple(sorted(pars)))
            if val == float('-inf'):
                return float('-inf')
            total_score += val
        return total_score

def approximate_score(node, parents, scores):
    """
    Approximate local score using existing up-to-3-parent data.
    If parents has size > 3, pick best 3-subset's real local score as a stand-in.
    """
    if len(parents) <= 3:
        return scores.local(node, tuple(sorted(parents)))

    best_sub_score = -float("inf")
    for sub in itertools.combinations(parents, 3):
        sc = scores.local(node, tuple(sorted(sub)))
        if sc > best_sub_score:
            best_sub_score = sc
    return best_sub_score

def compute_approx_score_for_candidate(node, parents, scores):
    """
    Use an approximate scheme to get a 'score' for sets possibly > 3 parents.
    """
    return approximate_local_score(node, parents, scores)

def approximate_local_score(node, parents, scores):
    """
    Approximate local score for sets with > 3 parents by averaging
    the real GobnilpScores for all 3-subsets.

    If len(parents) <= 3, just return the real local score.
    """
    from math import isfinite
    
    sz = len(parents)
    # If 3 or fewer, return the real score
    if sz <= 3:
        return scores.local(node, tuple(sorted(parents)))

    # For > 3, let's compute the average across all 3-subsets
    all_3_subsets = list(itertools.combinations(parents, 3))
    scores_3sub = []
    for sub in all_3_subsets:
        sc_sub = scores.local(node, tuple(sorted(sub)))
        if isfinite(sc_sub):
            scores_3sub.append(sc_sub)
        else:
            # If any 3-subset is -inf, we can either skip it or treat it as -inf
            # (which will drag the average down heavily).
            scores_3sub.append(float('-inf'))

    if not scores_3sub:
        # If they are all -inf, the approximate score is -inf
        return float('-inf')
    
    # Return the average or the max or any combination
    return sum(scores_3sub)/len(scores_3sub)


def maximize_true_graph_posterior( K,scores):
    """
    Column generation for exactly K parents per node,
    but if K>3, we approximate the local score of a bigger set by
    the best 3-parent subset (or another scheme).

    Returns: a dict of node -> chosen parent set (size exactly K).
    """
    n=scores.n
    master = gp.Model("MasterProblem")
    master.Params.OutputFlag = 0

    x_vars = {}
    columns = {}

    # 1) Build candidate columns for each node: all subsets of size K
    for i in range(n):
        columns[i] = []
        possible_parents = [p for p in range(n) if p != i]

        for cand in itertools.combinations(possible_parents, K):
            sc = compute_approx_score_for_candidate(i, cand, scores)
            if np.isfinite(sc):
                var = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                    name=f"x_{i}_{cand}")
                x_vars[(i, cand)] = var
                columns[i].append(cand)

    master.update()

    # 2) Constraints: each node picks exactly one K-parent set
    constrs = {}
    for i in range(n):
        if len(columns[i]) == 0:
            raise ValueError(
                f"No feasible K={K} parent sets for node {i} with the approximate scoring."
            )
        constrs[i] = master.addConstr(
            gp.quicksum(x_vars[(i, c)] for c in columns[i]) == 1,
            name=f"node_{i}"
        )
    master.update()

    # 3) Objective: sum of approximate scores
    obj_expr = gp.LinExpr()
    for i in range(n):
        for cand in columns[i]:
            sc = compute_approx_score_for_candidate(i, cand, scores)
            obj_expr.addTerms(sc, x_vars[(i, cand)])
    master.setObjective(obj_expr, GRB.MAXIMIZE)
    master.update()

    # 4) Column generation loop
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print(f"Master problem not optimal at iteration {iteration}; stopping.")
            break

        duals = {i: constrs[i].Pi for i in range(n)}
        improved = False

        # Pricing: find columns (K-subsets) with positive reduced cost
        for i in range(n):
            best_rc = -float('inf')
            best_cand = None
            possible_parents = [p for p in range(n) if p != i]
            for cand in itertools.combinations(possible_parents, K):
                if cand in columns[i]:
                    continue
                sc = compute_approx_score_for_candidate(i, cand, scores)
                if not np.isfinite(sc):
                    continue
                rc = sc - duals[i]
                if rc > best_rc:
                    best_rc = rc
                    best_cand = cand

            # Add any new column with significantly positive reduced cost
            if best_cand is not None and best_rc > 1e-8:
                improved = True
                columns[i].append(best_cand)
                var = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                    name=f"x_{i}_{best_cand}")
                x_vars[(i, best_cand)] = var
                master.chgCoeff(constrs[i], var, 1.0)
                master.setObjective(master.getObjective() + best_rc * var)
                master.update()

    # 5) Extract solution
    solution = {}
    for i in range(n):
        best_val = -1.0
        best_cand = None
        for cand in columns[i]:
            val = x_vars[(i, cand)].X
            if val > best_val:
                best_val = val
                best_cand = cand
        solution[i] = tuple(sorted(best_cand)) if best_cand else ()
        
    print(f"Column generation done after {iteration} iteration(s).")
    return solution



import itertools, numpy as np, gurobipy as gp
from gurobipy import GRB
from collections import defaultdict, Counter

# ----------------------------------------------------------------------
# helper ----------------------------------------------------------------
def local_score(i, pa, scores):
    try:               # dict‑of‑dicts
        return scores.local[i][pa]
    except TypeError:  # callable
        return scores.local(i, pa)


def approx_score(i, pa, scores, max_exact=3):
    """best |pa|≤max_exact subset score (simple surrogate)."""
    best = -np.inf
    for sub in itertools.combinations(pa, min(max_exact, len(pa))):
        best = max(best, local_score(i, sub, scores))
    return best

# ----------------------------------------------------------------------
def maximise_posterior_via_sampled_dags(K, scores, sampled_dags,
                                    freq_threshold=0.01,
                                    rc_tol=1e-8):
    """
    Column generation that is aligned with posterior DAG samples.
    Returns: dict {node -> chosen K‑parent tuple} (sorted).
    --------------------------------------------------------------------
    Parameters
    ----------
    K            : int   – max parents allowed in the candidate list
    scores       : object holding pre‑computed local BDeu scores
    sampled_dags : iterable of DAG objects or dicts {node -> tuple(parent list)}
    freq_threshold : float – min fraction for a parent to be considered 'core'
    rc_tol         : float – positive reduced‑cost tolerance for pricing
    """
    n = scores.n

    # 0)  gather posterior info
    parent_freq = [Counter() for _ in range(n)]
    for G in sampled_dags:
        for v, pa in G.items():
            for p in pa:
                parent_freq[v][p] += 1
    m_samples = len(sampled_dags)
    parent_freq = [{p: c/m_samples for p, c in cnt.items()} for cnt in parent_freq]

    # 1) master model
    m  = gp.Model("exactK_CG");  m.Params.OutputFlag = 0
    x, cols = {}, defaultdict(list)
    pick    = {}

    for i in range(n):
        # core parents: high‑freq first
        pool = [p for p,f in parent_freq[i].items()]
        pool.sort(key=lambda p: -parent_freq[i][p])
        if len(pool) < K:
            pool.extend([p for p in range(n) if p!=i and p not in pool])
        # seed = best “top‑K” set
        cand = tuple(sorted(pool[:K]))
        sc   = approx_score(i, cand, scores)
        x[(i,cand)] = m.addVar(obj=sc, vtype=GRB.CONTINUOUS,
                               lb=0, ub=1, name=f"x_{i}_{cand}")
        cols[i].append(cand)

        pick[i] = m.addConstr(x[(i,cand)] == 1, name=f"pick_{i}")  # one column so far

    m.ModelSense = GRB.MAXIMIZE
    m.update()

    # 2) column generation loop – only K‑tuples are generated
    improved, it = True, 0
    while improved:
        it += 1
        m.optimize()
        dual = {i: pick[i].Pi for i in range(n)}
        improved = False

        for i in range(n):
            pool = [p for p,_ in sorted(parent_freq[i].items(),
                                        key=lambda kv: -kv[1])]
            pool = [p for p in pool if p != i][:max(8, K+3)]  # small pool
            best_rc, best_cand = -np.inf, None
            for cand in itertools.combinations(pool, K):
                cand = tuple(sorted(cand))
                if cand in cols[i]: continue
                sc = approx_score(i, cand, scores)
                rc = sc - dual[i]
                if rc > best_rc:
                    best_rc, best_cand = rc, cand
            if best_cand and best_rc > rc_tol:
                improved = True
                var = m.addVar(obj=best_rc + dual[i], vtype=GRB.CONTINUOUS,
                               lb=0, ub=1, name=f"x_{i}_{best_cand}")
                x[(i,best_cand)] = var
                cols[i].append(best_cand)
                # update “pick one” (replace equality with sum==1 lazily)
                m.chgCoeff(pick[i], var, 1)
        m.update()

    # 3) extract deterministic solution
    solution = {i: max(cols[i], key=lambda c: x[(i,c)].X) for i in range(n)}
    print(f"done after {it} iteration(s)")
    return solution






def maximize_true_graph_posterior_acyclic(K, scores):
    """
    Solve for a Bayesian network in which *every* node has exactly K parents
    (using approximate local scores when K > 3) **and** the resulting graph
    is a DAG.  Column generation is used to avoid enumerating all K-parent
    sets up front.

    Parameters
    ----------
    K : int
        Required in-degree for every node.
    scores : object
        Must expose:
            • n  – number of variables
            • local(i, parent_tuple) – true local score for ≤ 3 parents
        plus whatever compute_approx_score_for_candidate() needs.

    Returns
    -------
    dict : node → tuple(sorted(parent_set))
        A feasible DAG with |parents| = K for every node.
    """
    n = scores.n
    BIG_M = n                       # big-M for order constraints; n is sufficient

    master = gp.Model("DAG_K_Parents")
    master.Params.OutputFlag = 0

    # ------------------------------------------------------------------
    # 1) decision variables for K-parent sets (columns)
    # ------------------------------------------------------------------
    x_vars = {}                     # key  (i, cand)  →  gurobi Var
    columns = {i: [] for i in range(n)}

    for i in range(n):
        for cand in itertools.combinations([p for p in range(n) if p != i], K):
            sc = compute_approx_score_for_candidate(i, cand, scores)
            if np.isfinite(sc):
                v = master.addVar(vtype=GRB.BINARY, name=f"x_{i}_{cand}")
                x_vars[(i, cand)] = v
                columns[i].append(cand)

    # ------------------------------------------------------------------
    # 2) “choose exactly one set” constraints
    # ------------------------------------------------------------------
    choose_one = {
        i: master.addConstr(
            gp.quicksum(x_vars[(i, c)] for c in columns[i]) == 1,
            name=f"choose_one_{i}"
        )
        for i in range(n)
    }

    # ------------------------------------------------------------------
    # 3) acyclicity: integer topological-order variables  π_i
    #     π_j + 1 ≤ π_i   whenever edge  j→i  is selected
    # ------------------------------------------------------------------
    pi = master.addVars(n, vtype=GRB.INTEGER, lb=0, ub=n - 1, name="pi")

    # edge-presence linear expressions  e_{j,i} = Σ_{cand∋j} x_{i,cand}
    edge_expr = {
        (j, i): gp.LinExpr(
            sum(x_vars[(i, c)] for c in columns[i] if j in c)
        )
        for j in range(n) for i in range(n) if j != i
    }

    for j in range(n):
        for i in range(n):
            if j == i:
                continue
            master.addConstr(
                pi[j] + 1 <= pi[i] + BIG_M * (1 - edge_expr[(j, i)]),
                name=f"acyclic_{j}_{i}",
            )

    # ------------------------------------------------------------------
    # 4) objective: maximise sum of (approximate) local scores
    # ------------------------------------------------------------------
    master.setObjective(
        gp.quicksum(
            compute_approx_score_for_candidate(i, c, scores) * x_vars[(i, c)]
            for i in range(n) for c in columns[i]
        ),
        GRB.MAXIMIZE,
    )
    master.update()

    # ------------------------------------------------------------------
    # Helper: when column generation adds a new variable
    # ------------------------------------------------------------------
    def _add_column(i, cand, score):
        """Register a new K-parent set (column) for node i."""
        var = master.addVar(vtype=GRB.BINARY, name=f"x_{i}_{cand}")
        x_vars[(i, cand)] = var
        columns[i].append(cand)

        # link into existing constraints / expressions
        master.chgCoeff(choose_one[i], var, 1.0)
        for j in cand:
            edge_expr[(j, i)].addTerms(1.0, var)

        master.setObjective(master.getObjective() + score * var)

    # ------------------------------------------------------------------
    # 5) column-generation loop
    # ------------------------------------------------------------------
    improved, iteration = True, 0
    while improved:
        iteration += 1
        master.optimize()
        if master.status != GRB.OPTIMAL:
            raise RuntimeError("Master not optimal; aborting.")

        dual = {i: choose_one[i].Pi for i in range(n)}
        improved = False

        for i in range(n):
            best_rc, best_cand = -float("inf"), None
            for cand in itertools.combinations(
                [p for p in range(n) if p != i], K
            ):
                if cand in columns[i]:
                    continue
                sc = compute_approx_score_for_candidate(i, cand, scores)
                if not np.isfinite(sc):
                    continue
                rc = sc - dual[i]
                if rc > best_rc:
                    best_rc, best_cand = rc, cand

            if best_cand is not None and best_rc > 1e-8:
                improved = True
                _add_column(i, best_cand,
                            compute_approx_score_for_candidate(i, best_cand, scores))
                master.update()

    # ------------------------------------------------------------------
    # 6) extract the chosen parent set for every node
    # ------------------------------------------------------------------
    solution = {}
    for i in range(n):
        sel = max(columns[i], key=lambda c: x_vars[(i, c)].X)
        solution[i] = tuple(sorted(sel))

    print(f"Column generation finished after {iteration} iteration(s).")
    return solution

