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
        """
        Return a dictionary: node -> {parents_tuple: local_score},
        for parent sets that are subsets of C[node].
        Handles both dict and np.ndarray for C.
        """



        V = len(C)  # number of nodes
        # Suppose each node i has M_i subsets...
        # For it to be a uniform 2D array, you need the same number of columns for each node,
        # often 2^|C[i]| if you consider all subsets, or something that sumu’s "opt" expects.
        
        # Let's say we do the maximum number of subsets among all i
        max_subset_count = max(2 ** len(C[i]) for i in range(V))

        # Make a big 2D array for (V, max_subset_count)
        arr = np.full((V, max_subset_count), float("-inf"), dtype=float)

        # For each node i:
        for i in range(V):
            # Enumerate subsets of C[i] in some order
            # Suppose subsets_i is a list of (subset_tuple, score)
            subsets_i = []
            for parents_tuple, sc in self.local_scores[i].items():
                # only keep subsets that are within C[i]
                if set(parents_tuple).issubset(C[i]):
                    subsets_i.append((parents_tuple, sc))

            # Sort them in a stable order and store them
            # Typically sumu's "opt" uses bit-encodings, so you'd want j to match that encoding.
            # For simplicity, let's just do enumerated:
            for j, (parents_tuple, sc) in enumerate(subsets_i):
                arr[i, j] = sc

            # If subsets_i has fewer than max_subset_count, the rest remain -inf

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


def bdeu_score_based_voting(K, **kwargs):
    """
    Select candidate parents using BDeu score-based voting.
    
    For each node v, calculate the contribution of each potential parent u to the BDeu score.
    Then select the top-K parents that maximize the score improvement for node v.
    
    Parameters
    ----------
    K : int
        Maximum number of parents to keep for each node.
    kwargs : dict
        Must contain:
            scores: An instance of GobnilpScores-like object.
            n: The number of variables/nodes in the network.
        
    Returns
    -------
    C : dict
        Dictionary of the form { v : tuple_of_parents }.
    """
    scores = kwargs.get("scores", None)
    if scores is None:
        raise ValueError("bdeu_score_based_voting requires 'scores' in kwargs.")
    
    n = kwargs.get("n", None)
    if n is None:
        n = scores.n if hasattr(scores, "n") else scores.data.n
    
    # Initialize the result dictionary for the parents
    C = {}
    
    # For each node v:
    for v in range(n):
        # Potential parents: all nodes except v itself
        potential_parents = [u for u in range(n) if u != v]
        
        # Calculate the score contribution for each potential parent
        parent_contributions = {}
        
        for u in potential_parents:
            # Evaluate the score improvement when adding u as a parent to node v
            improvement = score_improvement(v, u, scores, potential_parents)
            parent_contributions[u] = improvement
        
        # Sort parents by score improvement (descending order)
        sorted_parents = sorted(parent_contributions, key=parent_contributions.get, reverse=True)
        
        # Select the top-K parents based on their score improvement
        C[v] = tuple(sorted(sorted_parents[:K]))
    
    return C


import numpy as np

def synergy_for_node(v, K, scores, alpha=0.0):
    n = scores.n if hasattr(scores, "n") else scores.data.n
    all_candidates = [u for u in range(n) if u != v]

    # Precompute log-BDeu for empty set and singletons
    bdeu_empty = scores.local(v, np.array([], dtype=int))
    bdeu_S = bdeu_empty  # current parent's BDeu for the empty set
    singleton_bdeu = {}
    for u in all_candidates:
        singleton_bdeu[u] = scores.local(v, np.array([u]))

    # Start with no parents
    S = ()
    
    while len(S) < K:
        best_gain = float("-inf")
        best_u = None
        # Current BDeu for the set S
        bdeu_current = scores.local(v, np.array(S))

        for u in all_candidates:
            if u in S:
                continue
            # BDeu(S ∪ {u})
            bdeu_plus_u = scores.local(v, np.array(S + (u,)))
            # synergy gain = BDeu(S ∪ {u}) - BDeu(S) - alpha*(BDeu({u}) - BDeu(∅))
            gain = bdeu_plus_u - bdeu_current
            if alpha > 0:
                gain -= alpha * (singleton_bdeu[u] - bdeu_empty)

            if gain > best_gain:
                best_gain = gain
                best_u = u
        
        if best_gain <= 0 or best_u is None:
            break
        
        # Add the best candidate
        S = tuple(sorted(S + (best_u,)))
    
    return S

def synergy_based_parent_selection(K, scores, alpha=0.0):
    n = scores.n if hasattr(scores, "n") else scores.data.n
    C = {}
    for v in range(n):
        C[v] = synergy_for_node(v, K, scores, alpha)
    return C
import numpy as np

def stability_bdeu(K, scores, data, B=20, threshold=0.5, fill='top'):
    """
    Stability selection for candidate parents using BDeu scores.

    Parameters
    ----------
    K : int
        Max number of parents per node.
    scores : GobnilpScores
        BDeu scoring object for the full dataset (used to get 'n', etc).
    data : array-like or custom data object
        The original dataset. We assume we can bootstrap from it.
    B : int
        Number of bootstrap samples.
    threshold : float in [0,1]
        Frequency threshold for a parent to be considered stable.
    fill : str in { 'top', 'random', None }
        Method to fill/prune parents if stable set != K.

    Returns
    -------
    C : dict
        { v : tuple_of_parents } with exactly K parents or fewer if fill is None.
    """
    n = scores.n  # or scores.data.n
    # freq[v,u] = how many times u was chosen for v
    freq = np.zeros((n, n), dtype=float)

    # For each bootstrap sample:
    for _ in range(B):
        # 1) sample data with replacement
        sampled_data = bootstrap_data(data)  
        # 2) compute BDeu scores for this sampled_data
        #    must return a GobnilpScores-like object with local_scores for each node
        sampled_scores = compute_bdeu_scores(sampled_data)  

        # 3) pick top-K parents in *this bootstrap* for each node
        C_b = pick_top_k_parents(sampled_scores, K)  # a local method you define

        # 4) record
        for v, parents in C_b.items():
            for p in parents:
                freq[v, p] += 1

    # Now freq[v,p] = # times p was chosen as parent of v across B runs
    freq /= B  # convert to fraction in [0,1]

    # Determine stable set for each node v
    C_stable = {}
    for v in range(n):
        # stable parents = all u with freq[v,u] >= threshold
        stable_pars = [u for u in range(n) if u != v and freq[v,u] >= threshold]
        # fill/prune to K
        # e.g., if fill=='top', pick top freq
        if len(stable_pars) > K:
            stable_pars.sort(key=lambda u: freq[v,u], reverse=True)
            stable_pars = stable_pars[:K]
        elif len(stable_pars) < K and fill=='top':
            # add the next highest freq parents
            others = [u for u in range(n) if u not in stable_pars and u != v]
            others.sort(key=lambda u: freq[v,u], reverse=True)
            need = K - len(stable_pars)
            stable_pars += others[:need]

        C_stable[v] = tuple(sorted(stable_pars))

    return C_stable

def bootstrap_data(data):
    """
    Example stub. 
    Returns a bootstrap sample of your dataset (with replacement).
    Adjust to your data structure.
    """
    # If data is a numpy array: 
    idxs = np.random.choice(len(data), size=len(data), replace=True)
    return data[idxs, :]

def compute_bdeu_scores(data):
    """
    Example stub. Fit BDeu or GobnilpScores from 'data'.
    Return a GobnilpScores-like object: local_scores[node][parents_tuple].
    """
    # Pseudocode:
    # parsed_scores = your_own_method(data)
    # return GobnilpScores(parsed_scores)
    pass

def pick_top_k_parents(scores, K):
    """
    Example stub. For each node v, pick top-K parents by 'best local score' 
    from scores.local_scores[v]. 
    Return {v: tuple_of_parents}.
    """
    C_b = {}
    n = scores.n
    for v in range(n):
        if v not in scores.local_scores:
            C_b[v] = ()
            continue
        v_subsets = scores.local_scores[v]
        if not v_subsets:
            C_b[v] = ()
            continue
        # For simplicity, pick single-parent subsets 
        # (or do a real top-K approach over all subsets).
        # We'll just pick the top K singletons by BDeu:
        singletons = {}
        for parents_tuple, val in v_subsets.items():
            if len(parents_tuple) == 1:
                singletons[parents_tuple[0]] = val
        
        # sort singletons
        best_pars = sorted(singletons.keys(), key=lambda p: singletons[p], reverse=True)[:K]
        C_b[v] = tuple(best_pars)
    return C_b
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


def bdeu_score_based_voting(K, **kwargs):
    """
    Select candidate parents using BDeu score-based voting.
    
    For each node v, calculate the contribution of each potential parent u to the BDeu score.
    Then select the top-K parents that maximize the score improvement for node v.
    
    Parameters
    ----------
    K : int
        Maximum number of parents to keep for each node.
    kwargs : dict
        Must contain:
            scores: An instance of GobnilpScores-like object.
            n: The number of variables/nodes in the network.
        
    Returns
    -------
    C : dict
        Dictionary of the form { v : tuple_of_parents }.
    """
    scores = kwargs.get("scores", None)
    if scores is None:
        raise ValueError("bdeu_score_based_voting requires 'scores' in kwargs.")
    
    n = kwargs.get("n", None)
    if n is None:
        n = scores.n if hasattr(scores, "n") else scores.data.n
    
    # Initialize the result dictionary for the parents
    C = {}
    
    # For each node v:
    for v in range(n):
        # Potential parents: all nodes except v itself
        potential_parents = [u for u in range(n) if u != v]
        
        # Calculate the score contribution for each potential parent
        parent_contributions = {}
        
        for u in potential_parents:
            # Evaluate the score improvement when adding u as a parent to node v
            improvement = score_improvement(v, u, scores, potential_parents)
            parent_contributions[u] = improvement
        
        # Sort parents by score improvement (descending order)
        sorted_parents = sorted(parent_contributions, key=parent_contributions.get, reverse=True)
        
        # Select the top-K parents based on their score improvement
        C[v] = tuple(sorted(sorted_parents[:K]))
    
    return C

