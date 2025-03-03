# import numpy as np
# from itertools import combinations
# from scipy.optimize import linprog

# # --- Dummy parser and classes for demonstration purposes ---

# import data_io
# import heuristics
# # --- ILP with Cutting Planes ---
# class ILPwithCuttingPlanes:
#     def __init__(self, scores):
#         self.scores = scores
#         self.n_nodes = scores.n
#         self.local_scores = scores.local_scores

#     def formulate_ilp(self):
#         """
#         Formulate the ILP by creating an objective vector (c) and
#         equality constraints (A_eq and b_eq) so that exactly one candidate
#         parent set is selected per node.
#         """
#         n = self.n_nodes
#         c = []
#         for node in range(n):
#             # Extend c with the score for each candidate parent set for this node
#             c.extend(self.local_scores[node].values())
#         # Each node must have exactly one selected parent set.
#         A_eq = np.zeros((n, len(c)))
#         b_eq = np.ones(n)
#         idx = 0
#         for node in range(n):
#             for parent_set in self.local_scores[node].keys():
#                 A_eq[node, idx] = 1
#                 idx += 1
#         return c, A_eq, b_eq

#     def solve_ilp(self, c, A_eq, b_eq):
#         """
#         Solve the ILP using scipy's linprog.
#         (Note: This performs a linear relaxation; in practice you might
#         want to use a dedicated MILP solver such as Gurobi or CPLEX.)
#         """
#         result = linprog(c=-np.array(c), A_eq=A_eq, b_eq=b_eq,
#                          bounds=[(0, 1)] * len(c), method='highs')
#         if result.success:
#             return result.x
#         else:
#             return None

#     def add_cutting_plane(self, A_eq, b_eq, violated_constraint):
#         """
#         Add a cutting plane constraint.
#         violated_constraint should be a tuple (new_row, rhs).
#         """
#         new_constraint = violated_constraint
#         A_eq = np.vstack([A_eq, new_constraint[0]])
#         b_eq = np.append(b_eq, new_constraint[1])
#         return A_eq, b_eq

#     def check_violated_constraints(self, solution):
#         """
#         Placeholder for checking if any domain-specific constraints are violated.
#         Implement logic here to detect violations based on your criteria.
#         For this example, we assume no violations.
#         """
#         return None

#     def solve_with_cutting_planes(self, max_iter=10):
#         """
#         Solve the ILP, iteratively adding cutting planes if any constraint
#         is violated.
#         """
#         c, A_eq, b_eq = self.formulate_ilp()
#         solution = self.solve_ilp(c, A_eq, b_eq)
#         if solution is None:
#             print("Initial ILP solution failed")
#             return None
#         for _ in range(max_iter):
#             violated_constraint = self.check_violated_constraints(solution)
#             if violated_constraint:
#                 A_eq, b_eq = self.add_cutting_plane(A_eq, b_eq, violated_constraint)
#                 solution = self.solve_ilp(c, A_eq, b_eq)
#                 if solution is None:
#                     print("ILP solution failed after cutting plane addition")
#                     return None
#             else:
#                 return solution
#         return solution

# def interpret_solution(solution, scores):
#     """
#     Convert the flattened solution vector back into a configuration
#     mapping each node to its selected candidate parent set.
#     """
#     parent_set_config = []
#     idx = 0
#     for node in range(scores.n):
#         num_candidates = len(scores.local_scores[node])
#         for i in range(num_candidates):
#             if solution[idx + i] == 1:
#                 # Retrieve the candidate parent set (by order of keys)
#                 candidate = list(scores.local_scores[node].keys())[i]
#                 parent_set_config.append((node, candidate))
#         idx += num_candidates
#     return parent_set_config

# def filter_parent_sets_by_size(scores, k):
#     for node in scores.local_scores:
#         scores.local_scores[node] = {
#             parents: score
#             for parents, score in scores.local_scores[node].items()
#             if len(parents) == k
#         }

# # --- Main Execution ---
# if __name__ == '__main__':
#     # Replace with your actual parser if available:
#     parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/asia/asia_scores.jkl')
#     scores = heuristics.GobnilpScores(parsed_scores)
    
#     # Restrict candidate parent sets to exactly k parents.
#     k = 2 # Change k to the desired number of parents.

    
#     print("Filtered local_scores (only parent sets of size {}):".format(k))
#     for node, candidates in scores.local_scores.items():
#         print("Node {}: {}".format(node, candidates))
    
#     # Create and solve the ILP with cutting planes.
#     ilp_solver = ILPwithCuttingPlanes(scores)
#     best_solution = ilp_solver.solve_with_cutting_planes()
    
#     if best_solution is not None:
#         print("\nBest parent set configuration (solution vector):")
#         print(best_solution)
#         config = interpret_solution(best_solution, scores)
#         print("\nInterpreted configuration:")
#         for node, parents in config:
#             print(f"Node {node} selected parent set: {parents}")
#     else:
#         print("No feasible solution found.")
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

def approximate_local_score(node, parents, local_scores):
    """
    Approximate local score using existing up-to-3-parent data.
    If parents has size > 3, pick best 3-subset's real local score as a stand-in.
    """
    if len(parents) <= 3:
        return local_scores.local(node, tuple(sorted(parents)))

    best_sub_score = -float("inf")
    for sub in itertools.combinations(parents, 3):
        sc = local_scores.local(node, tuple(sorted(sub)))
        if sc > best_sub_score:
            best_sub_score = sc
    return best_sub_score

def maximize_true_graph_posterior(local_scores, n, K):
    """
    Column generation for exactly K parents per node,
    but if K>3, we approximate the local score of a bigger set by
    the best 3-parent subset (or another scheme).

    Returns: a dict of node -> chosen parent set (size exactly K).
    """
    master = gp.Model("MasterProblem")
    master.Params.OutputFlag = 0

    x_vars = {}
    columns = {}

    # Step 1: build candidate columns of size K
    for i in range(n):
        columns[i] = []
        possible_parents = [p for p in range(n) if p != i]

        # All combinations of size K
        for cand in itertools.combinations(possible_parents, K):
            sc = approximate_local_score(i, cand, local_scores)
            # If the approximate score is not -inf or NaN, keep it
            if np.isfinite(sc):
                var = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                    name=f"x_{i}_{cand}")
                x_vars[(i, cand)] = var
                columns[i].append(cand)

    master.update()

    # Step 2: constraints
    constrs = {}
    for i in range(n):
        if not columns[i]:
            raise ValueError(f"No feasible K={K} parent sets (even approximate) for node {i}.")
        constrs[i] = master.addConstr(
            gp.quicksum(x_vars[(i, c)] for c in columns[i]) == 1,
            name=f"node_{i}"
        )
    master.update()

    # Step 3: objective
    obj_expr = gp.LinExpr()
    for i in range(n):
        for cand in columns[i]:
            sc = approximate_local_score(i, cand, local_scores)
            obj_expr.addTerms(sc, x_vars[(i, cand)])
    master.setObjective(obj_expr, GRB.MAXIMIZE)
    master.update()

    # Step 4: Column generation loop
    improved = True
    iteration = 0
    while improved:
        iteration += 1
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print(f"Master not solved to optimality at iteration {iteration}. Stopping.")
            break

        # Dual values
        duals = {i: constrs[i].Pi for i in range(n)}
        improved = False

        # Pricing
        for i in range(n):
            best_rc = -float('inf')
            best_cand = None
            for cand in itertools.combinations([p for p in range(n) if p != i], K):
                if cand in columns[i]:
                    continue
                sc = approximate_local_score(i, cand, local_scores)
                if not np.isfinite(sc):
                    continue
                rc = sc - duals[i]
                if rc > best_rc:
                    best_rc = rc
                    best_cand = cand

            if best_cand is not None and best_rc > 1e-8:
                improved = True
                columns[i].append(best_cand)
                var = master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                    name=f"x_{i}_{best_cand}")
                x_vars[(i, best_cand)] = var
                master.chgCoeff(constrs[i], var, 1.0)
                # We can add (sc) * var or directly (rc) * var + duals[i]*var, etc.
                # But simpler is to just do sc * var:
                master.setObjective(master.getObjective() + rc * var)
                master.update()

    # Step 5: Extract solution
    solution = {}
    for i in range(n):
        best_val = -1
        best_cand = None
        for cand in columns[i]:
            val = x_vars[(i, cand)].X
            if val > best_val:
                best_val = val
                best_cand = cand
        solution[i] = tuple(sorted(best_cand)) if best_cand else ()

    print(f"Done. Column generation took {iteration} iteration(s).")
    return solution


# --------------------
# Example usage (pseudo code; adapt paths and modules as needed)

if __name__ == "__main__":
    import data_io
    import heuristics
    import heuristics_variable_data_experiments as var

    # Parse scores from your Gobnilp .jkl file
    parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/sachs/sachs_scores.jkl')
    scores = heuristics.GobnilpScores(parsed_scores)

    # Solve for some K > 3, e.g. K=4
    K = 4
    solution = maximize_true_graph_posterior(scores, scores.n, K)
    print("Solution:", solution)

    # Evaluate posterior of the found network
    bn = BayesianNetwork(solution, scores)
    posterior = bn.compute_posterior()
    print("Log Posterior of Found Network:", posterior)

    # Compare to the 'true' or reference network
    true_parents = var.get_true_parents('/home/gulce/Downloads/thesis/data/sachs/sachs.bif')
    bn_true = BayesianNetwork(true_parents, scores)
    print("True Parents:", true_parents)
    posterior_true = bn_true.compute_posterior()
    print("Log Posterior of Reference Network:", posterior_true)

    bn = BayesianNetwork({0: (2, 3), 1: (0, 2), 2: (0, 1), 3: (0, 2), 4: (2, 5), 5: (2, 3), 6: (2, 3), 7: (2, 5), 8: (9, 10), 9: (8, 10), 10: (8, 9)}, scores)
  
    posterior = bn.compute_posterior()
    print("Log of the Posterior Probability of the opT Network:", posterior)

