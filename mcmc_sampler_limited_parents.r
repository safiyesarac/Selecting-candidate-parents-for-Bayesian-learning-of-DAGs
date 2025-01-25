#!/usr/bin/env Rscript

# ===========================
# 1) Parse .jkl into a hash env, optionally rescaling log-scores
# ===========================
parse_jkl_to_env <- function(file_path, scale_factor = 0.001) {
  scores_env <- new.env(hash = TRUE, size = 1e5)
  con <- file(file_path, open = "r")
  
  # 1) Read the first line => number_of_nodes
  first_line <- readLines(con, n = 1, warn = FALSE)
  # not strictly used, but we read it to advance the cursor
  
  # 2) For each node block: "node_id count", then `count` lines: "<logscore> <num_parents> p1 p2 ..."
  while (TRUE) {
    node_line <- readLines(con, n = 1, warn = FALSE)
    if (!length(node_line)) break  # EOF
    node_line <- trimws(node_line)
    if (!nchar(node_line)) next    # skip empty
    
    parts <- strsplit(node_line, "\\s+")[[1]]
    if (length(parts) != 2) {
      warning("Unexpected node header line: ", node_line)
      next
    }
    current_node <- suppressWarnings(as.integer(parts[1]))
    lines_count  <- suppressWarnings(as.integer(parts[2]))
    if (is.na(current_node) || is.na(lines_count)) {
      warning("Node header parse error: ", node_line)
      next
    }
    
    for (i in seq_len(lines_count)) {
      score_line <- readLines(con, n = 1, warn = FALSE)
      if (!length(score_line)) {
        warning(sprintf("File ended prematurely for node %d", current_node))
        break
      }
      score_line <- trimws(score_line)
      if (!nchar(score_line)) next
      
      sl_parts <- strsplit(score_line, "\\s+")[[1]]
      if (length(sl_parts) < 2) {
        warning("Malformed score line: ", score_line)
        next
      }
      
      raw_score_val <- suppressWarnings(as.numeric(sl_parts[1]))
      num_par       <- suppressWarnings(as.integer(sl_parts[2]))
      if (is.na(raw_score_val) || is.na(num_par)) {
        warning("Score or parent count parse error: ", score_line)
        next
      }
      
      # Rescale if desired
      score_val <- raw_score_val * scale_factor
      
      if (num_par > 0) {
        par_nodes <- suppressWarnings(as.integer(sl_parts[3:(2 + num_par)]))
      } else {
        par_nodes <- integer(0)
      }
      
      # Build key: "child|p1,p2,..."
      sorted_par <- sort(par_nodes)
      key_str <- paste0(current_node, "|", paste(sorted_par, collapse=","))
      assign(key_str, score_val, envir = scores_env)
    }
  }
  close(con)
  return(scores_env)
}

# ===========================
# 2) Local node score lookup for a single node’s parents
# ===========================
local_score_node <- function(adjmat, node_i, scores_env) {
  # child => node_i
  # (row, col) => (parent, child)
  parents_idx <- which(adjmat[, node_i + 1] == 1) - 1
  key <- paste0(node_i, "|", paste(sort(parents_idx), collapse=","))
  val <- scores_env[[key]]
  if (is.null(val)) return(-Inf)
  return(val)
}

# ===========================
# 3) Check for cycles with DFS
# ===========================
has_cycle <- function(adjmat) {
  N <- nrow(adjmat)
  visited <- integer(N)  # 0=unvisited, 1=visiting, 2=done
  
  dfs <- function(u) {
    if (visited[u] == 1L) return(TRUE)  # found back-edge => cycle
    if (visited[u] == 2L) return(FALSE)
    visited[u] <<- 1L
    for (v in which(adjmat[u, ] == 1)) {
      if (dfs(v)) return(TRUE)
    }
    visited[u] <<- 2L
    return(FALSE)
  }
  
  for (u in seq_len(N)) {
    if (visited[u] == 0L) {
      if (dfs(u)) return(TRUE)
    }
  }
  return(FALSE)
}

# ===========================
# 4) MCMC step
#    - 'include_parent' toggles whether local score includes parent's perspective
#    - 'max_in_degree' limits # of parents a child can have
# ===========================
mcmc_step <- function(adjmat, scores_env, iteration=NA, temperature=1.0,
                      include_parent=FALSE, max_in_degree=Inf) {
  N <- nrow(adjmat)
  
  # Randomly choose an edge to flip (i <- child, j <- parent)
  repeat {
    i <- sample.int(N, 1) - 1  # child
    j <- sample.int(N, 1) - 1  # parent
    if (i != j) break
  }
  
  old_val <- adjmat[j+1, i+1]
  new_val <- if (old_val == 1) 0 else 1
  
  # Propose flipping edge j -> i
  adjmat[j+1, i+1] <- new_val
  
  # 1) Check in-degree constraint if adding edge
  if (new_val == 1) {
    in_degree_i <- sum(adjmat[, i+1])
    if (in_degree_i > max_in_degree) {
      # revert immediately; reject
      adjmat[j+1, i+1] <- old_val
      cat(sprintf("Iter=%d: Proposed edge %d->%d => would exceed in-degree limit => REJECT\n",
                  iteration, j, i))
      return(adjmat)
    }
  }
  
  # 2) Check cycle
  if (has_cycle(adjmat)) {
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: Proposed edge %d->%d => cycle => REJECT\n", iteration, j, i))
    return(adjmat)
  }
  
  # ----------------------------------------------------------------------
  # If your .jkl is purely child-based (standard BDeu), only node i changes:
  # ----------------------------------------------------------------------
  
  # measure old local score_i by toggling that edge in a temp manner:
  adjmat[j+1, i+1] <- old_val
  old_score_i <- local_score_node(adjmat, i, scores_env)
  adjmat[j+1, i+1] <- new_val
  new_score_i <- local_score_node(adjmat, i, scores_env)
  
  # ----------------------------------------------------------------------
  # If your scoring is "nonsymmetric" (depends on parent node j as well),
  # then *ALSO* measure old/new local scores for node j if include_parent=TRUE.
  # ----------------------------------------------------------------------
  old_score_j <- 0
  new_score_j <- 0
  if (include_parent) {
    # measure parent's old score (temp revert)  
    adjmat[j+1, i+1] <- old_val
    old_score_j <- local_score_node(adjmat, j, scores_env)
    # measure parent's new score (with flipped edge)
    adjmat[j+1, i+1] <- new_val
    new_score_j <- local_score_node(adjmat, j, scores_env)
  }
  
  # revert adjacency so difference is computed properly
  adjmat[j+1, i+1] <- old_val
  total_old <- old_score_i + old_score_j
  total_new <- new_score_i + new_score_j
  
  # Handle case where both totals are -Inf to avoid NaN in difference
  if (is.infinite(total_old) && is.infinite(total_new)) {
    diff_val <- 0
  } else {
    diff_val <- total_new - total_old
  }
  
  ratio <- exp(temperature * diff_val)
  
  # final accept/reject with NA check for ratio
  if (is.na(ratio)) {
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: %d->%d resulted in undefined ratio => REJECT\n", iteration, j, i))
  } else if (runif(1) < ratio) {
    # accept => keep the new_val
    adjmat[j+1, i+1] <- new_val
    cat(sprintf("Iter=%d: %d->%d old=%.4f new=%.4f diff=%.4f => ratio=%.4f => ACCEPT\n",
                iteration, j, i, total_old, total_new, diff_val, ratio))
  } else {
    # reject => revert to old
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: %d->%d old=%.4f new=%.4f diff=%.4f => ratio=%.4f => REJECT\n",
                iteration, j, i, total_old, total_new, diff_val, ratio))
  }
  
  return(adjmat)
}

# ===========================
# 5) DAG -> string
# ===========================
dag_to_string <- function(adjmat) {
  N <- ncol(adjmat)
  out <- character(N)
  for (child in seq_len(N)) {
    node_id <- child - 1
    parents <- which(adjmat[, child] == 1) - 1
    if (!length(parents)) {
      out[child] <- sprintf("%d <- {}", node_id)
    } else {
      out[child] <- sprintf("%d <- {%s}", node_id, paste(parents, collapse=","))
    }
  }
  paste(out, collapse=", ")
}

# ===========================
# 6) MAIN
# ===========================
args <- commandArgs(TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript my_mcmc_sampler.R <jkl_file> <num_iterations> <outfile> [num_samples] [max_in_degree]\n")
  quit(status = 1)
}
jkl_file    <- args[1]
num_iter    <- as.integer(args[2])
outfile     <- args[3]
num_samples <- if (length(args) >= 4) as.integer(args[4]) else 10
max_in_degree <- if (length(args) >= 5) as.integer(args[5]) else Inf

# ----------------------------------------------------------------------
# Tuning parameters for your scenario:
# ----------------------------------------------------------------------
scale_factor <- 0.001
temperature  <- 5.0
# By default, we assume child-only BDeu. If your .jkl is nonsymmetric, set TRUE:
include_parent_score <- FALSE  
# ----------------------------------------------------------------------

cat("Reading local scores from:", jkl_file, "\n")
cat("Using scale_factor =", scale_factor, "\n")
scores_env <- parse_jkl_to_env(jkl_file, scale_factor=scale_factor)
all_keys   <- ls(scores_env)
cat("Number of node|parents combos parsed:", length(all_keys), "\n")

# Infer number of nodes from largest node index
max_node <- 0
for (k in all_keys) {
  node_str <- strsplit(k, "\\|")[[1]][1]
  node_i   <- as.integer(node_str)
  if (!is.na(node_i) && node_i > max_node) {
    max_node <- node_i
  }
}
N <- max_node + 1
cat("Inferred N =", N, "\n")

# Initialize random DAG with ~10% edges, remove edges until acyclic
adjmat <- matrix(rbinom(N*N, 1, 0.1), nrow=N)
diag(adjmat) <- 0
while (has_cycle(adjmat)) {
  ones <- which(adjmat == 1, arr.ind=TRUE)
  kill_idx <- sample(seq_len(nrow(ones)), 1)
  adjmat[ ones[kill_idx,1], ones[kill_idx,2] ] <- 0
}

# Ensure no node exceeds max_in_degree from the start
for (col_i in seq_len(N)) {
  while (sum(adjmat[, col_i]) > max_in_degree) {
    # randomly remove edges in this column until within limit
    parents <- which(adjmat[, col_i] == 1)
    kill_idx <- sample(parents, 1)
    adjmat[kill_idx, col_i] <- 0
  }
}

cat("Initial DAG:\n", dag_to_string(adjmat), "\n\n")
cat("Running MCMC for", num_iter, "burn-in iterations...\n")

# Burn-in
for (iter in seq_len(num_iter)) {
  adjmat <- mcmc_step(
    adjmat,
    scores_env,
    iteration=iter,
    temperature=temperature,
    include_parent=include_parent_score,
    max_in_degree=max_in_degree
  )
}

cat("\nSampling", num_samples, "DAGs => writing to", outfile, "\n")
cat("", file=outfile)  # overwrite file

# Thin the chain => # of steps between successive samples
thin_steps <- 50

for (s in seq_len(num_samples)) {
  for (k in seq_len(thin_steps)) {
    iter <- num_iter + (s-1)*thin_steps + k
    adjmat <- mcmc_step(
      adjmat,
      scores_env,
      iteration=iter,
      temperature=temperature,
      include_parent=include_parent_score,
      max_in_degree=max_in_degree
    )
  }
  dag_str <- dag_to_string(adjmat)
  cat(sprintf("Sample %d => %s\n", s, dag_str))
  cat(dag_str, "\n", file=outfile, append=TRUE)
}

cat("Done.\n")
