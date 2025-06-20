




parse_jkl_to_env <- function(file_path, scale_factor = 0.001) {
  scores_env <- new.env(hash = TRUE, size = 1e5)
  con <- file(file_path, open = "r")
  
  
  first_line <- readLines(con, n = 1, warn = FALSE)
  
  
  
  while (TRUE) {
    node_line <- readLines(con, n = 1, warn = FALSE)
    if (!length(node_line)) break  
    node_line <- trimws(node_line)
    if (!nchar(node_line)) next    
    
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
      
      
      score_val <- raw_score_val * scale_factor
      
      if (num_par > 0) {
        par_nodes <- suppressWarnings(as.integer(sl_parts[3:(2 + num_par)]))
      } else {
        par_nodes <- integer(0)
      }
      
      
      sorted_par <- sort(par_nodes)
      key_str <- paste0(current_node, "|", paste(sorted_par, collapse=","))
      assign(key_str, score_val, envir = scores_env)
    }
  }
  close(con)
  return(scores_env)
}




local_score_node <- function(adjmat, node_i, scores_env) {
  
  
  parents_idx <- which(adjmat[, node_i + 1] == 1) - 1
  key <- paste0(node_i, "|", paste(sort(parents_idx), collapse=","))
  val <- scores_env[[key]]
  if (is.null(val)) return(-Inf)
  return(val)
}




has_cycle <- function(adjmat) {
  N <- nrow(adjmat)
  visited <- integer(N)  
  
  dfs <- function(u) {
    if (visited[u] == 1L) return(TRUE)  
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






mcmc_step <- function(adjmat, scores_env, iteration=NA, temperature=1.0,
                      include_parent=FALSE) {
  N <- nrow(adjmat)
  
  repeat {
    i <- sample.int(N, 1) - 1  
    j <- sample.int(N, 1) - 1  
    if (i != j) break
  }
  
  old_val <- adjmat[j+1, i+1]
  new_val <- if (old_val == 1) 0 else 1
  adjmat[j+1, i+1] <- new_val
  
  
  if (has_cycle(adjmat)) {
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: Proposed edge %d->%d => cycle => REJECT\n", iteration, j, i))
    return(adjmat)
  }
  
  
  
  
  
  
  adjmat[j+1, i+1] <- old_val
  old_score_i <- local_score_node(adjmat, i, scores_env)
  adjmat[j+1, i+1] <- new_val
  new_score_i <- local_score_node(adjmat, i, scores_env)
  
  
  
  
  
  old_score_j <- 0
  new_score_j <- 0
  if (include_parent) {
    
    adjmat[j+1, i+1] <- old_val
    old_score_j <- local_score_node(adjmat, j, scores_env)
    
    adjmat[j+1, i+1] <- new_val
    new_score_j <- local_score_node(adjmat, j, scores_env)
  }
  
  
  adjmat[j+1, i+1] <- old_val
  total_old <- old_score_i + old_score_j
  total_new <- new_score_i + new_score_j
  
  
  if (is.infinite(total_old) && is.infinite(total_new)) {
    diff_val <- 0
  } else {
    diff_val <- total_new - total_old
  }
  
  ratio <- exp(temperature * diff_val)
  
  
  if (is.na(ratio)) {
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: %d->%d resulted in undefined ratio => REJECT\n", iteration, j, i))
  } else if (runif(1) < ratio) {
    
    adjmat[j+1, i+1] <- new_val
    cat(sprintf("Iter=%d: %d->%d old=%.4f new=%.4f diff=%.4f => ratio=%.4f => ACCEPT\n",
                iteration, j, i, total_old, total_new, diff_val, ratio))
  } else {
    
    adjmat[j+1, i+1] <- old_val
    cat(sprintf("Iter=%d: %d->%d old=%.4f new=%.4f diff=%.4f => ratio=%.4f => REJECT\n",
                iteration, j, i, total_old, total_new, diff_val, ratio))
  }
  
  return(adjmat)
}




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




args <- commandArgs(TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript my_mcmc_sampler.R <jkl_file> <num_iterations> <outfile> [num_samples]\n")
  quit(status = 1)
}
jkl_file    <- args[1]
num_iter    <- as.integer(args[2])
outfile     <- args[3]
num_samples <- if (length(args) >= 4) as.integer(args[4]) else 10




scale_factor <- 0.001
temperature  <- 5.0

include_parent_score <- FALSE  


cat("Reading local scores from:", jkl_file, "\n")
cat("Using scale_factor =", scale_factor, "\n")
scores_env <- parse_jkl_to_env(jkl_file, scale_factor=scale_factor)
all_keys   <- ls(scores_env)
cat("Number of node|parents combos parsed:", length(all_keys), "\n")


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


adjmat <- matrix(rbinom(N*N, 1, 0.1), nrow=N)
diag(adjmat) <- 0
while (has_cycle(adjmat)) {
  ones <- which(adjmat == 1, arr.ind=TRUE)
  kill_idx <- sample(seq_len(nrow(ones)), 1)
  adjmat[ ones[kill_idx,1], ones[kill_idx,2] ] <- 0
}

cat("Initial DAG:\n", dag_to_string(adjmat), "\n\n")
cat("Running MCMC for", num_iter, "burn-in iterations...\n")


for (iter in seq_len(num_iter)) {
  adjmat <- mcmc_step(
    adjmat,
    scores_env,
    iteration=iter,
    temperature=temperature,
    include_parent=include_parent_score
  )
}

cat("\nSampling", num_samples, "DAGs => writing to", outfile, "\n")
cat("", file=outfile)  


thin_steps <- 50

for (s in seq_len(num_samples)) {
  for (k in seq_len(thin_steps)) {
    iter <- num_iter + (s-1)*thin_steps + k
    adjmat <- mcmc_step(
      adjmat,
      scores_env,
      iteration=iter,
      temperature=temperature,
      include_parent=include_parent_score
    )
  }
  dag_str <- dag_to_string(adjmat)
  cat(sprintf("Sample %d => %s\n", s, dag_str))
  cat(dag_str, "\n", file=outfile, append=TRUE)
}

cat("Done.\n")
