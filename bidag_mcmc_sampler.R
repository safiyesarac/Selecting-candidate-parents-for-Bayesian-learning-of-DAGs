# Load required libraries
library(MCMCpack)
library(MASS)  # For multivariate Gaussian likelihood
library(parallel)  # For parallel execution of chains

# Specify the path to your CSV file
file_path <- "~/Downloads/thesis/data/asia_dataset.csv"

# Read the dataset
data <- read.csv(file_path, header = TRUE)

# Convert the data frame to a numerical matrix
data_matrix <- as.matrix(sapply(data, as.numeric))

# Number of nodes (variables in the DAG)
num_nodes <- ncol(data_matrix)

# Number of MCMC iterations
total_iter <- 12000
num_chains <- 4  # Number of MCMC chains to run
target_dags <- 10000  # Target number of unique DAGs

# --- Likelihood Function ---
calculate_likelihood <- function(dag, data_matrix) {
  likelihood <- 0  
  for (j in 1:num_nodes) {
    parents <- which(dag[, j] == 1)  
    if (length(parents) == 0) {
      log_likelihood <- sum(dnorm(data_matrix[, j], mean = mean(data_matrix[, j]), 
                                  sd = sd(data_matrix[, j]), log = TRUE))
    } else {
      X <- data_matrix[, parents, drop = FALSE]
      y <- data_matrix[, j]
      fit <- lm(y ~ X)
      sigma <- sd(fit$residuals)
      predicted_mean <- predict(fit, newdata = as.data.frame(X))
      log_likelihood <- sum(dnorm(y, mean = predicted_mean, sd = sigma, log = TRUE))
    }
    likelihood <- likelihood + log_likelihood  
  }
  return(as.numeric(likelihood))  
}

# --- Function to Ensure Acyclicity ---
ensure_acyclic <- function(dag) {
  temp_dag <- dag
  in_degree <- colSums(temp_dag)
  queue <- which(in_degree == 0)
  visited <- 0
  while (length(queue) > 0) {
    node <- queue[1]
    queue <- queue[-1]
    visited <- visited + 1
    for (j in 1:num_nodes) {
      if (temp_dag[node, j] == 1) {
        temp_dag[node, j] <- 0
        in_degree[j] <- in_degree[j] - 1
        if (in_degree[j] == 0) {
          queue <- c(queue, j)
        }
      }
    }
  }
  return(visited == num_nodes)
}

# --- Initialize a Random DAG ---
initialize_dag <- function(n) {
  dag <- matrix(0, nrow = n, ncol = n)
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      dag[i, j] <- sample(0:1, 1)
    }
  }
  return(dag)
}

# --- Proposal Step: Modify a DAG ---
propose_dag <- function(dag) {
  new_dag <- dag
  i <- sample(1:num_nodes, 1)
  j <- sample(1:num_nodes, 1)
  new_dag[i, j] <- 1 - new_dag[i, j]
  if (i == j) new_dag[i, j] <- 0
  if (!ensure_acyclic(new_dag)) return(dag)
  return(new_dag)
}

# --- Metropolis-Hastings MCMC Sampling (Single Chain) ---
metropolis_mcmc <- function(chain_id, total_iter, max_dags) {
  dag <- initialize_dag(num_nodes)
  current_likelihood <- calculate_likelihood(dag, data_matrix)
  sampled_dags <- list()
  dag_hashes <- list()
  
  for (iter in 1:total_iter) {
    proposed_dag <- propose_dag(dag)
    proposed_likelihood <- calculate_likelihood(proposed_dag, data_matrix)
    acceptance_prob <- exp(proposed_likelihood - current_likelihood)
    acceptance_prob <- min(1, as.numeric(acceptance_prob))  
    if (runif(1) < acceptance_prob) {
      dag <- proposed_dag
      current_likelihood <- proposed_likelihood
    }
    dag_string <- paste(as.vector(dag), collapse = "")
    if (!(dag_string %in% dag_hashes)) {
      sampled_dags <- append(sampled_dags, list(dag))
      dag_hashes <- append(dag_hashes, dag_string)
    }
    if (length(sampled_dags) >= max_dags) break  
    if (iter %% 1000 == 0) {
      cat("Chain:", chain_id, "Iteration:", iter, "Unique DAGs:", length(sampled_dags), "\n")
    }
  }
  return(sampled_dags)
}

# --- Run Multiple Chains in Parallel ---
run_multiple_chains <- function(num_chains, total_iter, target_dags) {
  cl <- makeCluster(num_chains)  
  clusterExport(cl, c("initialize_dag", "propose_dag", "ensure_acyclic",
                      "calculate_likelihood", "metropolis_mcmc", "data_matrix", 
                      "num_nodes", "total_iter", "target_dags"))  # ✅ FIXED: Export `target_dags`
  results <- parLapply(cl, 1:num_chains, function(chain_id) {
    metropolis_mcmc(chain_id, total_iter, target_dags / num_chains)
  })
  stopCluster(cl)  
  return(results)
}

# Run Multiple MCMC Chains
sampled_dags_all_chains <- run_multiple_chains(num_chains, total_iter, target_dags)

# --- Collect Unique DAGs Across All Chains ---
unique_dags <- list()
dag_hashes <- list()
for (chain in 1:num_chains) {
  for (dag in sampled_dags_all_chains[[chain]]) {
    dag_string <- paste(as.vector(dag), collapse = "")
    if (!(dag_string %in% dag_hashes)) {
      unique_dags <- append(unique_dags, list(dag))
      dag_hashes <- append(dag_hashes, dag_string)
    }
    if (length(unique_dags) >= target_dags) break
  }
  if (length(unique_dags) >= target_dags) break
}

cat("Total Unique DAGs:", length(unique_dags), "\n")

# --- Save Unique DAGs to File ---
output_file <- "~/Downloads/thesis/data/asia_unique_sampled_mh_dags.txt"
con <- file(output_file, open = "w")

for (dag in unique_dags) {
  node_strs <- vector("character", num_nodes)
  for (j in 1:num_nodes) {
    parent_indices <- which(dag[, j] == 1)
    parent_nodes <- sort(parent_indices - 1)
    
    # Fix: Handling empty parent sets as "{}" instead of "{{}}"
    parent_str <- if (length(parent_nodes) == 0) "{}" else paste(parent_nodes, collapse = ", ")
    
    # Construct the DAG representation for the current node
    node_strs[j] <- paste0(j - 1, " <- {", parent_str, "}")
  }
  
  # Join all nodes into a single line (One DAG per line)
  dag_line <- paste(node_strs, collapse = ", ")
  writeLines(dag_line, con)
}

close(con)
cat("Unique DAGs have been written to", output_file, "\n")
