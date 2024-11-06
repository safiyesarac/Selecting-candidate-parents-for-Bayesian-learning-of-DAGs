from nonsymmetric_sampler import py_calculate_hat_weights

# Test with sample BDeu scores
size = 3  # Example size of the DAG
weights = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]

# Call the function and print the result
result = py_calculate_hat_weights(size, weights)
print("Result of nonsymmetric DAG sampling:", result)



# Assume py_run_nonsymmetric_sampler is correctly set up in nonsymmetric_sampler.pyx

from nonsymmetric_sampler import py_run_nonsymmetric_sampler
import re
def parse_jkl_scores(file_path, max_rows=8):
    """Parses the JKL scores file to extract up to max_rows scores, formatted as a list of lists."""
    scores = []
    with open(file_path, 'r') as file:
        for line in file:
            # Stop if we've reached the desired number of rows
            if len(scores) >= max_rows:
                break

            # Skip lines that do not start with a numeric score (e.g., metadata lines)
            if not re.match(r"^-?\d+\.\d+", line.strip()):
                continue

            parts = line.strip().split()
            try:
                # The first part of each valid line is the score
                score = float(parts[0])
                scores.append([score])  # Wrap in a list for compatibility
            except ValueError as e:
                print(f"Warning: Could not convert line to float: {line.strip()} - {e}")
    return scores

# Use the function and print results for debugging
jkl_scores = parse_jkl_scores('modular-dag-sampling-master/scores.jkl')
#print("Parsed JKL scores (as list of lists):", jkl_scores)  # Verify the format before passing to the sampler

import logging
from logging.handlers import RotatingFileHandler
from nonsymmetric_sampler import py_run_nonsymmetric_sampler

# Configure the logger
log_file = "sampler.log"
max_log_size = 5 * 1024 * 1024  # 5 MB limit for log file size
backup_count = 3  # Keep up to 3 backup files

logger = logging.getLogger("NonsymmetricSamplerLogger")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Run the sampler with the logger
dag_size = len(jkl_scores)
num_samples = 1
sampled_dags = py_run_nonsymmetric_sampler(dag_size, num_samples, logger)
print("Sampled DAGs:", sampled_dags)


