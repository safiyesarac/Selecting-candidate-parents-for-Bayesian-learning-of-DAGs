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
