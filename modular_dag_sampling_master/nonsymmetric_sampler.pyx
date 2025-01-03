from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
import os
# Import SubTable class and its methods
cdef extern from "subtable.h":
    cdef cppclass SubTable[T]:
        SubTable()
        SubTable(uint32_t n)
        # Omit the non-const version if not needed
        # T& operator()(uint32_t R, uint32_t U)
        T operator()(uint32_t R, uint32_t U) const

# Import function from C++ namespace
cdef extern from "nonsymmetric.h" namespace "nonsymmetric_":
    cdef vector[SubTable[double]] calculate_hat_weights(int size, vector[vector[double]]& weights)

# Python wrapper function
# Python wrapper function
def py_calculate_hat_weights(int size, list weights):
    cdef vector[vector[double]] cpp_weights
    cdef vector[double] w_vec

    # Convert Python list to C++ vector of vectors
    for w_list in weights:
        w_vec = vector[double]()
        for value in w_list:
            w_vec.push_back(<double>value)
        cpp_weights.push_back(w_vec)

    # Call the C++ function
    cdef vector[SubTable[double]] result = calculate_hat_weights(size, cpp_weights)

    # Convert result to Python-readable format
    py_result = []
    for subtable in result:
        table_data = []
        for R in range(1 << size):
            row_data = []
            for U in range(1 << size):
                value = subtable(R, U)  # No ambiguity now
                row_data.append(value)
            table_data.append(row_data)
        py_result.append(table_data)

    return py_result

# Import required C++ structures and functions
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t


from libcpp.string cimport string  # Import C++ string for conversion
import logging

# Correctly declare NonSymmetricSampler as an extern class from the header
cdef extern from "nonsymmetric.h" namespace "nonsymmetric_":
    cdef cppclass NonSymmetricSampler[T]:
        NonSymmetricSampler(const char* filename)  # Use const char* instead of std::string
        vector[int] sample()
import networkx as nx
import matplotlib.pyplot as plt



def write_dags(dags):
    """
    Write the DAGs to the console.
    """
    for dag in dags:
        size = len(dag)
        dag_strings = []
        for i in range(size):
            bitmask = dag[i]
            parents = []
            for j in range(size):
                if bitmask & (1 << j):
                    parents.append(j)
            parent_str = ', '.join(map(str, parents))
            dag_strings.append(f"{i} <- {{{parent_str}}}")
        print(', '.join(dag_strings))


# Main function to run the sampler with logging and file check
def py_run_nonsymmetric_sampler(int size, int num_dags, object logger):
    """
    Run the nonsymmetric sampler and return sampled DAGs with logging.
    """
    cdef NonSymmetricSampler[double]* sampler = NULL

    # Define file path
    file_path = "/home/gulce/Downloads/thesis/asia_scores.jkl"

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info("Initializing nonsymmetric sampler.")

        # Convert file path to UTF-8 encoded bytes first
        encoded_file_path = file_path.encode('utf-8')
        sampler = new NonSymmetricSampler[double](<const char*>encoded_file_path)

        py_dags = []
        for i in range(num_dags):
            logger.info(f"Sampling DAG {i + 1}/{num_dags}.")
            try:
                sampled_dag = sampler.sample()
                
            except Exception as e:
                logger.error(f"Error during sampling of DAG {i + 1}: {e}")
                continue

            # Convert the sampled DAG to Python list and store it
            py_dags.append([int(x) for x in sampled_dag])
            logger.info(f"Sampling DAG {i + 1} completed successfully.")
     

            # Visualize the DAG if the visualize flag is True
         

        write_dags(py_dags)

    except Exception as e:
        logger.error(f"An error occurred in py_run_nonsymmetric_sampler: {e}")
        raise

    finally:
        if sampler is not NULL:
            logger.info("Cleaning up sampler.")
            del sampler

    logger.info("Sampling completed successfully.")
    return py_dags
