from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

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


