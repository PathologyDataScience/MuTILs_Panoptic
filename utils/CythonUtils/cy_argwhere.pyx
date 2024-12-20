# cython: language_level=3

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

# Define the function with type annotations for better performance
def argwhere_1d(list[int] arr):
    cdef int i
    cdef int count = 0
    cdef int arr_len = len(arr)
    cdef int* result_temp = <int*>malloc(arr_len * sizeof(int))

    # Check each element and if it meets the condition, add its index to result_temp
    for i in range(arr_len):
        if arr[i] != 0:
            result_temp[count] = i
            count += 1

    # Copy the relevant part of the result_temp to a Python list
    result = [result_temp[i] for i in range(count)]

    # Free the allocated memory
    free(result_temp)

    return result

def argwhere_2d(list list_of_lists):
    cdef int i, j
    cdef int rows = len(list_of_lists)
    cdef int cols = len(list_of_lists[0]) if rows > 0 else 0
    cdef int count = 0
    # Assuming the number of non-zero elements will not exceed rows * cols
    cdef int* temp_rows = <int*>malloc(rows * cols * sizeof(int))
    cdef int* temp_cols = <int*>malloc(rows * cols * sizeof(int))

    # Iterate over each element in the 2D list
    for i in range(rows):
        for j in range(len(list_of_lists[i])):
            if list_of_lists[i][j] != 0:
                temp_rows[count] = i
                temp_cols[count] = j
                count += 1

    # Prepare the result list
    result = [(temp_rows[i], temp_cols[i]) for i in range(count)]

    # Free the allocated memory
    free(temp_rows)
    free(temp_cols)

def cy_argwhere2d(cnp.ndarray[cnp.npy_bool, ndim=2] arr):
    cdef int nrows = arr.shape[0]
    cdef int ncols = arr.shape[1]
    cdef list indices = []

    for i in range(nrows):
        for j in range(ncols):
            if arr[i, j]:  # This condition works for boolean arrays; True values are considered "non-zero".
                indices.append([i, j])

    return np.array(indices, dtype=np.intp).reshape(-1, 2) if indices else np.empty((0, 2), dtype=np.intp)