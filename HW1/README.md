A solution to the homework assignment from the DSSC course in Advanced Programming and Algorithic Design (A.A 2020) by Prof. Casagrande.

1. Implement the strassen_matrix_mult function to multiply two $2^n \times 2^n$ matrices by using the Strassen's algorithm.
2. Generalize strassen_matrix_mult to deal with any kind of matrix pair that can be multiplied (possibly also non square matrices) and prove that the asymptotic complexity does not change. 
3. Improve the implementation of the function by reducing the number of auxiliary matrices and test the effects on execution time.
4. Answer to the following question: how much is the minimum auxiliary space required to evaluate the strassen's algorithm? Motivate.


The repository contains various implementations of strassen_matrix_mult which attempt to solve the problems listed above. In particular:
- strassengen_evenpad is able to multiply any non even matrices, by padding to the closest even dimension.
- strassengen_powpad which pads to the closest power of 2 dimension.
- matrix class which implements the concept of a matrix.


The solutions are extensively discussed in the IPYNB file called HW1. 
