from matrixclass import *

# The commented function below was used in the initial attempts to implement the algorithm. However, it was excluded
# from the final version to reduce function calls and make the code faster.
""" def pad(A: Matrix, B: Matrix):
    refval = [A.num_of_rows+A.num_of_rows%2, A.num_of_cols+A.num_of_cols%2, B.num_of_cols+B.num_of_cols%2]
    
    Ap = Matrix([[0 for col in range(refval[1])] for row in range(refval[0]) ])
    Bp = Matrix([[0 for col in range(refval[2])] for row in range(refval[1]) ])

    Ap.assign_submatrix(0,0,A)
    Bp.assign_submatrix(0,0,B)

    return Ap, Bp """


def strassen_gen_evenpad(A: Matrix, B: Matrix, cut: int = 32)->Matrix:
    '''Generalized version of strassen_mult which pads matrices to next even dimensions.
    No optimization.

    Input: Matrix A and B to be multiplied and a cut value which determines the base case.
    Output: A matrix of size A.num_of_rows x B.num_of_cols, which is the result of AB
    '''
    # Check that matrix multiplication can be performed
    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')

    Abp = A.num_of_rows
    Bbp = B.num_of_cols

    # Base Case
    if max(A.num_of_rows, A.num_of_cols, B.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    # Padding conditions
    if A.num_of_cols%2 != 0:
        refval = [A.num_of_rows+A.num_of_rows%2, A.num_of_cols+A.num_of_cols%2, B.num_of_cols+B.num_of_cols%2]
    
        Ap = Matrix([[0 for col in range(refval[1])] for row in range(refval[0]) ])
        Bp = Matrix([[0 for col in range(refval[2])] for row in range(refval[1]) ])

        Ap.assign_submatrix(0,0,A)
        Bp.assign_submatrix(0,0,B)
        A = Ap
        B = Bp
    
        
    else:
        if A.num_of_rows%2 != 0:
            refval = A.num_of_rows+A.num_of_rows%2
            Ap = Matrix([[0 for col in range(A.num_of_cols)] for row in range(refval) ])
            Ap.assign_submatrix(0,0,A)
            A = Ap


        if B.num_of_cols%2 != 0:
            refval = B.num_of_cols+B.num_of_cols%2
            Bp = Matrix([[0 for col in range(refval)] for row in range(B.num_of_rows) ])
            Bp.assign_submatrix(0,0,B)
            B = Bp

    # Subquadrants
    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)

    # Result Matrix
    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)
   
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10= B11 + B12
    
    # Recursive Steps
    P1 = strassen_gen_evenpad(A11, S1, cut)
    P2 = strassen_gen_evenpad(S2, B22, cut)
    P3 = strassen_gen_evenpad(S3, B11, cut)
    P4 = strassen_gen_evenpad(A22, S4, cut)
    P5 = strassen_gen_evenpad(S5, S6, cut)
    P6 = strassen_gen_evenpad(S7, S8, cut)
    P7 = strassen_gen_evenpad(S9, S10, cut)
   
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    result.assign_submatrix(0,0, C11)
    result.assign_submatrix(0,result.num_of_cols//2, C12)
    result.assign_submatrix(result.num_of_rows//2, 0, C21)
    result.assign_submatrix(result.num_of_rows//2,result.num_of_cols//2, C22)

    return  result.submatrix(0, Abp, 0, Bbp)
 

def strassen_optimized_evenpad(A: Matrix, B: Matrix, cut: int = 32)->Matrix:
    '''Generalized version of strassen_mult which pads matrices to next even dimensions.
    Using add_submatrix() method to optimize.

    Input: Matrix A and B to be multiplied and a cut value which determines the base case.
    Output: A matrix of size A.num_of_rows x B.num_of_cols, which is the result of AB
    '''
     
    #Check that matrix multiplication can be performed
    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')
    Abp = A.num_of_rows
    Bbp = B.num_of_cols

    # Base Case
    if max(A.num_of_rows, A.num_of_cols, B.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    # Padding
    if A.num_of_cols%2 != 0:
        refval = [A.num_of_rows+A.num_of_rows%2, A.num_of_cols+A.num_of_cols%2, B.num_of_cols+B.num_of_cols%2]
    
        Ap = Matrix([[0 for col in range(refval[1])] for row in range(refval[0]) ])
        Bp = Matrix([[0 for col in range(refval[2])] for row in range(refval[1]) ])

        Ap.assign_submatrix(0,0,A)
        Bp.assign_submatrix(0,0,B)
        A = Ap
        B = Bp
    
        
    else:
        if A.num_of_rows%2 != 0:
            refval = A.num_of_rows+A.num_of_rows%2
            Ap = Matrix([[0 for col in range(A.num_of_cols)] for row in range(refval) ])
            Ap.assign_submatrix(0,0,A)
            A = Ap


        if B.num_of_cols%2 != 0:
            refval = B.num_of_cols+B.num_of_cols%2
            Bp = Matrix([[0 for col in range(refval)] for row in range(B.num_of_rows) ])
            Bp.assign_submatrix(0,0,B)
            B = Bp

    # Subquadrants
    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)

    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)
   
    
    #Recursive steps. 
    P = strassen_optimized_evenpad(A11, B12 - B22, cut) #P1
    result.add_submatrix(0,result.num_of_cols//2, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, P)
    
    P= strassen_optimized_evenpad(A11 + A12, B22, cut) #P2
    result.add_submatrix(0,0, -1*P)
    result.add_submatrix(0,result.num_of_cols//2, P)

    P = strassen_optimized_evenpad(A21 + A22, B11, cut) #P3
    result.add_submatrix(result.num_of_rows//2, 0, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, -1*P)

    P = strassen_optimized_evenpad(A22, B21 - B11, cut) #P4
    result.add_submatrix(0,0, P)
    result.add_submatrix(result.num_of_rows//2, 0, P)

    P = strassen_optimized_evenpad(A11 + A22, B11 + B22, cut) #P5
    result.add_submatrix(0,0, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, P)

    # Since P6 and P7 are used only once in the definition of C, they are computed directly inside the method. 
    result.add_submatrix(0,0,strassen_optimized_evenpad(A12 - A22, B21 + B22, cut))
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2,-1*strassen_optimized_evenpad(A11 - A21, B11 + B12, cut))
    

    return result.submatrix(0, Abp, 0, Bbp)

def strassen_optimized2_evenpad(A: Matrix, B: Matrix, cut: int = 32)->Matrix:
    '''Generalized version of strassen_mult which pads matrices to next even dimensions. To optimize,
    we collapse the definition of S into P and put the definition of C inside assign_submatrix().

    Input: Matrix A and B to be multiplied and a cut value which determines the base case.
    Output: A matrix of size A.num_of_rows x B.num_of_cols, which is the result of AB
    '''
    
    #Check that matrix multiplication can be performed
    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')
    Abp = A.num_of_rows
    Bbp = B.num_of_cols
    
    # Base Case
    if max(A.num_of_rows, A.num_of_cols, B.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    # Padding
    if A.num_of_cols%2 != 0:
        refval = [A.num_of_rows+A.num_of_rows%2, A.num_of_cols+A.num_of_cols%2, B.num_of_cols+B.num_of_cols%2]
    
        Ap = Matrix([[0 for col in range(refval[1])] for row in range(refval[0]) ])
        Bp = Matrix([[0 for col in range(refval[2])] for row in range(refval[1]) ])

        Ap.assign_submatrix(0,0,A)
        Bp.assign_submatrix(0,0,B)
        A = Ap
        B = Bp
    
        
    else:
        if A.num_of_rows%2 != 0:
            refval = A.num_of_rows+A.num_of_rows%2
            Ap = Matrix([[0 for col in range(A.num_of_cols)] for row in range(refval) ])
            Ap.assign_submatrix(0,0,A)
            A = Ap


        if B.num_of_cols%2 != 0:
            refval = B.num_of_cols+B.num_of_cols%2
            Bp = Matrix([[0 for col in range(refval)] for row in range(B.num_of_rows) ])
            Bp.assign_submatrix(0,0,B)
            B = Bp

    # Subquadrants
    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)


    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)
   
    # Recursive Steps
    P1 = strassen_gen_evenpad(A11, B12 - B22, cut)
    P2 = strassen_gen_evenpad(A11 + A12, B22, cut)
    P3 = strassen_gen_evenpad(A21 + A22, B11, cut)
    P4 = strassen_gen_evenpad(A22, B21 - B11, cut)
    P5 = strassen_gen_evenpad(A11 + A22, B11 + B22, cut)
   
    result.assign_submatrix(0,0, P5 + P4 - P2 + strassen_gen_evenpad(A12 - A22, B21 + B22, cut))
    result.assign_submatrix(0,result.num_of_cols//2, P1 + P2)
    result.assign_submatrix(result.num_of_rows//2, 0, P3 + P4)
    result.assign_submatrix(result.num_of_rows//2,result.num_of_cols//2, P5 + P1 - P3 - strassen_gen_evenpad(A11 - A21, B11 + B12, cut))

    return  result.submatrix(0, Abp, 0, Bbp)