from matrixclass import *


def strassen_gen_powerpad(A: Matrix, B: Matrix, case: int, cut: int =32 )->Matrix:
    '''Generalized version of strassen_mult which pads matrices to next power of 2 dimensions.

    Input: Matrix A and B to be multiplied, a cut value which determines the base case, and a case variable.
    Case 0: No optimization
    Case 1: Optimization using add_submatrix().
    Case 2: Optimization collapsing S into P and putting C directly into assign_submatrix()

    Output: A matrix of size A.num_of_rows x B.num_of_cols, which is the result of AB
    '''
    
    #Check that matrix multiplication can be performed
    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')

    # Take max of the three dimensions and find next biggest power of 2
    refval = max(A.num_of_rows, A.num_of_cols, B.num_of_cols)
    refval = 2**math.ceil(math.log(refval,2))


    #Create matrix of square power of 2 dimensions        
    Ap = Matrix([[0 for col in range(refval)] for row in range(refval) ])
    Bp = Matrix([[0 for col in range(refval)] for row in range(refval) ])

    #fill it in with old matrix        
    Ap.assign_submatrix(0,0,A)
    Bp.assign_submatrix(0,0,B)

    #return strassen algorithm for square matrices where dim is power of 2
    #Get the submatrix so we dont see extra 0's

    if case == 0:         #non optimized

        return strassen_matrix_mult(Ap, Bp, cut).submatrix(0, A.num_of_rows, 0, B.num_of_cols)

    elif case == 1:         #optmized case using add.submatrix()

        return strassen_matrix_mult2(Ap, Bp, cut).submatrix(0, A.num_of_rows, 0, B.num_of_cols)

    elif case==2:          #optimized case using collapse of definitions

        return strassen_matrix_mult3(Ap, Bp, cut).submatrix(0, A.num_of_rows, 0, B.num_of_cols)
    else:
        raise ValueError('Case is not available. Select 0, 1 or 2.')

   

def strassen_matrix_mult(A: Matrix, B: Matrix, cut: int = 32)-> Matrix:
    '''Strassen matrix multiplication for matrices of size 2^i x 2^i.
    Input: Matrices A and B to be multiplied and a cut value which determines the base case.
    Output: Resulting Matrix AB of size A.num_of_rows x B.num_of_cols
    '''

    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied.')


    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    
    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)

    # Build matrix of results
    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)

    # First batch of sums Theta(n^2)
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

    
    #Recursive Calls
    P1 = strassen_matrix_mult(A11,S1, cut)
    P2 = strassen_matrix_mult(S2, B22, cut)
    P3 = strassen_matrix_mult(S3,B11, cut)
    P4 = strassen_matrix_mult(A22,S4, cut)
    P5 = strassen_matrix_mult(S5,S6, cut)
    P6 = strassen_matrix_mult(S7,S8, cut)
    P7 = strassen_matrix_mult(S9,S10, cut)


    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    
    #Copyig Cij into the resulting matrix
    result.assign_submatrix(0,0, C11)
    result.assign_submatrix(0,result.num_of_cols//2, C12)
    result.assign_submatrix(result.num_of_rows//2, 0, C21)
    result.assign_submatrix(result.num_of_rows//2,result.num_of_cols//2, C22)
    return result


#optimized version    
def strassen_matrix_mult2(A: Matrix, B: Matrix, cut: int = 32)-> Matrix:
    '''Strassen matrix multiplication for matrices of size 2^i x 2^i using add_submatrix()
    optimization.

    Input: Matrices A and B to be multiplied and a cut value which determines the base case.
    Output: Resulting Matrix AB of size A.num_of_rows x B.num_of_cols
    '''

    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')

    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    # Recursive step

    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)

    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)

    P = strassen_matrix_mult2(A11, B12 - B22, cut) #P1
    result.add_submatrix(0,result.num_of_cols//2, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, P)
    
    P= strassen_matrix_mult2(A11 + A12, B22, cut) #P2
    result.add_submatrix(0,0, -1*P)
    result.add_submatrix(0,result.num_of_cols//2, P)#Anziche assign submatrix dovrei usare un add submatrix

    P = strassen_matrix_mult2(A21 + A22, B11, cut) #P3
    result.add_submatrix(result.num_of_rows//2, 0, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, -1*P)

    P = strassen_matrix_mult2(A22, B21 - B11, cut) #P4
    result.add_submatrix(0,0, P)
    result.add_submatrix(result.num_of_rows//2, 0, P)

    P = strassen_matrix_mult2(A11 + A22, B11 + B22, cut) #P5
    result.add_submatrix(0,0, P)
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2, P)

    result.add_submatrix(0,0,strassen_matrix_mult2(A12 - A22, B21 + B22, cut))
    result.add_submatrix(result.num_of_rows//2,result.num_of_cols//2,-1*strassen_matrix_mult2(A11 - A21, B11 + B12, cut))

    return result    


def strassen_matrix_mult3(A: Matrix, B: Matrix, cut: int = 32)-> Matrix:
    '''Strassen matrix multiplication for matrices of size 2^i x 2^i using optimization based
    on collapsing S into P and computing C directly in assign_submatrix().

    Input: Matrices A and B to be multiplied and a cut value which determines the base case.
    Output: Resulting Matrix AB of size A.num_of_rows x B.num_of_cols
    '''



    if A.num_of_cols != B.num_of_rows: 
        raise ValueError('The two matrices cannot be multiplied')

        
    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) <= cut:
        return gauss_matrix_mult(A,B)

    # Recursive step

    A11, A12, A21, A22 = get_matrix_quadrants(A)
    B11, B12, B21, B22 = get_matrix_quadrants(B)

    result = Matrix([[0 for x in range(B.num_of_cols)]
                     for y in range(A.num_of_rows)],
                    clone_matrix=False)

    P1 = strassen_matrix_mult3(A11, B12 - B22, cut)
    P2 = strassen_matrix_mult3(A11 + A12, B22, cut)
    P3 = strassen_matrix_mult3(A21 + A22, B11, cut)
    P4 = strassen_matrix_mult3(A22, B21 - B11, cut)
    P5 = strassen_matrix_mult3(A11 + A22, B11 + B22, cut)
   
    result.assign_submatrix(0,0, P5 + P4 - P2 + strassen_matrix_mult3(A12 - A22, B21 + B22, cut))
    result.assign_submatrix(0,result.num_of_cols//2, P1 + P2)
    result.assign_submatrix(result.num_of_rows//2, 0, P3 + P4)
    result.assign_submatrix(result.num_of_rows//2,result.num_of_cols//2, P5 + P1 - P3 - strassen_matrix_mult3(A11 - A21, B11 + B12, cut))

    return result