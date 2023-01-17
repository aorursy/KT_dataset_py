import numpy as np # linear algebra



# determinant of a 3 × 3 matrix

def det3(m):

    a = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])

    b = m[0][1]*(m[0][0]*m[2][2]-m[0][2]*m[2][0])

    c = m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

    return a + (-1)*b + c



# determinant of a 2 × 2 matrix

def det2(m):

    return (m[0][0]*m[1][1] - m[0][1]*m[1][0])



# transpose matrix

def transpose(m):

    mT = np.zeros((3, 3))

    n = len(m)

    for i in range(n):

        for j in range(n):

             mT[i][j] = m[j][i]

    return mT

# inverse matrix

def inverse(m):

    n = len(m)

    array = []

    for i in range(n):

        print("inverse matrix: ", m[i,:])

        array.append (m[i,:])

    print("array: ", array)



matrix = np.random.rand(3, 3)

matrix_1 = np.random.rand(2, 2)

print("matrix 3 x 3: ",matrix)

print("matrix 2 x 2: ",matrix_1)

print("det3: ",det3(matrix))

print("det2: ",det2(matrix_1))

print("transpose: ",transpose(matrix))

inverse(matrix)


