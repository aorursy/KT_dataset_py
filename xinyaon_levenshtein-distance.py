def visualize(x):
    for i in range(len(x)):
        print(x[i])
# edit a to b
matrix = []
a = "exponential"
b = "polynomial"

# initialize
for i in range(len(a)+1):
    matrix.append([-1]*(len(b)+1))

matrix[0][0] = 0

for i in range(1,len(a)+1):
    matrix[i][0] = matrix[i-1][0] + 1

for i in range(1, len(b)+1):
    matrix[0][i] = matrix[0][i-1] + 1

    
for i in range(1, len(a)+1):
    for j in range(1, len(b)+1):
        if(b[j-1] == a[i-1]):
            matrix[i][j] = min(matrix[i-1][j-1], matrix[i-1][j]+1, matrix[i][j-1]+1)
        else:
            matrix[i][j] = min(matrix[i-1][j-1]+1, matrix[i-1][j]+1, matrix[i][j-1]+1)

matrix
