# Let us import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Let us set up the files
dataset = "../input/DS_BEZDEKIRIS_STD.data"
# Let us read the data from the file and see the first five rows of the data
data = pd.read_csv(dataset, header = None)
data.head()
def irisLabel(s):
    s = s.lower()
    if s == "iris-setosa":
        return 0
    if s == "iris-versicolor":
        return 1
    if s == "iris-virginica":
        return 2
    return 3 #covering all cases
iris_data = pd.read_csv(dataset,header=None, converters={4:irisLabel})
# Now we display the first five rows of the data
iris_data.head()
iris_data.iloc[:,0:4]
#iris_data.iloc[:]
# Now we will seperate the features and labels and store them in features and lables variables.
Features = iris_data.iloc[:,0:4].values
Labels = iris_data.iloc[:,4:5].values
# Now we will print the features 
Features
# Now we will print the labels
Labels
## Now store Feature matrix as X
X = Features
## We are creating a matrix 'A' as considered above.
A = np.array([[1,0,0,0],[0,1,0,0]])
print(A)
X_transpose = np.transpose(X)
print(X_transpose)
X1 = A @ X_transpose ## @ - symbol used for matrix multiplication
print(X1.shape)
## Converting the matrix X1 back to 150*2 and append label creating X1_final of 150*3
X1 = np.transpose(X1)
X1_final = np.hstack((X1,Labels))
print(X1_final)
## Now store Feature matrix as X
X = Features

## Create matrix A as given in the Exercise above
A = np.array([[1,0,0,0],[0,0,0,1]])

## As A is 2*4 matrix. To compute X' convert X into 4*150 using transpose function. 
##As for matrix multiplication dimensions should match
X_transpose = np.transpose(X)

## Compute X' same as X1. Only the name is changed. X1 is now 2*150 dimensional.
## Meaning we have extracted 2 features out of 4
X1 = A @ X_transpose

## Convert the matrix X1 back to 150*2 and append label creating X1_final of 150*3
X1 = np.transpose(X1)
X1_final = np.hstack((X1,Labels))
#Plot the points on graph and visualize.
plt.figure(1, figsize=(5,5))
plt.scatter(X1_final[:,0],X1_final[:,1],c=X1_final[:,2],s=60)
plt.show()
### Your Code Here

B = np.array([[0,1,0,0],[0,0,1,0]])
#print(B)

X_transpose = np.transpose(X)
#print(X_transpose)

X1 = B @ X_transpose ## @ - symbol used for matrix multiplication
#print(X1.shape)

X1 = np.transpose(X1)
X1_final = np.hstack((X1,Labels))
#print(X1_final)


#Plot the points on graph and visualize.
plt.figure(1, figsize=(5,5))
plt.scatter(X1_final[:,0],X1_final[:,1],c=X1_final[:,2],s=60)
plt.show()

### Your Code Here

### Your Code Here

c = np.array([[0,0,1,0],[0,0,0,1]])
#print(B)

X_transpose = np.transpose(X)
#print(X_transpose)

X1 = c @ X_transpose ## @ - symbol used for matrix multiplication
#print(X1.shape)

X1 = np.transpose(X1)
X1_final = np.hstack((X1,Labels))
#print(X1_final)


#Plot the points on graph and visualize.
plt.figure(1, figsize=(5,5))
plt.scatter(X1_final[:,0],X1_final[:,1],c=X1_final[:,2],s=60)
plt.show()
