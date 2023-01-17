# Loading the libraries needed



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy.linalg import eigh 

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading MNIST Dataset

df = pd.read_csv("/kaggle/input/MNIST_train.csv")
# shape of the dataset

df.shape
# Storing labels separately

label = df['label']
# Removing label from the dataset and storing only actual data

data = df.drop("label",axis=1)
data.shape
# Visualizing first 5 rows of the dataset

data.head()
print(data.shape)

print(label.shape)
# display or plot a number.

plt.figure(figsize=(7,7))

idx = 100



grid_data = data.iloc[idx].values.reshape(28,28)  # reshape from 1d to 2d pixel array

plt.imshow(grid_data, interpolation = "none", cmap = "gray")

plt.show()



print(label[idx])
label_sample = label #.head(1000)

data_sample = data #.head(1000)



print("the shape of sample data = ", data_sample.shape)
# Data-preprocessing: Standardizing the data



data_std = StandardScaler().fit_transform(data_sample)

print(data_std.shape)
#find the co-variance matrix which is : (1/n) * A^T * A



# matrix multiplication using numpy



covar_matrix = np.matmul(data_std.T , data_std)/data_std.shape[0]



print ( "The shape of variance matrix = ", covar_matrix.shape)
covar_matrix
# finding the top two eigen-values and corresponding eigen-vectors 

# for projecting onto a 2-Dim space.



# the parameter 'eigvals' is defined (low value to heigh value) 

# eigh function will return the eigen values in asending order

# this code generates only the top 2 (782 and 783) eigenvalues.

values, vectors = eigh(covar_matrix, eigvals=(782,783))



print("Shape of eigen vectors = ",vectors.shape)

# converting the eigen vectors into (2,d) shape for easyness of further computations

vectors = vectors.T



print("Updated shape of eigen vectors = ",vectors.shape)

# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector

# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
#swapping the rows of eigen vectors since the largest value is in second place

final_vectors = np.vstack((vectors[1],vectors[0]))
final_vectors.shape
# projecting the original data sample on the plane 

#formed by two principal eigen vectors by vector-vector multiplication.



new_coordinates = np.matmul(final_vectors, data_std.T)



print (" resultanat new data points' shape ", final_vectors.shape, "X", data_std.T.shape," = ", new_coordinates.shape)
# appending label to the 2d projected data

new_coordinates = np.vstack((new_coordinates, label_sample)).T



# creating a new data frame for ploting the labeled points.

dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

print(dataframe.head())
# ploting the 2d data points with seaborn



sns.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
# initializing the pca

from sklearn import decomposition

pca = decomposition.PCA()
# configuring the parameteres

# the number of components = 2

pca.n_components = 2

pca_data = pca.fit_transform(data_sample)



# pca_reduced will contain the 2-d projects of simple data

print("shape of pca_reduced.shape = ", pca_data.shape)



# attaching the label for each 2-d data point 

pca_data = np.vstack((pca_data.T, label_sample)).T
# creating a new data fram which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))

sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
# PCA for dimensionality redcution (non-visualization)



pca.n_components = 784

pca_data = pca.fit_transform(data_std)



percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);



cum_var_explained = np.cumsum(percentage_var_explained)



# Plot the PCA spectrum

plt.figure(1, figsize=(6, 4))



plt.clf()

plt.plot(cum_var_explained, linewidth=2)

plt.axis('tight')

plt.grid()

plt.xlabel('n_components')

plt.ylabel('Cumulative_explained_variance')

plt.show()





# If we take 200-dimensions, approx. 90% of variance is expalined.