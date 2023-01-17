# standard import statements:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# importing the 'train.csv' dataset into the dataframe 'df':

df = pd.read_csv('../input/input-train/train.csv')



# separating the predictor variable 'label' from its 784 features 'dataset':

label = df.iloc[:,0:1]

dataset = df.drop('label', axis=1)



# just getting the shapes of all the datasets we created:

print("Shape of the dataset: ", dataset.shape)

print("Shape of the label: ", label.shape)



# feature scaling - standardizing the dataset:

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

std_dataset = std_scaler.fit_transform(dataset)



# Shape of the standardized dataset:

print("Shape of the standardized dataset: ", std_dataset.shape)



# creating the co-variance matrix 'covar_matrix':

covar_matrix = np.matmul(std_dataset.T, std_dataset)

print("Shape of the covariance matrix: ", covar_matrix.shape)



# calculating the eigne values and eigen vectors:

# Here, since we are trying to reduce the dimensions from 784 to 2, we'll only need top 2 eigen values & vectors

# Also, 'eigh' function returns the eigen values & vectors in the ascending order starting from 0. 

# Since we need the top 2 values, we'll hardcode 'eigvals' as 782 & 783

# Please read the documentation for in-depth details:



from scipy.linalg import eigh



eig_values, eig_vectors = eigh(covar_matrix, eigvals=(782,783))



print("Eigen values: ", eig_values)

print("Shape of eigen vectors: ", eig_vectors.shape)



# mutiplying the 'std_dataset' with the 'eig_vectors', we'll get the dataset that is reduced to 2 dimensions.

reduced_dataset = np.matmul(std_dataset, eig_vectors)

print("Shape of the reduced_dataset: ", reduced_dataset.shape)



# appending the 'label' column to the 'reduced_dataset', we get the 'pca_dataset' which is a numpy array:

pca_dataset = np.hstack((reduced_dataset, label))



# creating the pandas dataframe out of the above obtained 'pca_dataset':

final_dataset = pd.DataFrame(pca_dataset, columns=('1st PC', '2nd PC', 'Label'))

print("Shape of the final pca_dataset:", final_dataset.shape)

print(final_dataset.head(5))



# plotting using seaborn

import seaborn as sn

sn.FacetGrid(final_dataset, hue='Label', height=8).map(plt.scatter, '1st PC', '2nd PC').add_legend()

plt.show()
# standard import statements:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# importing the 'train.csv' dataset into the dataframe 'df':

df = pd.read_csv('../input/input-train/train.csv')



# separating the predictor variable 'label' from its 784 features 'dataset':

label = df.iloc[:,0:1]

dataset = df.drop('label', axis=1)



# perform feature scaling (standardizing the input dataset):

from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler()

std_dataset = s_scaler.fit_transform(dataset)



# getting PCA from decomposition from sklearn:

from sklearn import decomposition

pca = decomposition.PCA()



# n_components is the no. of dimensions in which we want to reduce the original dataset.

# in this case, we are trying to reduce 784 dimensions to 2 dimensions for visualization:

pca.n_components = 2



# applying the PCA algorithm to the given dataset (actually performing dimensionality reduction on the 784 dimensions):

pca_dataset = pca.fit_transform(std_dataset)



# appending the 'label' column to the obtained dataset (used in plotting):

# hstack - horizontal stack

pca_dataset = np.hstack((pca_dataset, label))



# creating the pandas dataframe 'final_dataset' out of the newly reduced dataset 'pca_dataset':

final_dataset = pd.DataFrame(pca_dataset, columns=("1st PC", "2nd PC", "Label"))

print("Shape of the FINAL pca_dataset: ", final_dataset.shape)

print(final_dataset.head(5))



# plotting the dataframe using seaborn:

import seaborn as sn

sn.FacetGrid(final_dataset, hue='Label', height=8).map(plt.scatter, '1st PC', '2nd PC').add_legend()

plt.show()