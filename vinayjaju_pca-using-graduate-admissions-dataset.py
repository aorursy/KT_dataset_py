# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

# reading dataset

df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
display(df.head())

df.info()

display(df.describe())
#dividing into features and labels

features=df.iloc[:,1:-1]

labels=df.iloc[:,-1]

display(features[:5])

display(labels[:5])
new_labels=pd.cut(np.array(labels),3, labels=["bad", "medium", "good"])

print(new_labels.shape)

new_labels[:5]
from sklearn.preprocessing import StandardScaler
# Normalizing the data

standardized_data=StandardScaler().fit_transform(features)

standardized_data[:5]
print('NumPy covariance matrix: \n%s' %np.cov(standardized_data.T))

cov_mat = np.cov(standardized_data.T)
#Calculating the eigen values and vectors

eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 

                      eig_pairs[1][1].reshape(7,1)))



print('Matrix W:\n', matrix_w)

Y = standardized_data.dot(matrix_w)

Y.shape
pca_data=np.vstack((Y.T,new_labels)).T

pca_df=pd.DataFrame(data=pca_data,columns=("1st Component","2nd Component","Chances of getting in?"))

pca_df.head()
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(12, 8)

sns.scatterplot(x="1st Component", y="2nd Component", hue="Chances of getting in?", data=pca_df);
from sklearn import decomposition

pca=decomposition.PCA()
pca.n_components=2

pca_data=pca.fit_transform(standardized_data)

print("The reduced shape is", pca_data.shape)
pca_data[:5]
pca_data=np.vstack((pca_data.T,new_labels)).T

pca_df=pd.DataFrame(data=pca_data,columns=("1st Component","2nd Component","Chances of getting in?"))

pca_df.head()
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(12, 8)

sns.scatterplot(x="1st Component", y="2nd Component", hue="Chances of getting in?", data=pca_df);