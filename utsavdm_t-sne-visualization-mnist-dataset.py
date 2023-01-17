# standard import statements

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# importing the 'train.csv' dataset into the dataframe 'df':

df = pd.read_csv('../input/train.csv')



# separating the predictor variable 'label' from its 784 features:

label = df.iloc[:,0:1]

dataset = df.drop('label', axis=1)



print("Shape of the label:", label.shape)

print("Shape of the dataset:", dataset.shape)



# feature scaling - standardizing the input data:

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

std_dataset = std_scaler.fit_transform(dataset)

del dataset

print("Shape of the standardized dataset: ", std_dataset.shape)



# getting TSNE from sklearn.manifold

from sklearn.manifold import TSNE



# below are the values of the parameteres we are using here:

# n_components = 2 (since we are trying to reduce the dimensions from 784-D to 2-D)

# perplexity = 50 (value should be higher since the dataset is huge in this case)

# n_iter = 1500 (no. of steps/ iterations that we want TSNE to go through)

# learning_rate = 200 (default value)

# we should normally run TSNE algorithms multiple times to see the stability of the plot that we get.

# the more the plot remains unchanged, the more is its reliability.

# But, this algorithm takes too long to run (since the input is of shape 42000 x 784 & size ~73MB), 

# it may even throw MEMORY ERROR on standard systems that have 4GB of RAM.

model = TSNE(n_components=2, perplexity=50, n_iter=1500, random_state=0)



# applying the t-SNE algorithm on the dataset (actually performing the dimensionality reduction)

# NOTE: This line takes 1 hour or more to run!!!

tsne_dataset = model.fit_transform(std_dataset)

print("Shape of the reduced t_sne dataset: ", tsne_dataset.shape)



# appending the 'label' column to the tsne_dataset:

tsne_dataset = np.hstack((tsne_dataset,label))



# creating the dataframe out of the tsne_dataset for plotting purpose:

final_dataset = pd.DataFrame(tsne_dataset, columns=('Dimension 1', 'Dimension 2', 'Label'))



# plotting the final t-SNE plot using seaborn:

import seaborn as sn

sn.FacetGrid(final_dataset, hue='Label', height = 8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.show()