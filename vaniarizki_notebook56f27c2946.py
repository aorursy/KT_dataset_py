from learntools.core import binder; binder.bind(globals())

from learntools.python.ex3 import *

import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

print('Setup complete.')



mat = np.array([[0,1,3,4,5], [1,0,2,3,4], [3,2,0,1,2], [4,3,1,0,1], [5,4,2,1,0]])

dists = squareform(mat)

linkage_matrix = linkage(dists, "complete")

dendrogram(linkage_matrix, labels=["A","B","C","D","E"])

plt.title("Complete Link")

plt.show()





linkage_matrix2 = linkage(dists, "average")

dendrogram(linkage_matrix2, labels=["A","B","C","D","E"])

plt.title("Average")

plt.show()



linkage_matrix3 = linkage(dists, "single")

dendrogram(linkage_matrix3, labels=["A","B","C","D","E"])

plt.title("Single link")

plt.show()