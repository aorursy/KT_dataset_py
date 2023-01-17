# numpy.linalg.eig module takes in a square matrix as the input and returns eigen values and eigen vectors.

# It also raises an LinAlgError if the eigenvalue computation does not converge.



import numpy as np

from numpy import linalg as LA



input = np.array([[2,-1],[4,3]])



w, v = LA.eig(input)