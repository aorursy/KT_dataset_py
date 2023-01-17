################################################################################
# PCA was completed in my earlier kernel (See EDA (some PCA))
# The input file contains PC dimensions of Family History and Insurance History vars upto 3 dimensions
# In this kernel we will run a gaussian mixture model for clustering of the 
# Family history principal components
#################################################################################

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Already contains the 3D-principal components for insurance and family history vars - see my earlier kernel 
pcfull = pd.read_csv('../input/impData4.csv')
pccomp = pcfull[['FHPC1','FHPC2','FHPC3']]

# This initializes the centroids of the 3 clear clusters (see earlier kernel)
# After eyeballing where the centroids might be
# Eyeballing was done using rgl for R 3.4.3, which can't be used in R 3.4.2 kaggle kernels!! :(
m1 = np.array([[0, -3, 0],[0, -1.5, 0],[0, 0.5 ,0]])

# using GaussianMixture from scikitlearn v0.19. 
# Key to successful clustering is to specify shared covariance matrix option [Not available in R!]
# and some initial weights / responsibilities of the hidden distributions, and the centroid means as above
# Kind of 'helping' the EM algorithm attaing 'proper' convergence
gmm = GaussianMixture(n_components=3,covariance_type='tied',weights_init=[0.01,0.45,0.54],means_init=m1,tol=0.0000001,verbose=1,max_iter=1000,n_init=20).fit(pccomp)



# The posterior responsibilities / weights
print ('The posterior weight distributions:')
print(gmm.weights_)

# The posterior means of 3 latent distributions
print ('The posterior means of each Gaussian model in the mixture')
print(gmm.means_)

#The posterior covariance matrix (shared, in this case)
print ('The posterior shared covariance matrix')
print(gmm.covariances_)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Directly predict response from data
pred = gmm.predict(pccomp[['FHPC1','FHPC2','FHPC3']])
pcpred = np.array(pred)
ax.scatter(pccomp[['FHPC2']],pccomp[['FHPC3']],pccomp[['FHPC1']],c=pcpred)
ax.set_xlabel('Family History PC 2')
ax.set_ylabel('Family History PC 3')
ax.set_zlabel('Family History PC 1')

plt.rcParams["figure.figsize"][0] = 200
plt.rcParams["figure.figsize"][1] = 200
plt.show()