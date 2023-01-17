# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
def dist(pt1, pt2, dist_type):
    """calculate distance between 2 pts, capable of handing eculidean and manhantan distance"""

    if  pt1 is None or pt2 is None:
        return None

    if len(pt1) != len(pt2):
        return None

    distance = 0
    if dist_type.lower() == 'euclidean':
        distance = math.sqrt( sum((pt1[i] - pt2[i])**2 for i in range(len(pt1))) )

    elif dist_type.lower() == 'manhantan':
        distance = sum(pt1[i] - pt2[i] for i in range(len(pt1)))

    else:
        raise ValueError('undefined distance type!')

    return distance

def get_centroid(cluster):
    """cluster is a list of pts [[x1,y1,..], [x2,y2,...],...], get centroids of a given cluster"""
    if  cluster is None or len(cluster) == 0:
        raise ValueError('Empty Cluster')

    centroid = np.mean(cluster,axis=0)
    return centroid


def get_cluster(data,label,k):
    """A VERY LAZY AND SLOW WAY TO GET CLUSTERS"""
    return [data[n,:] for n in range(len(data)) if label[n] == k]

def kmeans(data, k, threshold=0.01, max_iter=300):
    """The core K Means"""
    n_pts = len(data)
    label = np.random.randint(k,size=n_pts)
    min_delta = int(n_pts*threshold)

    n_iter = 0
    dist_mat = np.zeros([n_pts, k])
    while n_iter < max_iter:
        print('Iteration:',n_iter)
        delta = 0
        for i in range(k):
            cluster_i = get_cluster(data,label,i)
            centroid_i = get_centroid(cluster_i)
            #print(centroid_i)
            for n in range(n_pts):
                dist_mat[n][i] = dist(data[n,:],centroid_i,'euclidean')

        for j in range(n_pts):
            tmp = label[j]

            label[j] = np.argmin(dist_mat[j] )
            if tmp != label[j] :
                delta += 1

        if delta < min_delta:
            break

        n_iter += 1


    return label

if __name__ == '__main__':
    #Axes3D is needed for display 3d figure
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt


    np.random.seed(1)
    N = 500
    n_dims = 2
    test_data = np.random.rand(N,n_dims)

    k = 5
    results = kmeans(test_data,k)
    if n_dims == 2:
        plt.scatter(test_data[:,0],test_data[:,1], c=results)
        plt.show()
    elif n_dims == 3:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=results)
        plt.show()
    else:
        pass