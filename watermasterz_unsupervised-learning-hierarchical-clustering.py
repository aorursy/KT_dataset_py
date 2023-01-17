import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt
dir = "../input/mall-customers/Mall_Customers.csv"

data = pd.read_csv(dir)

print(data.head(),"\n\n")

data['gender'] = [1 if g=='Male' else 0 for g in data['Genre']]

print(data.head())
plt.figure(figsize=(6,3))

plt.title("Annual Income plot")

plt.xlabel("Spending Score (1-100)")

plt.ylabel("Annual Income (k$)")

plt.scatter(data["Annual Income (k$)"], data["Spending Score (1-100)"]);





plt.figure(figsize=(6,3))

plt.title("Gender plot")

plt.scatter(data["gender"], data["Spending Score (1-100)"])

plt.xlabel("gender")

plt.ylabel("Spending Score (1-100)")



plt.figure(figsize=(6,3))

plt.title("Age plot")

plt.scatter(data["Age"], data["Spending Score (1-100)"])

plt.xlabel("Age")

plt.ylabel("Spending Score (1-100)");



plt.figure(figsize=(6,3))

plt.title("Age vs Income plot")

plt.scatter(data["Age"], data["Annual Income (k$)"])

plt.xlabel("Age")

plt.ylabel("Annual Income (k$)");
def convert_to_points(data1, data2):

    points = []

    maxx = max(data1)

    maxy = max(data2)

    for i in data.index:

        # Scaling down the points by dividing them by max

        points.append( np.array((data1[i] / maxx, data2[i]/ maxy)))



    points = np.array(points)

    return points

points = convert_to_points(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.figure(figsize=(6,3))

plt.title("Scaled Down data")

plt.scatter(points[:, 0], points[:, 1])

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")
points.shape
def edist(a, b):

    return np.linalg.norm(a-b)

    # euclidean distance

    # l2 norm

    

def mdist(a, b):

    return np.linalg.norm(a-b, 1)

    # manhatten distance

    # l1 norm
# return the minimum distance between the clusters

def single_linkage(cA, cB, dist_type):

    

    min_dist= float('inf')

    

    for i in cA:

        for j in cB:

            if dist_type=='e':

                dist = edist(i, j)

            elif dist_type=='m':

                dist = mdist(i, j)

                

            if dist <= min_dist:

                min_i = i

                min_j = j

                min_dist = dist

                

    return min_dist
# return average distance between the clusters

def avg_linkage(cA, cB, dist_type):

    

    avg = 0

    count = 0

    for i in cA:

        for j in cB:

            if dist_type=='e':

                dist = edist(i, j)

            elif dist_type=='m':

                dist = mdist(i, j)

            

            avg += dist

            count += 1

                

    return avg/count
# return max distance between the clusters

def complete_linkage(cA, cB, dist_type):

    

    max_dist= float('-inf')

    

    for i in cA:

        for j in cB:

            if dist_type=='e':

                dist = edist(i, j)

            elif dist_type=='m':

                dist = mdist(i, j)

                

            if dist >= max_dist:

                max_i = i

                max_j = j

                max_dist = dist

                

    return max_dist
def centroid_linkage(cA, cB, dist_type):

    

    # centroid of points in cartesian plane =  mean of the x and y coordinates respectively

    a = np.mean(cA, axis=-2)

    b = np.mean(cB, axis=-2)

    

    if dist_type=='e':

        dist = edist(a, b)

    elif dist_type=='m':

        dist = mdist(a, b)

        

    return dist

def min_dist(cluster, linkage, dist_type):

    matrix = np.empty((len(cluster), len(cluster)))

    

    for i, c1 in enumerate(cluster):

        for j, c2 in enumerate(cluster):

            #print(c1)

            #print(c2)

            

            if linkage=='single':

                dist = single_linkage(c1, c2, dist_type)

            elif linkage=='average':

                dist = avg_linkage(c1, c2, dist_type)

            elif linkage=='complete':

                dist = complete_linkage(c1, c2, dist_type)

            elif linkage=='cent':

                dist = centroid_linkage(c1, c2, dist_type)

            else:

                print("Not a valid linkage. Exiting....")

                exit()

                

            matrix[i][j] = dist

            if i == j:

                matrix[i][j]=float(1e4)

    

    return matrix
def plot(c, xlab='x', ylab='y'):

    x = np.zeros(len(c), dtype=list)

    y = np.zeros(len(c), dtype=list)

    plt.xlabel(xlab)

    plt.ylabel(ylab)

    for n, i in enumerate(c):

        x[n]=list()

        y[n]=list()

        for j in i:

            #print(j)

            x[n].append(j[0])

            y[n].append(j[1])

    for i in range(len(c)):

        plt.scatter(x[i], y[i], label=f"Cluster {i+1}")

    

    plt.legend()

def Get_Clusters(points, N=1, linkage='average', dist_type='e', cut_off=2.7,verbose=1):

    

    # Initialize a cluster based on the input points

    old_min = float("inf")

    cluster = []

    for i in points:

        cluster.append([i])

    print(f"Cluster shape: {np.array(cluster).shape}\n(no of clusters, cluster size, coordinate)\n")

    

    len_segment = int(len(points)* 0.1)+1

    # can specify the number of clusters desired

    

    while len(cluster) !=N:

        sample_mat = min_dist(cluster, linkage, dist_type)

        done = False

        for i in range(sample_mat.shape[0]):

            for j in range(sample_mat.shape[1]):

                if sample_mat[i][j] == np.amin(sample_mat):

                                       

                    if done==False:

                        mini = i

                        minj = j

                        done = True

                    if np.amin(sample_mat) >= cut_off*old_min:

                        print("distance too great, stopping cluster formation")

                        return cluster

                    

        for i in cluster[minj]:

            cluster[mini].append(i)

        del cluster[minj]

        

        

        

        if len(cluster)%20==0:

            if verbose:

                

                print(f"Number of clusters formed: {len(cluster)}")

                print(f"distance between the 2 closest clusters: {np.amin(sample_mat)}\n\n")

        

        if len(cluster)%len_segment==0:

            old_min = np.amin(sample_mat)



    return cluster

clust = Get_Clusters(points ,linkage='cent')

plt.figure(figsize=(15,8))

plot(clust, "Annual Income", "Spending_score")
clust = Get_Clusters(points,cut_off=1.6, linkage = 'single', dist_type = 'e', verbose=0)

plt.figure(figsize=(15,8))

plot(clust, "Annual Income", "Spending_score")
clust = Get_Clusters(points,N=5 ,linkage = 'complete', dist_type = 'm', verbose=0)

plt.figure(figsize=(15,8))

plot(clust, "Annual Income", "Spending_score")
clust = Get_Clusters(points, N=20 ,linkage = 'average', dist_type = 'e', verbose=0)

plt.figure(figsize=(15,8))

plot(clust, "Annual Income", "Spending_score")
age_vs_score = convert_to_points(data['Age'], data['Spending Score (1-100)'])

cluster = Get_Clusters(age_vs_score, linkage='complete', verbose=0)

plt.figure(figsize=(15,8))

plot(cluster, "Age", "Spending Score")
age_vs_income = convert_to_points(data['Age'], data['Annual Income (k$)'])

cluster = Get_Clusters(age_vs_income, linkage='complete', verbose=0)

plt.figure(figsize=(15,8))

plot(cluster, "Age", "Annual Income")