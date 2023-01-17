# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



path = '/kaggle/input/predict-disease/coris.dat'

#path = '/home/zzc/Downloads/coris.dat'

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import random

from numpy.random import seed

from sklearn.cluster import KMeans



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate, cross_val_score, LeaveOneOut

seed(1)

np.random.seed(1)

syspath = "/kaggle/input/att_faces/s"  #   10/2.pgm

#arr = np.zeros((10304,1))



personality = 10

train_faces = 4

test_faces = 2

# A = np.zeros((10304,personality * faces))

train_set = np.zeros((10304, (train_faces + test_faces) * personality))

Atrain = np.zeros((10304, train_faces * personality))

Atest = np.zeros((10304, test_faces * personality))

y = np.zeros((personality * (train_faces + test_faces), 1))

y_train = np.zeros((personality * train_faces, 1))

y_test = np.zeros((personality * test_faces, 1))



for i in range(personality):

    for j in range(train_faces + test_faces):

        im = Image.open("/kaggle/input/att_faces/s" + str(i + 1) + "/" + str(j + 1) + ".pgm")

        #im = Image.open("/home/zzc/att_faces/s" + str(i) + "/" + str(j + 1) + ".pgm")

        arr = np.array(im)

        # plt.figure("face" + str(1))

        # plt.imshow(arr, cmap='gray')

        # plt.show()

        arr = arr.flatten()  # .reshape(arr.flatten().shape[0],1)

        train_set[:, i * (train_faces + test_faces) + j] = arr.T

        # y[(0) * faces + j] = i + 1

        y[i * (train_faces + test_faces) + j] = i + 1







# for j in range(test_faces):

#     #im = Image.open("/home/zzc/att_faces/s" + str(i + 1) + "/" + str(j + 1) + ".pgm")

#     im = Image.open("/home/zzc/att_faces/s" + str(1) + "/" + str(j + 1) + ".pgm")

#     arr = np.array(im)

#     arr = arr.flatten()  # .reshape(arr.flatten().shape[0],1)

#     Atrain[:, (0) * test_faces + j] = arr.T

#     # y[(0) * faces + j] = i + 1

#     y[(0) * test_faces + j] = 1

r = np.arange((train_faces + test_faces) * personality)

random.shuffle(r)

train_set = train_set[:,r]

Atrain = train_set[:, 0:train_faces * personality]

meanface = np.mean(Atrain, axis = 1)

meanface = meanface.reshape(meanface.shape[0],1)

train_set = train_set - meanface



Atrain = train_set[:, 0:train_faces * personality]

Atest = train_set[:, train_faces * personality:]

y = y[r]

y_train = y[0:train_faces * personality,0]

y_test = y[train_faces * personality:,0]



# #plot figure for mean face

# meanface = meanface.reshape((112, 92))

# plt.figure("mean face")

# plt.imshow(meanface, cmap='gray')

# plt.show()



ATA = np.dot(Atrain.T, Atrain)

Lambda, V = np.linalg.eig(ATA)

ind = np.argsort(Lambda)

ind = np.flip(ind)

Lambda = Lambda[ind]

V = V[:,ind]

U = np.dot(Atrain,V)

# U, S, Vt = np.linalg.svd(Atrain, full_matrices=True)

# V = Vt.T

eigens = 10

eigenface = U[:, 0:eigens]

eigenface = eigenface / np.sqrt(Lambda[0:eigens])

#len = np.linalg.norm(eigenface[:,0])
# Show variances

x_var = np.arange(personality * train_faces) + 1

sum_var = np.zeros((1, personality * train_faces))

for ijk in range(personality * train_faces):

    sum_var[0, ijk] = sum(Lambda[0:ijk+1])



fig = plt.figure("variances")

plt.plot(x_var, Lambda/(personality * train_faces))

plt.xlabel('The NO.p PC')

plt.title('Variance corresponding to each eigenvector')

plt.show()



fig_sum = plt.figure("Cumulative variance")

plt.plot(x_var, sum_var[0,:]/(personality * train_faces))

plt.xlabel('Number of PCs')

plt.title('Cumulative variance')

plt.show()
#show eigen faces

for k in range(eigens):

    eface = eigenface[:, k]

    eface = eface.reshape((112, 92))

    plt.figure("eigen face" + str(k+1))

    plt.imshow(eface, cmap='gray')

    plt.show()
im = Image.open("/kaggle/input/att_faces/s" + str(7) + "/" + str(2) + ".pgm")

face1 = np.array(im)

#face1 = Atrain[:,0]

#face1 = np.array(im).flatten()

#face1 = face1.reshape((face1.shape[0],1))

faceshow = face1.reshape((112, 92))

meanfacereshaped = meanface.reshape((112, 92))

faceshow = faceshow - meanfacereshaped

plt.figure("original face")

plt.imshow(faceshow, cmap='gray')

plt.show()

face1 = face1.reshape((10304,1))

face1 = face1 - meanface



mul = eigenface[:, 0:40]

Omega = np.dot(mul.T, face1)

reconface = np.dot(mul, Omega)

reconface = reconface.reshape((112, 92))

plt.figure("reconstructed face")

plt.imshow(reconface, cmap='gray')

plt.show()



mul = eigenface[:, 0:30]

Omega = np.dot(mul.T, face1)

reconface = np.dot(mul, Omega)

reconface = reconface.reshape((112, 92))

plt.figure("reconstructed face")

plt.imshow(reconface, cmap='gray')

plt.show()



mul = eigenface[:, 0:20]

Omega = np.dot(mul.T, face1)

reconface = np.dot(mul, Omega)

reconface = reconface.reshape((112, 92))

plt.figure("reconstructed face")

plt.imshow(reconface, cmap='gray')

plt.show()



mul = eigenface[:, 0:10]

Omega = np.dot(mul.T, face1)

reconface = np.dot(mul, Omega)

reconface = reconface.reshape((112, 92))

plt.figure("reconstructed face")

plt.imshow(reconface, cmap='gray')

plt.show()



mul = eigenface[:, 0:5]

Omega = np.dot(mul.T, face1)

reconface = np.dot(mul, Omega)

reconface = reconface.reshape((112, 92))

plt.figure("reconstructed face")

plt.imshow(reconface, cmap='gray')

plt.show()
eig_for_clust = eigenface[:,[0, 1]]

Omega = np.dot(Atrain.T,eig_for_clust)

x = Omega[:,0]

y_coor = Omega[:,1]

dotsize = 300

plt.figure()

plt.scatter(x[y_train == 1],y_coor[y_train == 1], s = dotsize, c = 'r', marker = 'o')

plt.scatter(x[y_train == 2],y_coor[y_train == 2], s = dotsize, c = 'b', marker = 'o')

plt.scatter(x[y_train == 3],y_coor[y_train == 3], s = dotsize, c = 'g', marker = 'o')

plt.scatter(x[y_train == 4],y_coor[y_train == 4], s = dotsize, c = 'r', marker = 's')

plt.scatter(x[y_train == 5],y_coor[y_train == 5], s = dotsize, c = 'b', marker = 's')

plt.scatter(x[y_train == 6],y_coor[y_train == 6], s = dotsize, c = 'g', marker = 's')

plt.scatter(x[y_train == 7],y_coor[y_train == 7], s = dotsize, c = 'r', marker = 'X')

plt.scatter(x[y_train == 8],y_coor[y_train == 8], s = dotsize, c = 'b', marker = 'X')

plt.scatter(x[y_train == 9],y_coor[y_train == 9], s = dotsize, c = 'g', marker = 'X')

plt.scatter(x[y_train == 10],y_coor[y_train == 10], s = dotsize, c = 'k', marker = 'x')

plt.legend(['person 1','person 2','person 3','person 4','person 5','person 6','person 7','person 8','person 9','person 10' ])

plt.show()

def accuracy(Y_pred, Y_label):

    acc = np.sum(Y_pred == Y_label)/len(Y_pred)

    return acc

eig_for_learn = eigenface[:,0:10]

Omega = np.dot(Atrain.T, eig_for_learn)

Omega_test = np.dot(Atest.T,eig_for_learn)

knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')

knn.fit(Omega,y_train)

# res = knn.predict(Omega_test)

# acc = accuracy(res, y_test)

# print("Accuracy of knn predictor:", acc)

Omega_vali = np.dot(train_set.T, eig_for_learn)

scores_knn = cross_validate(knn, Omega_vali, y, cv=5, scoring='accuracy')

print(scores_knn['test_score'])



## Cross-validation

cv_num = 5

res_mat = np.zeros((10,cv_num))

range_vali = np.array([1,2,3,4,5,6,7,8,9,10])

for k in range_vali:

    eig_for_learn = eigenface[:, 0:k]

    Omega = np.dot(Atrain.T, eig_for_learn)

    Omega_test = np.dot(Atest.T, eig_for_learn)

    knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')

    knn.fit(Omega, y_train)

    # res = knn.predict(Omega_test)

    # acc = accuracy(res, y_test)

    # print("Accuracy of knn predictor:", acc)

    Omega_vali = np.dot(train_set.T, eig_for_learn)

    scores_knn = cross_validate(knn, Omega_vali, y, cv=cv_num, scoring='accuracy')

    res_mat[k - 1, :] = scores_knn['test_score']

    # print(scores_knn['test_score'])

mean_mat = np.mean(res_mat, axis= 1)

print(res_mat)

print(mean_mat)
cv_num = 5

res_mat_ori = np.zeros((10,cv_num))

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')

knn.fit(Atrain.T, y_train)

scores_knn = cross_validate(knn, train_set.T, y, cv=cv_num, scoring='accuracy')

mean_mat = np.mean(scores_knn['test_score'])

print(scores_knn['test_score'])

print(mean_mat)
eig_for_clust = eigenface[:,[0, 1]]

Omega = np.dot(Atrain.T,eig_for_clust)

x = Omega[:,0]

y_coor = Omega[:,1]

dotsize = 300

plt.figure()

plt.scatter(x[y_train == 1],y_coor[y_train == 1], s = dotsize, c = 'r', marker = 'o')

plt.scatter(x[y_train == 2],y_coor[y_train == 2], s = dotsize, c = 'b', marker = 'o')

plt.scatter(x[y_train == 3],y_coor[y_train == 3], s = dotsize, c = 'g', marker = 'o')

plt.scatter(x[y_train == 4],y_coor[y_train == 4], s = dotsize, c = 'r', marker = 's')

plt.scatter(x[y_train == 5],y_coor[y_train == 5], s = dotsize, c = 'b', marker = 's')

plt.scatter(x[y_train == 6],y_coor[y_train == 6], s = dotsize, c = 'g', marker = 's')

plt.scatter(x[y_train == 7],y_coor[y_train == 7], s = dotsize, c = 'r', marker = 'X')

plt.scatter(x[y_train == 8],y_coor[y_train == 8], s = dotsize, c = 'b', marker = 'X')

plt.scatter(x[y_train == 9],y_coor[y_train == 9], s = dotsize, c = 'g', marker = 'X')

plt.scatter(x[y_train == 10],y_coor[y_train == 10], s = dotsize, c = 'k', marker = 'x')

plt.legend(['person 1','person 2','person 3','person 4','person 5','person 6','person 7','person 8','person 9','person 10' ])

plt.show()











kmeans = KMeans(n_clusters=10, random_state=0).fit(Omega)

# print(kmeans.labels_)

y_clust = kmeans.labels_

y_clust = y_clust + 1

# kmeans.predict([[0, 0], [12, 3]])



# print(kmeans.cluster_centers_)

plt.figure()

plt.scatter(x[y_clust == 1],y_coor[y_clust == 1], s = dotsize, c = 'r', marker = 'o')

plt.scatter(x[y_clust == 2],y_coor[y_clust == 2], s = dotsize, c = 'b', marker = 'o')

plt.scatter(x[y_clust == 3],y_coor[y_clust == 3], s = dotsize, c = 'g', marker = 'o')

plt.scatter(x[y_clust == 4],y_coor[y_clust == 4], s = dotsize, c = 'r', marker = 's')

plt.scatter(x[y_clust == 5],y_coor[y_clust == 5], s = dotsize, c = 'b', marker = 's')

plt.scatter(x[y_clust == 6],y_coor[y_clust == 6], s = dotsize, c = 'g', marker = 's')

plt.scatter(x[y_clust == 7],y_coor[y_clust == 7], s = dotsize, c = 'r', marker = 'X')

plt.scatter(x[y_clust == 8],y_coor[y_clust == 8], s = dotsize, c = 'b', marker = 'X')

plt.scatter(x[y_clust == 9],y_coor[y_clust == 9], s = dotsize, c = 'g', marker = 'X')

plt.scatter(x[y_clust == 10],y_coor[y_clust == 10], s = dotsize, c = 'k', marker = 'x')

plt.legend(['person 1','person 2','person 3','person 4','person 5','person 6','person 7','person 8','person 9','person 10' ])

plt.show()
