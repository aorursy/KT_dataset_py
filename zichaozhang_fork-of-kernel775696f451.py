# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

from numpy.random import seed

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_validate, cross_val_score, LeaveOneOut

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

seed(1)

path = '/kaggle/input/predict-disease/coris.dat'

#path = '/home/zzc/Downloads/coris.dat'

raw_dat = np.genfromtxt(path, skip_header = 3, delimiter = ',', dtype = float)

# Random split

randomize = np.arange(len(raw_dat))

np.random.shuffle(randomize)

raw_dat = raw_dat[randomize]



def accuracy(Y_pred, Y_label):

    acc = np.sum(Y_pred == Y_label)/len(Y_pred)

    return acc



train_len = int(round(len(raw_dat) * 0.5))  #split training data and testing data

X_train = raw_dat[0:train_len,1:10]

Y_train = raw_dat[0:train_len,10:]

X_test = raw_dat[train_len:,1:10]

Y_test = raw_dat[train_len:,10:]



mean = np.mean(X_train, axis = 0)

std_dev = np.std(X_train, axis = 0)

X_train = np.divide(np.subtract(X_train, mean), std_dev)

X_test = np.divide(np.subtract(X_test, mean), std_dev)



lmb = 0.001

w = np.zeros((X_train.shape[1],))

w = w.reshape((w.shape[0],1))

b = np.zeros((1,))

iter = 500

lr = 0.2

step = 1

for ite in range(iter):

    f_pred = Y_train - np.clip(1 / (1.0 + np.exp(-np.add(np.dot(X_train, w), b))), 1e-6, 1 - 1e-6)

    w_grad = np.dot(X_train.T, f_pred) + lmb * w

    b_grad = np.sum(f_pred)



    w = w + lr / np.sqrt(step) * w_grad

    b = b + lr / np.sqrt(step) * b_grad

    step = step + 1

    if iter == 200:

        a = 1



test_pred = np.clip(1 / (1.0 + np.exp(-np.add(np.dot(X_test, w), b))), 1e-6, 1 - 1e-6)

test_pred = np.round(test_pred)

acc = accuracy(test_pred, Y_test)

print(acc)
X = raw_dat[:,1:10]

Y = raw_dat[:,10:]



mean = np.mean(X, axis = 0)

std_dev = np.std(X, axis = 0)

X = np.divide(np.subtract(X, mean), std_dev)

total_len = X.shape[0]

logreg_corr = 0

knn_corr = 0

for index in range(total_len):

    X_test = X[index,:]

    Y_test = Y[index,:]

    if index == 0:

        X_train = X[index+1:,:]

        Y_train = Y[index+1:,:]

    else:

        X_train1 = X[0:index, :]

        X_train2 = X[index + 1:, :]

        X_train = np.vstack((X_train1, X_train2))

        Y_train1 = Y[0:index, :]

        Y_train2 = Y[index + 1:, :]

        Y_train = np.vstack((Y_train1, Y_train2))

    w = np.zeros((X.shape[1],))

    w = w.reshape((w.shape[0], 1))

    b = np.zeros((1,))

    iter = 500

    lr = 0.002

    step = 1

    lmb = 0.001

    for ite in range(iter):

        f_pred = Y_train - np.clip(1 / (1.0 + np.exp(-np.add(np.dot(X_train, w), b))), 1e-6, 1 - 1e-6)

        w_grad = np.dot(X_train.T, f_pred) + lmb * w

        b_grad = np.sum(f_pred)



        w = w + lr / np.sqrt(step) * w_grad

        b = b + lr / np.sqrt(step) * b_grad

        step = step + 1



    test_pred = np.clip(1 / (1.0 + np.exp(-np.add(np.dot(X_test, w), b))), 1e-6, 1 - 1e-6)

    test_pred = np.round(test_pred)



    if test_pred == Y_test:

        logreg_corr = logreg_corr + 1

    #KNN

    knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')

    Y_train = Y_train.reshape((1,Y_train.shape[0]))

    Y_train = Y_train[0,:]

    knn.fit(X_train,Y_train)

    y_pred=knn.predict([X_test])

    if y_pred == Y_test:

        knn_corr = knn_corr + 1



#acc = accuracy(test_pred, Y_test)

acc_logreg = logreg_corr/total_len

acc_knn = knn_corr/total_len



print('Method\tAccuracy')

print('KNN\t%6.3f'%(acc_knn))

print('Logistic regression\t%6.3f'%(acc_logreg))
Y = np.ravel(Y)

num_instances = len(X)

loocv = LeaveOneOut()

model = LogisticRegression()

resultlog = cross_val_score(model, X, Y, cv=loocv)





knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')

loocv = LeaveOneOut()

resultknn = cross_val_score(knn, X, Y, cv=loocv)

print("Accuracy of logistic regression: %.3f%%" % (resultlog.mean()*100.0))

print("Accuracy of knn: %.3f%%" % (resultknn.mean()*100.0))
Y = np.ravel(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = total_len - 1)

Y_train = np.ravel(Y_train)

Y_test = np.ravel(Y_test)

logreg = LogisticRegression(random_state=0).fit(X_train, Y_train)

knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')

knn.fit(X_train,Y_train)

scores_logreg = cross_validate(logreg, X, Y, cv=5, scoring='accuracy')

print('test score for logistic regression:',scores_logreg['test_score'])

scores_knn = cross_validate(knn, X, Y, cv=5, scoring='accuracy')

print('test score for knn:',scores_knn['test_score'])