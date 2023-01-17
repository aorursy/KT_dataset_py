import numpy as np

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from datetime import datetime

import matplotlib.pyplot as plt
data = np.genfromtxt('../input/insar-fashion-mnist-challenge/train.csv', dtype=np.uint8, delimiter=',', skip_header=1)

labels_train = data[:,0]

inputs_train = data[:,1:785]

labels, counters = np.unique(labels_train, return_counts=True)

for lab, num in zip(labels, counters):

    print("Label {} has {} samples".format(lab, num))
fig, ax_array = plt.subplots(6, 6)

axes = ax_array.flatten()

for i, ax in enumerate(axes):

    ax.imshow(np.reshape(inputs_train[i,:], (28, 28)), cmap='gray_r')

plt.show()
n = inputs_train.shape[0]

X = inputs_train[0:n,:]

Y = labels_train[0:n]

print(X.shape, Y.shape)
%%time

pca = PCA(n_components=200)

X_new = pca.fit_transform(X)

#print(pca.explained_variance_ratio_)

print(sum(pca.explained_variance_ratio_))
pipe = Pipeline([

    ('sc', StandardScaler()),

    ('pca', PCA(n_components=100)),

    ('svc', SVC(kernel='poly'))

])

gamma_grid = np.logspace(-2,2,5)

C_grid = [10] #np.logspace(-2,2,5)

parameters = {'svc__gamma': gamma_grid, 'svc__C': C_grid}
%%time

K = 5 # 1/2 train, 1/2 validation

clf = GridSearchCV(estimator=pipe, param_grid=parameters, cv=K, scoring="accuracy", n_jobs=2)

#clf = RandomizedSearchCV(svc, parameters, cv=K, scoring="accuracy", random_state=0)

clf.fit(X, Y)

print(clf.best_params_)

print(clf.best_score_)
data = np.genfromtxt('../input/insar-fashion-mnist-challenge/test_inputs.csv', dtype=np.uint8, delimiter=',', skip_header=1)

ypred_test = clf.predict(data)
ntest = ypred_test.shape[0]

id=np.arange(ntest)

data_to_submit = np.concatenate((id.reshape(ntest, 1), ypred_test.reshape(ntest,1)), axis=1)

nom = 'submission_' + datetime.now().strftime('%H:%M:%S_%d-%m-%Y') + '.csv'

np.savetxt(nom, data_to_submit, delimiter=',', fmt='%4d', header="id,predicted")