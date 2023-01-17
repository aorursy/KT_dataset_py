import os

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['figure.figsize'] = (12, 9)

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



RANDOM_STATE = 17
# change this if necessary

PATH_TO_SAMSUNG_DATA = "../input"
X_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_train.txt"))

y_train = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_train_labels.txt")).astype(int)



X_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA, "samsung_test.txt"))

y_test = np.loadtxt(os.path.join(PATH_TO_SAMSUNG_DATA,

                                  "samsung_test_labels.txt")).astype(int)
# Checking dimensions

assert(X_train.shape == (7352, 561) and y_train.shape == (7352,))

assert(X_test.shape == (2947, 561) and y_test.shape == (2947,))
X = np.concatenate([X_train,X_test], axis=0)

y = np.concatenate([y_train,y_test], axis= 0)

# Your code here
np.unique(y)
n_classes = np.unique(y).size
# Your code here

sds = StandardScaler()

X_scaled = sds.fit_transform(X)
# Your code here

pca = PCA(random_state = 10)

X_pca = pca.fit_transform(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_), color = 'red', lw = 2)

plt.xlim(50,70)

plt.show()
# Your code here

# 2
# Your code here

pca.explained_variance_ratio_[0]

# 2
# Your code here

plt.scatter(X_pca[:,0], X_pca[:,1] ,c = y, s=20, cmap='viridis');

# 2
# Your code here

kmean = KMeans(n_clusters = 6, n_init = 100, random_state = 10)

kmean.fit(X_pca[:,0:1])
# Your code here

cluster_labels = kmean.labels_

plt.scatter(X_pca[:,0],X_pca[:,1] , c=cluster_labels, s=20, cmap='viridis');
tab = pd.crosstab(y, cluster_labels, margins=True)

tab.index = ['walking', 'going up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']

tab
# 4
# Your code here

inertia = []

for k in tqdm_notebook(range(1, n_classes + 1)):

    kmeans = KMeans(n_clusters = k, n_init = 100, random_state = 10)

    kmeans.fit(X_pca[:,0:1])

    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1,n_classes+1), inertia)
# 3
ag = AgglomerativeClustering(n_clusters=n_classes, 

                             linkage='ward').fit(X_pca)
# Your code here

print(metrics.adjusted_rand_score(y, ag.labels_), metrics.adjusted_rand_score(y, kmean.labels_))
# 1
# Your code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
svc = LinearSVC(random_state=RANDOM_STATE)

svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
%%time

# Your code here

best_svc = GridSearchCV(svc, param_grid=svc_params, cv = 3)

best_svc.fit(X_train_scaled, y_train)
best_svc.best_params_, best_svc.best_score_
# 3
y_predicted = best_svc.predict(X_test_scaled)
tab = pd.crosstab(y_test, y_predicted, margins=True)

tab.index = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab.columns = ['walking', 'climbing up the stairs',

             'going down the stairs', 'sitting', 'standing', 'laying', 'all']

tab
# 4
# 1
# Your code here

pca1 = PCA(random_state = RANDOM_STATE)

X_train_pca = pca1.fit_transform(X_train_scaled)

X_test_pca = pca1.transform(X_test_scaled)

svc1 = LinearSVC(random_state = RANDOM_STATE)

best_svc1 = GridSearchCV(svc1, param_grid = svc_params, cv = 3)

best_svc1.fit(X_train_pca, y_train)
print(best_svc1.best_params_, best_svc1.best_score_)
# 2