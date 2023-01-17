from tensorflow import keras

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt



from sklearn.cluster import KMeans 



from sklearn import metrics



from sklearn.svm import SVC



import pandas as pd
train = pd.read_csv('../input/fashion-mnist_train.csv')

train_x = train[list(train.columns)[1:]].values

train_y = train['label'].values

## normalize and reshape the predictors  

train_x = train_x / 255



## create train and validation datasets

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)



## reshape the inputs

train_x = train_x.reshape(-1, 784)

val_x = val_x.reshape(-1, 784)





clf = SVC(gamma=0.1)

clf.fit(train_x, train_y) 



predict=clf.predict(val_x)

acc = (100/predict.shape[0] )*((predict == val_y)==True).sum()

print(" le teaux de réussite du modele est de ",acc,"%")
res = np.arange(9,dtype="double")

for k in np.arange(9):

    km = KMeans(n_clusters=k+2)

    km.fit(train_x)

    res[k] = metrics.silhouette_score(train_x,km.labels_)

    

print(res)



plt.title("Silhouette")

plt.xlabel("numéro du cluster")

plt.plot(np.arange(2,11,1),res)