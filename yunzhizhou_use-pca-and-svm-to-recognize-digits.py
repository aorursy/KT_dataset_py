# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from sklearn.decomposition import PCA,RandomizedPCA

from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

labeled_images = pd.read_csv("../input/train.csv")

images = labeled_images.iloc[:,1:]

labels = labeled_images.iloc[:,:1]
pca = PCA(n_components=64, whiten=True).fit(images)

new_images = pca.transform(images)

train_images,test_images, train_labels, test_labels = train_test_split(new_images,labels, train_size = 0.8)



clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images, test_labels)
test_images = pd.read_csv("../input/test.csv")

new_test_images = pca.transform(test_images)

results = clf.predict(new_test_images)
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True, index_label="ImageId")