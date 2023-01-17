from __future__ import print_function

import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import sys

from time import time

from sklearn.decomposition import PCA

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

input_dir="../input"

# an util function for show images

def plot_number(row, w=28, h=28, labels=True):

    if labels:

        label = row[0]

        pixels = row[1:]

    else:

        label = ''

        pixels = row[0:]

    pixels = 255-np.array(pixels, dtype='uint8')

    pixels = pixels.reshape((w, h))

    if labels:

        plt.title('Label is {label}'.format(label=label))

    plt.imshow(pixels, cmap='gray')



def plot_slice(rows, size_w=28, size_h=28, labels=True):

    num = rows.shape[0]

    w = 4

    h = math.ceil(num / w)

    fig, plots = plt.subplots(h, w)

    fig.tight_layout()



    for n in range(0, num):

        s = plt.subplot(h, w, n+1)

        s.set_xticks(())

        s.set_yticks(())

        plot_number(rows.ix[n], size_w, size_h, labels)

    plt.show()

# an util function for invoke classifier

def gradient_boosting_classifier(train_x, train_y):  

    from sklearn.ensemble import GradientBoostingClassifier  

    model = GradientBoostingClassifier(n_estimators=200)  

    model.fit(train_x, train_y)  

    return model
train = pd.read_csv('%s/train.csv' % input_dir, header=0)

X_train = train.drop(['label'], axis='columns', inplace=False)

y_train = train['label']

    

X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.30, random_state=4)

n_components = 20

t0 = time()

pca = PCA(n_components=n_components, svd_solver='randomized',

              whiten=True).fit(X_train)

print("done in %0.3fs" % (time() - t0))

X_train_pca = pca.transform(X_train)

print("the pca featured take",pca.explained_variance_ratio_.sum()*100)

original_data_after_pca=pca.inverse_transform(X_train_pca)

plot_slice(pd.DataFrame(original_data_after_pca[0:12]),labels=False)
t0 = time()

#model=logistic_regression_classifier(X_train_pca,y_train)

model=gradient_boosting_classifier(X_train_pca,y_train)

print("done in %0.3fs" % (time() - t0))

score = model.score(pca.transform(X_ts), y_ts)

print(score)



test = pd.read_csv('%s/test.csv' % input_dir, header=0)

pred = pd.DataFrame(model.predict(pca.transform(test.values)),columns=['Label'])

pred['ImageId'] = pred.index +1

pred.to_csv('submission.csv', index=False)