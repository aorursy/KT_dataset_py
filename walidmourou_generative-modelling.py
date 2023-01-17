import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm, multivariate_normal
data = pd.read_csv('../input/iris.csv')
data.head()
# frequency of each class

data['class'].value_counts()
# Encoding classes

class_encoder = LabelEncoder()

data['class'] = class_encoder.fit_transform(data['class'].values)
data.head()
# list of class names

list(class_encoder.classes_)
X_train, X_test, y_train, y_test = train_test_split(data[['lsepal','wsepal','lpetal','wpetal']].values, data['class'].values, test_size=0.33, random_state=42)
v, freq = np.unique(y_train, return_counts = True)
pi = freq/freq.sum()
pi
def mean_var(X_train, y_train, idx_feature):

    mu = [X_train[np.argwhere(y_train == i)].reshape((-1,4))[:,idx_feature].mean() for i in range(len(list(class_encoder.classes_)))]

    vr = [X_train[np.argwhere(y_train == i)].reshape((-1,4))[:,idx_feature].var() for i in range(len(list(class_encoder.classes_)))]

    return mu, vr
mu, vr = mean_var(X_train, y_train, idx_feature=0)
# The mean for each class

mu
# The variance for each class

vr
def predict_class(x, pi, mu, vr):

    prb = [pi[i] * norm.pdf(x, loc=mu[i], scale=vr[i]) for i in range(len(mu))]

    return np.argmax(prb)
def test_accuracy(X_test, y_test, pi, mu, vr):

    cnt = 0

    for k, x in enumerate(X_test[:,0]):

        if predict_class(x, pi, mu, vr) == y_test[k]:

            cnt += 1

    return cnt/len(y_test)
feature_name = ['lsepal','wsepal','lpetal','wpetal']

for idx_feature in range(4):

    mu, vr = mean_var(X_train, y_train, idx_feature)

    print('The feature ',feature_name[idx_feature], ' has an accuracy of : ', test_accuracy(X_test, y_test, pi, mu, vr)*100, '%')
def multivariate_mean_var(X_train, y_train):

    mu = [X_train[np.argwhere(y_train == i)].reshape((-1,4)).mean(axis=0) for i in range(len(list(class_encoder.classes_)))]

    vr = [np.cov(X_train[np.argwhere(y_train == i)].reshape((-1,4)).T) for i in range(len(list(class_encoder.classes_)))]

    return mu, vr
def multivariate_predict_class(x, pi, mu, vr):

    prb = [pi[i] * multivariate_normal.pdf(x, mean=mu[i], cov=vr[i]) for i in range(len(mu))]

    return np.argmax(prb)
def multivariate_test_accuracy(X_test, y_test, pi, mu, vr):

    cnt = 0

    for k, x in enumerate(X_test):

        if multivariate_predict_class(x, pi, mu, vr) == y_test[k]:

            cnt += 1

    return cnt/len(y_test)
mu, vr = multivariate_mean_var(X_train, y_train)

print('The multivariate Gaussian estimation has an accuracy of : ', multivariate_test_accuracy(X_test, y_test, pi, mu, vr)*100, '%')