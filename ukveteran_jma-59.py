from sklearn import datasets

iris   = datasets.load_iris()

data   = iris.data

target = iris.target

iris.data.shape





data.ndim





from sklearn import preprocessing

import numpy as np

X = np.array([[ 1., -1.,  2.],

              [ 2.,  0.,  0.],

             [ 0.,  1., -1.]])

X_scaled = preprocessing.scale(X)

import pandas

df = pandas.read_csv('../input/000-example.csv', index_col=False, header=0);

Group    = df["Group"].values

Features = df[["F1","F2","F3"]].values

Features = Features.astype(float)





from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(Features, Group)

BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)

print(clf.predict(Features))

print(clf.predict_proba(Features))



import numpy as np

X = np.random.randint(5, size=(6, 3))

y = np.array([1, 2, 3, 4, 5, 6])

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X, y)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

print(clf.predict_proba(X))

print(clf.predict_proba(np.array([3,3,2])))



XX = np.random.randint(5, size=(6, 3))

yy = np.array([1, 2, 3, 4, 5, 6])



clf.partial_fit(XX,yy)

print(clf.predict_proba(X))



measurements = [

     {'city': 'Dubai', 'temperature': 33.},

     {'city': 'Dubai', 'temperature': 12.},

     {'city': 'San Fransisco', 'temperature': 18.},

 ]



from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()



vec.fit_transform(measurements).toarray()





vec.get_feature_names()

['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']