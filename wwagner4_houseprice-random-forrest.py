import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import ensemble

import random



%matplotlib inline



plt.style.use('bmh')
df_final = pd.read_csv("../input/ctrain01/CTrain01.csv")

print(f"df_final : {df_final.shape}")



df_train = df_final[df_final.SalePrice.notna()]

df_test = df_final[df_final.SalePrice.isna()]

print(f"df_train : {df_train.shape}")

print(f"df_test : {df_test.shape}")
non_features = ['Id', 'SalePrice']



xkeys = [key for key in df_final.keys() if key not in non_features]



X = df_train[xkeys].values

y = df_train['SalePrice'].values



print(f"X : {X.shape}")

print(f"y : {y.shape}")
from sklearn.model_selection import train_test_split



mds = [None, 1, 2, 3, 10, 100, 1000]

data = []

for md in mds:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=random.randrange(10000))

    scores = []

    ns = range(100)

    for n in ns:

        clf = ensemble.RandomForestClassifier(max_depth=md, n_estimators=1)

        clf.fit(X_train, y_train.ravel())

        s = clf.score(X_test, y_test.ravel())

        scores.append(s)

    data.append(scores)



plt.boxplot(data, labels=[str(x) for x in mds])

plt.title("'max_depth' crossvalidation")

plt.show()
ests = [1, 2, 3, 10, 20, 50]

data = []

for est in ests:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=random.randint(0, 5000))

    scores = []

    ns = range(100)

    for n in ns:

        clf = ensemble.RandomForestClassifier(max_depth=None, n_estimators=est)

        clf.fit(X_train, y_train.ravel())

        s = clf.score(X_test, y_test.ravel())

        scores.append(s)

    data.append(scores)



plt.boxplot(data, labels=ests)

plt.title("'n_estimators' crossvalidation")

plt.show()
clf = ensemble.RandomForestClassifier(max_depth=None, n_estimators=1)

clf.fit(X, y.ravel())



X1 = df_test[xkeys].values

print("X1 shape", X1.shape)



p = clf.predict(X1)

t = df_train['SalePrice']

plt.boxplot([p, t], labels=["pred", "train"])

plt.title("'predicted / training sale price")

plt.show()
p1 = list(p)

ids = list(df_test['Id'])

print(len(ids), len(p1) )

print(p1[0], ids[0] )

data = list(zip(ids, p1))

df_subm = pd.DataFrame(data, columns=["Id", "SalePrice"])

df_samp = pd.read_csv("../input/ctrain01/sample_submission.csv")

print("subm", df_subm.shape)

print("samp", df_samp.shape)

print("subm", df_subm.keys())

print("samp", df_samp.keys())



print(df_subm['Id'].describe())

print(df_samp['Id'].describe())



nam = "subm_ctrain_03.csv"

df_subm.to_csv(nam, index=False)

print(f"wrote submission to {nam}")