from sklearn import datasets

import numpy as np
iris = datasets.load_iris() #iris flowers 鳶尾花數據集分類

X = iris.data # features

Y = iris.target # labels
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=0) #30% test , 70% train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_1 = sc.fit_transform(X_train)

X_test_1 = sc.fit_transform(X_test)

X_train_1[:3],X_test_1[:3]
sc = StandardScaler()

sc.fit(X_train)

X_train_2 = sc.transform(X_train)

X_test_2 = sc.transform(X_test)

X_train_2[:3],X_test_2[:3]