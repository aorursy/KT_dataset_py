import pandas as pd
data = pd.read_csv('../input/train.csv')
data.head()

#Take label out to perform PCA
y = data['label']
X = data.loc[:,data.columns !='label']
print(y.shape)
print(X.shape)
##Perform PCA for dimension Reduction
#First we will do standarization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=784)
pca.fit(X_std)
#The Amount of variance each PCA covered
var = pca.explained_variance_ratio_
#Commulative Variance
var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
pca = PCA(n_components=200)
pca.fit(X_std)
X_transformed = pca.fit_transform(X_std)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size = .3)
clf = DecisionTreeClassifier(random_state = 0)
clf.fit(X_train,y_train)
print(accuracy_score(y_test,clf.predict(X_test)))
from sklearn.model_selection import GridSearchCV
parameters = {'min_samples_leaf':(10,15,20), 'max_depth':(25,30,35)}
clf = DecisionTreeClassifier()
clf1 = GridSearchCV(clf,parameters)
clf1.fit(X_train,y_train)
print(accuracy_score(y_test,clf1.predict(X_test)))
test_data = pd.read_csv('../input/test.csv')
test_data.head()
test_data_std = StandardScaler().fit_transform(test_data)
pca.fit(test_data_std)
test_data_transformed = pca.fit_transform(test_data_std)
prediction = np.array(clf1.predict(test_data_transformed))