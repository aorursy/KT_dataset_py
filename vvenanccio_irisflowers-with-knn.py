import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline

df = pd.read_csv('../input/iris.csv.csv')


df.columns


df

df.describe()

sb.pairplot(df, hue='target')

X = np.array(df.drop('target',1))
X

y = np.array(df.target)
y

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X,y)

knn.predict([[4.2,3.1,2.1,0.9]])

knn.predict([[6.4,2.2,1.1,0.9]])

knn.predict([[6.3,1.2,0.1,0.9]])

knn.predict([[1.3,0.2,0.1,0.9]])

knn.predict([[6.3,1.2,5.1,4.9]])































