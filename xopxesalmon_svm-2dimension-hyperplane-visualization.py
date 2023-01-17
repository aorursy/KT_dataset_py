import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix

from mpl_toolkits.mplot3d import Axes3D
df_iris=pd.read_csv('../input/iris/Iris.csv')

df_iris.head()
df=df_iris.iloc[:100,1:4]



ax = plt.axes(projection='3d')

ax.scatter(df['SepalLengthCm'], df['SepalWidthCm'], df['PetalLengthCm'])

ax.set_xlabel('Sepal Length (cm)')

ax.set_ylabel('Sepal Width (cm)')

ax.set_zlabel('Petal Length (cm)')

# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray)

X=df.to_numpy()



# Converting string value to int type for labels: Setosa = 0, Versicolor = 1

y=df_iris.iloc[:100,-1]

y = LabelEncoder().fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)



svc = LinearSVC()

svc.fit(X_train, y_train)
plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c='b')

ax.set_xlabel('Sepal Length (cm)')

ax.set_ylabel('Sepal Width (cm)')

ax.set_zlabel('Petal Length (cm)')



zz = lambda xx,yy: (-svc.intercept_[0]-svc.coef_[0][0]*xx-svc.coef_[0][1]*yy) / svc.coef_[0][2]

tmpx = np.linspace(4, 7, 20)

tmpy = np.linspace(2, 5, 20)

xx,yy = np.meshgrid(tmpx,tmpy)

ax.plot_surface(xx, yy, zz(xx,yy), cmap='Reds')
plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c='b')

ax.set_xlabel('Sepal Length (cm)')

ax.set_ylabel('Sepal Width (cm)')

ax.set_zlabel('Petal Length (cm)')



zz = lambda xx,yy: (-svc.intercept_[0]-svc.coef_[0][0]*xx-svc.coef_[0][1]*yy) / svc.coef_[0][2]

tmpx = np.linspace(4, 7, 20)

tmpy = np.linspace(2, 5, 20)

xx,yy = np.meshgrid(tmpx,tmpy)

ax.plot_surface(xx, yy, zz(xx,yy), cmap='Reds')

for ii in range(0,360,1):

    ax.view_init(elev=10., azim=ii)
plt.figure()

ax = plt.axes(projection='3d')

ax.scatter(X_train[:,0], X_train[:,1], X_train[:,2], c='b')

ax.set_xlabel('Sepal Length (cm)')

ax.set_ylabel('Sepal Width (cm)')

ax.set_zlabel('Petal Length (cm)')



zz = lambda xx,yy: (-svc.intercept_[0]-svc.coef_[0][0]*xx-svc.coef_[0][1]*yy) / svc.coef_[0][2]

tmpx = np.linspace(4, 7, 20)

tmpy = np.linspace(2, 5, 20)

xx,yy = np.meshgrid(tmpx,tmpy)

ax.plot_surface(xx, yy, zz(xx,yy), cmap='Reds')

for ii in range(0,30,1):

    ax.view_init(elev=25, azim=ii)
y_pred = svc.predict(X_test)

confusion_matrix(y_test, y_pred)