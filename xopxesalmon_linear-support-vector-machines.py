import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix
df_iris=pd.read_csv('../input/iris/Iris.csv')

df_iris.head()
setosa=df_iris.iloc[:50,:]

versicolor=df_iris.iloc[50:100,:]

virginica=df_iris.iloc[100:150,:]



sns.distplot(a=setosa['PetalLengthCm'], label="Iris-setosa")

sns.distplot(a=versicolor['PetalLengthCm'], label="Iris-versicolor" )

sns.distplot(a=virginica['PetalLengthCm'], label="Iris-virginica")



# Add title

plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()
df=df_iris.iloc[:100,2:4]
g=sns.jointplot(x=df_iris['SepalWidthCm'], y=df_iris['PetalLengthCm'],kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=5, linewidth=1)

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$Sepal$ $Width$ $(cm)$", "$Petal$ $Length$ $(cm)$");
X=df.to_numpy()
y=df_iris.iloc[:100,-1]

y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)



svc = LinearSVC()

model_fit=svc.fit(X_train, y_train)
decision_function = model_fit.decision_function(X_train)

support_vector_indices = np.where((2 * y_train - 1) * decision_function <= 1)[0]

support_vectors = X_train[support_vector_indices]
plt.figure()

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='winter')

ax=plt.gca()

xlim=ax.get_xlim()

w = svc.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(xlim[0], xlim[1])

yy = a * xx - svc.intercept_[0] / w[1]

plt.plot(xx, yy)

yy = a * xx - (svc.intercept_[0] - 1) / w[1]

plt.plot(xx, yy, 'k--')

yy = a * xx - (svc.intercept_[0] + 1) / w[1]

plt.plot(xx, yy, 'k--')

plt.xlabel('Sepal Width (cm)')

plt.ylabel('Petal Length (cm)')

plt.title('Train vectors and support vectors')

plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,

                linewidth=1, facecolors='none', edgecolors='r')
plt.figure()

plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='winter')

ax=plt.gca()

xlim=ax.get_xlim()

w = svc.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(xlim[0], xlim[1])

yy = a * xx - svc.intercept_[0] / w[1]

plt.plot(xx, yy)

yy = a * xx - (svc.intercept_[0] - 1) / w[1]

plt.plot(xx, yy, 'k--')

yy = a * xx - (svc.intercept_[0] + 1) / w[1]

plt.plot(xx, yy, 'k--')

plt.xlabel('Sepal Width (cm)')

plt.ylabel('Petal Length (cm)')

plt.title('Test vectors')
y_pred = svc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
