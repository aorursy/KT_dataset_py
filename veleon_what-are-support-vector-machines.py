import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.svm import SVC



iris = datasets.load_iris()

print(iris['DESCR'])
X = iris["data"][:, (2,3)]

y = (iris["target"] == 2).astype(np.float64)
plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.Paired)

plt.xlabel("Petal Length")

plt.ylabel("Petal Width")
# Code from https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html

def plotSVM(clf, X, y, size=(5,5), xlab=None, ylab=None, title=None):

    # figure number

    fignum = 1





    clf.fit(X, y)



    # get the separating hyperplane

    w = clf.coef_[0]

    a = -w[0] / w[1]

    xx = np.linspace(-3, 8)

    yy = a * xx - (clf.intercept_[0]) / w[1]



    # plot the parallels to the separating hyperplane that pass through the

    # support vectors (margin away from hyperplane in direction

    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in

    # 2-d.

    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))

    yy_down = yy - np.sqrt(1 + a ** 2) * margin

    yy_up = yy + np.sqrt(1 + a ** 2) * margin



    # plot the line, the points, and the nearest vectors to the plane

    plt.figure(fignum, figsize=size)

    plt.clf()

    plt.plot(xx, yy, 'k-')

    plt.plot(xx, yy_down, 'k--')

    plt.plot(xx, yy_up, 'k--')



    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,

            facecolors='none', zorder=10, edgecolors='k')

    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,

            edgecolors='k')



    plt.axis('tight')

    x_min = 0

    x_max = 8

    y_min = 0

    y_max = 3



    XX, YY = np.mgrid[x_min:x_max:800j, y_min:y_max:800j]

    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(XX.shape)

    plt.figure(fignum, figsize=size)

    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)



    plt.xlim(x_min, x_max)

    plt.ylim(y_min, y_max)

    

    plt.xlabel(xlab)

    plt.ylabel(ylab)

    if title is not None:

        plt.title(title)

    

    plt.show()
for c in [0.01, 0.1, 1, 10, 1000]:

    clf = SVC(kernel='linear', C=c)

    plotSVM(clf, X, y, (5,5), "Petal Length", "Petal Width", "C = "+str(c))
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.13, random_state=42)



# Code From https://github.com/ageron/handson-ml/blob/master/05_support_vector_machines.ipynb

def plot_dataset(X, y, axes):

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")

    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")

    plt.axis(axes)

    plt.grid(True, which='both')

    plt.xlabel(r"$x_1$", fontsize=20)

    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)



def plot_predictions(clf, axes):

    x0s = np.linspace(axes[0], axes[1], 100)

    x1s = np.linspace(axes[2], axes[3], 100)

    x0, x1 = np.meshgrid(x0s, x1s)

    X = np.c_[x0.ravel(), x1.ravel()]

    y_pred = clf.predict(X).reshape(x0.shape)

    y_decision = clf.decision_function(X).reshape(x0.shape)

    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
for d in [1, 2, 3, 10]:

    clf = SVC(kernel="poly", degree=d, C=5, coef0=10, gamma='auto')

    clf.fit(X,y)

    plt.title('Degree = '+str(d), fontsize=20)

    plot_predictions(clf, [-1.5, 2.5, -1, 1.5])

    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    plt.show()
for g in [1, 3, 5, 10, 30]:

    clf = SVC(kernel="rbf", C=5, coef0=10, gamma=g)

    clf.fit(X,y)

    plt.title('Gamma = '+str(g), fontsize=20)

    plot_predictions(clf, [-1.5, 2.5, -1, 1.5])

    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    plt.show()
# Code from https://github.com/ageron/handson-ml/blob/master/05_support_vector_machines.ipynb

np.random.seed(42)

m = 50

X = 2 * np.random.rand(m, 1)

y = (4 + 3 * X + np.random.randn(m, 1)).ravel()



def plot_svm_regression(svm_reg, X, y, axes):

    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)

    y_pred = svm_reg.predict(x1s)

    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")

    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")

    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")

    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')

    plt.plot(X, y, "bo")

    plt.xlabel(r"$x_1$", fontsize=18)

    plt.legend(loc="upper left", fontsize=18)

    plt.axis(axes)
from sklearn.svm import SVR

for e in [0.1, 1, 2]:

    reg = SVR(epsilon=e, gamma='auto', kernel='linear')

    reg.fit(X,y)

    plot_svm_regression(reg, X, y, [0, 2, 3, 11])

    plt.title("Epsilon = " + str(e))

    plt.show()
for k in ['linear', 'poly', 'rbf']:

    reg = SVR(epsilon=1, gamma='auto', kernel=k)

    reg.fit(X,y)

    plot_svm_regression(reg, X, y, [0, 2, 3, 11])

    plt.title("Kernel = " + str(k))

    plt.show()