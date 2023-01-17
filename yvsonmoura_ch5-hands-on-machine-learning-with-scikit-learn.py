## Pages 155-158

# Support Vector Machine - very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression, and even outlier detection.
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

svm_clf = Pipeline([
("scaler", StandardScaler()),
("linear_svc", LinearSVC(C=1, loss="hinge")),
])

svm_clf.fit(X, y)

svm_clf.predict([[5.5, 1.7]])
## Pages 159

# Nonlinear SVM Classification
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])

polynomial_svm_clf.fit(X, y)
polynomial_svm_clf.predict(X)

## Page 160a - Figure 5-6

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Plot design
plt.style.use('ggplot')

# Data
moons = make_moons(n_samples=100, shuffle=True, noise=0.15, random_state=42)
X = moons[0]
y = moons[1]

X_firstmoon = X[np.where(y == 0)] # First Moon
X_secondmoon = X[np.where(y == 1)] # Second Moon

# Meshgrid
XX = np.arange(-3, 3, 0.025)
X_bg = np.array([[0,0]])

for i in range(len(XX)):
    for j in range(len(XX)):
        X_bg = np.append(X_bg, [[XX[i], XX[j]]], axis=0)
    

# Linear SVM Model using Polynomial Features
C_hyperparameter = 10
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=C_hyperparameter, loss="hinge"))
    ])

polynomial_svm_clf.fit(X, y)
y_pred = polynomial_svm_clf.predict(X_bg)

# Plot
fig = plt.subplots(1, 1, figsize=(16,9))

plt.scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=125)
plt.scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=125)
plt.xlabel(r'$X_1$', fontsize=25, color='k')
plt.ylabel(r'$X_2$', fontsize=25, color='k')
plt.title('Linear SVM classifier using polynomial features (C = %.1f)' % C_hyperparameter, fontsize=25, color='k')

# Background color - Predicted values
plt.scatter(X_bg[np.where(y_pred==0),0], X_bg[np.where(y_pred==0),1], color='purple', marker='.', s=50, alpha=0.1)
plt.scatter(X_bg[np.where(y_pred==1),0], X_bg[np.where(y_pred==1),1], color='green', marker='.', s=50, alpha=0.1)

## Page 160b

# Polynomial Kernel - Kernel Trick
from sklearn.svm import SVC

d1 = 3
C1_hyperparameter = 5
r1 = 1

d2 = 10
C2_hyperparameter = 5
r2 = 100

poly_kernel_svm_clf1 = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=d1, coef0=r1, C=C1_hyperparameter))
    ])

poly_kernel_svm_clf1.fit(X, y)

y_pred1 = poly_kernel_svm_clf1.predict(X_bg)

poly_kernel_svm_clf2 = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=d2, coef0=r2, C=C2_hyperparameter))
    ])

poly_kernel_svm_clf2.fit(X, y)

y_pred2 = poly_kernel_svm_clf2.predict(X_bg)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(24,9))

# Data
axes[0].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[0].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[0].set_title('d = {0}, r = {1}, C ={2}' .format(d1, r1, C1_hyperparameter), fontsize=25, color='k')

axes[1].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[1].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[1].set_title('d = {0}, r = {1}, C ={2}' .format(d2, r2, C2_hyperparameter), fontsize=25, color='k')

# Background color - Predicted values
axes[0].scatter(X_bg[np.where(y_pred1==0),0], X_bg[np.where(y_pred1==0),1], color='purple', marker='.', s=30, alpha=0.1)
axes[0].scatter(X_bg[np.where(y_pred1==1),0], X_bg[np.where(y_pred1==1),1], color='green', marker='.', s=30, alpha=0.1)

axes[1].scatter(X_bg[np.where(y_pred2==0),0], X_bg[np.where(y_pred2==0),1], color='purple', marker='.', s=30, alpha=0.1)
axes[1].scatter(X_bg[np.where(y_pred2==1),0], X_bg[np.where(y_pred2==1),1], color='green', marker='.', s=30, alpha=0.1)


## Page 161/162

# Adding Similarity Features


from sklearn.svm import SVC

gamma1 = 0.1
C1_hyperparameter = 0.001

gamma2 = 0.1
C2_hyperparameter = 1000

gamma3 = 5
C3_hyperparameter = 0.001

gamma4 = 5
C4_hyperparameter = 1000


rbf_kernel_svm_clf1 = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=gamma1, C=C1_hyperparameter))
])
rbf_kernel_svm_clf1.fit(X, y)

y_pred1 = rbf_kernel_svm_clf1.predict(X_bg)

rbf_kernel_svm_clf2 = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=gamma2, C=C2_hyperparameter))
])
rbf_kernel_svm_clf2.fit(X, y)

y_pred2 = rbf_kernel_svm_clf2.predict(X_bg)

rbf_kernel_svm_clf3 = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=gamma3, C=C3_hyperparameter))
])
rbf_kernel_svm_clf3.fit(X, y)

y_pred3 = rbf_kernel_svm_clf3.predict(X_bg)

rbf_kernel_svm_clf4 = Pipeline([
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=gamma4, C=C4_hyperparameter))
])
rbf_kernel_svm_clf4.fit(X, y)

y_pred4 = rbf_kernel_svm_clf4.predict(X_bg)


# Plot
fig, axes = plt.subplots(2, 2, figsize=(20,10))

# Data
axes[0, 0].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[0, 0].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[0, 0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0, 0].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[0, 0].set_title(r'$\gamma = {0}, C ={1}$' .format(gamma1, C1_hyperparameter), fontsize=25, color='k')

axes[0, 1].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[0, 1].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[0, 1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0, 1].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[0, 1].set_title(r'$\gamma = {0}, C ={1}$' .format(gamma2, C2_hyperparameter), fontsize=25, color='k')

axes[1, 0].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[1, 0].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[1, 0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1, 0].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[1, 0].set_title(r'$\gamma = {0}, C ={1}$' .format(gamma3, C3_hyperparameter), fontsize=25, color='k')

axes[1, 1].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='b', marker='s', s=50)
axes[1, 1].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='g', marker='^', s=50)
axes[1, 1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1, 1].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[1, 1].set_title(r'$\gamma = {0}, C ={1}$' .format(gamma4, C4_hyperparameter), fontsize=25, color='k')

# Background color - Predicted values
axes[0, 0].scatter(X_bg[np.where(y_pred1==0),0], X_bg[np.where(y_pred1==0),1], color='purple', marker='.', s=20, alpha=0.1)
axes[0, 0].scatter(X_bg[np.where(y_pred1==1),0], X_bg[np.where(y_pred1==1),1], color='green', marker='.', s=20, alpha=0.1)

axes[0, 1].scatter(X_bg[np.where(y_pred2==0),0], X_bg[np.where(y_pred2==0),1], color='purple', marker='.', s=20, alpha=0.1)
axes[0, 1].scatter(X_bg[np.where(y_pred2==1),0], X_bg[np.where(y_pred2==1),1], color='green', marker='.', s=20, alpha=0.1)

axes[1, 0].scatter(X_bg[np.where(y_pred3==0),0], X_bg[np.where(y_pred3==0),1], color='purple', marker='.', s=20, alpha=0.1)
axes[1, 0].scatter(X_bg[np.where(y_pred3==1),0], X_bg[np.where(y_pred3==1),1], color='green', marker='.', s=20, alpha=0.1)

axes[1, 1].scatter(X_bg[np.where(y_pred4==0),0], X_bg[np.where(y_pred4==0),1], color='purple', marker='.', s=20, alpha=0.1)
axes[1, 1].scatter(X_bg[np.where(y_pred4==1),0], X_bg[np.where(y_pred4==1),1], color='green', marker='.', s=20, alpha=0.1)

plt.tight_layout()
## Pages 164/165

## SVM Regression
from sklearn.svm import LinearSVR

# Noisy Data
X = 2*np.random.rand(50, 1)
Y = 2*X + np.random.normal(0, 0.3, size=(50, 1))

# Model
e1 = 1.5
svm_reg1 = LinearSVR(epsilon=e1)
svm_reg1.fit(X, Y)

y_pred1 = svm_reg1.predict(X)

e2 = 0.5
svm_reg2 = LinearSVR(epsilon=e2)
svm_reg2.fit(X, Y)

y_pred2 = svm_reg2.predict(X)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].scatter(X, Y, color='b', marker='o', s=50)
axes[0].plot(X, y_pred1, color='r', lw=1)
axes[0].plot(X, y_pred1+e1, "k-", lw=1)
axes[0].plot(X, y_pred1-e1, "k-", lw=1)
axes[0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0].set_ylabel(r'$y$', fontsize=25, color='k')
axes[0].set_title(r'$\epsilon = {0}$' .format(e1), fontsize=25, color='k')

axes[1].scatter(X, Y, color='b', marker='o', s=50)
axes[1].plot(X, y_pred2, color='r', lw=1)
axes[1].plot(X, y_pred2+e2, "k--", lw=1)
axes[1].plot(X, y_pred2-e2, "k--", lw=1)
axes[1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1].set_ylabel(r'$y$', fontsize=25, color='k')
axes[1].set_title(r'$\epsilon = {0}$' .format(e2), fontsize=25, color='k')


## Pages 165/166

# Kernelized SVM model
from sklearn.svm import SVR

# Noisy Data
X = np.arange(-8, 8, 0.16).reshape(100, 1)
Y = X**2 + X - 1 + np.random.normal(0, 4, size=(100, 1))

# Model
d1 = 2
e1 = 10
C1 = 100
svm_poly_reg1 = SVR(kernel="poly", degree=d1, C=C1, epsilon=e1)
svm_poly_reg1.fit(X, Y)

y_pred1 = svm_poly_reg1.predict(X)

d2 = 2
e2 = 10
C2 = 0.1
svm_poly_reg2 = SVR(kernel="poly", degree=d2, C=C2, epsilon=e2)
svm_poly_reg2.fit(X, Y)

y_pred2 = svm_poly_reg2.predict(X)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
axes[0].scatter(X, Y, color='b', marker='o', s=50)
axes[0].plot(X, y_pred1, color='r', lw=1)
axes[0].plot(X, y_pred1+e1, "k-", lw=1)
axes[0].plot(X, y_pred1-e1, "k-", lw=1)
axes[0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0].set_ylabel(r'$y$', fontsize=25, color='k')
axes[0].set_title(r'$degree = {0}, C = {1}, \epsilon = {2}$' .format(d1, C1, e1), fontsize=25, color='k')

axes[1].scatter(X, Y, color='b', marker='o', s=50)
axes[1].plot(X, y_pred2, color='r', lw=1)
axes[1].plot(X, y_pred2+e2, "k--", lw=1)
axes[1].plot(X, y_pred2-e2, "k--", lw=1)
axes[1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1].set_ylabel(r'$y$', fontsize=25, color='k')
axes[1].set_title(r'$degree = {0}, C = {1}, \epsilon = {2}$' .format(d2, C2, e2), fontsize=25, color='k')
