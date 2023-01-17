## Page 177/178

# Training and Visualizing a Decision Tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file=os.getcwd()+"/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

!dot -Tpng iris_tree.dot -o iris_tree.png

%matplotlib inline

fig = plt.subplots(1,1, figsize=(10,10))

img = mpimg.imread(os.getcwd()+"/iris_tree.png")
imgplot = plt.imshow(img)

plt.tick_params(top=False, bottom=False, left=False, labelleft=False, labelbottom=False)
plt.show()
## Pages 181/182

# Estimating Class Probabilities
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))
## Page 185

# Regularization Hyperparameters
import numpy as np
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Plot design
plt.style.use('ggplot')

# Data
moons = make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=42)
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

# Decision Tree Model
tree_clf_free = Pipeline(
    [("scaler", StandardScaler()),
     ("decision_tree", DecisionTreeClassifier())
    ])

tree_clf_free.fit(X, y)

y_pred1 = tree_clf_free.predict(X_bg)

tree_clf_restricted = Pipeline(
    [("scaler", StandardScaler()),
     ("decision_tree", DecisionTreeClassifier(min_samples_leaf=4))
    ])

tree_clf_restricted.fit(X, y)

y_pred2 = tree_clf_restricted.predict(X_bg)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16,6))

# Data
axes[0].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='green', marker='o', s=50)
axes[0].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='b', marker='s', s=50)
axes[0].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[0].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[0].set_xlim(-1.5, 2.5)
axes[0].set_ylim(-1.0, 1.5)
axes[0].set_title('No restrictions', fontsize=25, color='k')

axes[1].scatter(X_firstmoon[:,0], X_firstmoon[:,1], color='green', marker='o', s=50)
axes[1].scatter(X_secondmoon[:,0], X_secondmoon[:,1], color='b', marker='s', s=50)
axes[1].set_xlabel(r'$X_1$', fontsize=25, color='k')
axes[1].set_ylabel(r'$X_2$', fontsize=25, color='k')
axes[1].set_xlim(-1.5, 2.5)
axes[1].set_ylim(-1.0, 1.5)
axes[1].set_title('min_samples_leaf = 4', fontsize=25, color='k')

# Background color - Predicted values
axes[0].scatter(X_bg[np.where(y_pred1==0),0], X_bg[np.where(y_pred1==0),1], color='purple', marker='.', s=40, alpha=0.2)
axes[0].scatter(X_bg[np.where(y_pred1==1),0], X_bg[np.where(y_pred1==1),1], color='green', marker='.', s=40, alpha=0.2)

axes[1].scatter(X_bg[np.where(y_pred2==0),0], X_bg[np.where(y_pred2==0),1], color='purple', marker='.', s=40, alpha=0.2)
axes[1].scatter(X_bg[np.where(y_pred2==1),0], X_bg[np.where(y_pred2==1),1], color='green', marker='.', s=40, alpha=0.2)
## Page 186

# Regression with Decision Tree
from sklearn.tree import DecisionTreeRegressor

# Data
moons = make_moons(n_samples=400, shuffle=True, noise=0.07, random_state=3113)
X = moons[0]
y = moons[1]
X1 = moons[0][np.where(moons[1]==1), 0].reshape(200, 1)
X2 = moons[0][np.where(moons[1]==1), 1].reshape(200, 1)
y1 = moons[1][np.where(moons[1]==1)]


tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X1, X2)

export_graphviz(
    tree_reg,
    out_file=os.getcwd()+"/moon_tree.dot",
    feature_names=['x1'],
    class_names=y1,
    rounded=True,
    filled=True
)

!dot -Tpng moon_tree.dot -o moon_tree.png

%matplotlib inline

fig = plt.subplots(1,1, figsize=(16,9))

img = mpimg.imread(os.getcwd()+"/moon_tree.png")
imgplot = plt.imshow(img)

plt.tick_params(top=False, bottom=False, left=False, labelleft=False, labelbottom=False, grid_alpha=0)
plt.show()
## Page 187a

# Predictions of two Decision Tree regression models

plt.style.use('default')

# Domain
XX = np.arange(-0.5, 2.5, 0.005)
XX = XX.reshape(len(XX), 1)
        
# Prediction (using tree_reg instance of previous code lines)
y_pred_tree = tree_reg.predict(XX)
        
# Plot
fig = plt.subplots(1, 1, figsize=(10, 6))
plt.scatter(X1, X2, color='b', s=15)
plt.plot(XX, y_pred_tree, color='red', lw=2, alpha=0.8)
plt.title('max_depth = 2')

## Page 187b

# Regularizing a Decision Tree regressor

# Domain
XX = np.arange(-0.5, 2.5, 0.005)
XX = XX.reshape(len(XX), 1)
        
# Prediction (using tree_reg instance of previous code lines)
tree_reg = DecisionTreeRegressor(min_samples_leaf=10)
tree_reg.fit(X1, X2)
y_pred_tree = tree_reg.predict(XX)
        
# Plot
fig = plt.subplots(1, 1, figsize=(10, 6))
plt.scatter(X1, X2, color='b', s=15)
plt.plot(XX, y_pred_tree, color='red', lw=2, alpha=0.8)
plt.title('min_samples_leaf = 10') # the minimum number of samples a leaf node must have
