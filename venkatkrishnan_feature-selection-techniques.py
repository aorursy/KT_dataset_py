import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn import linear_model, datasets

import matplotlib.pyplot as plt
import seaborn as sns
X = np.array([[1,1,1],
              [2,2,0],
              [3,3,1],
              [4,4,0],
              [5,5,1],
              [6,6,0],
              [7,7,1],
              [8,7,0],
              [9,7,1]])

# Create dataframe
data_frm = pd.DataFrame(X)

data_frm
# Get the correlation matrix
corr_matrix = data_frm.corr().abs()

corr_matrix
# Select the upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
print(upper)

# Pick the column that has correlation more than 0.95
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
data_frm.drop(to_drop, axis=1)
X, y = make_regression(n_samples=10000, 
                      n_features=100,
                      n_informative=2,
                      random_state=1)

print(X.shape, y.shape)
# Creating linear model

lr_model = linear_model.LinearRegression()
# Recursive feature elimination

rfecv = RFECV(estimator=lr_model, step=1, scoring='neg_mean_squared_error')

rfecv.fit(X, y)

rfecv.transform(X)

rfecv.n_features_
X = [[0,1,0],
    [0,1,1],
    [0,1,0],
    [0,1,1],
    [0,1,0],
    [1,0,0]]
# Variance Threshold
thresholder = VarianceThreshold(threshold=(0.75 * (0.25)))
thresholder.fit_transform(X)
iris_data = datasets.load_iris()

X = iris_data.data
y = iris_data.target

# Create VarianceThreshold object with variance threshold above 0.5
thresholder = VarianceThreshold(threshold=0.5)

# Conduct variance threshold
X_high_variance = thresholder.fit_transform(X)
print(len(X_high_variance))
X_high_variance[0:5]