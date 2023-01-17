import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skimage import color
import sys
data = pd.read_csv('../input/colour-data.csv')
data.head()
X = data[['R', 'G', 'B']].values / 255
y = data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X,y)
bayes_rgb_model = GaussianNB()
bayes_rgb_model.fit(X_train, y_train)
print(bayes_rgb_model.score(X_test, y_test))
#convert rgb to lab colour
def rgbTolab(X):
    X = X.reshape(-1,1,1,3)
    X = color.rgb2lab(X).reshape(-1,3)
    return X
#create pipeline model
bayes_lab_model = make_pipeline(
    StandardScaler(),
    FunctionTransformer(rgbTolab),
    GaussianNB())
bayes_lab_model.fit(X_train, y_train)
print(bayes_lab_model.score(X_test, y_test))
knn_rgb_model = KNeighborsClassifier(n_neighbors = 8)
knn_rgb_model.fit(X_train, y_train)
print(knn_rgb_model.score(X_test, y_test))
knn_lab_model = make_pipeline(
    FunctionTransformer(rgbTolab),
    KNeighborsClassifier(n_neighbors = 8))
knn_lab_model.fit(X_train, y_train)
print(knn_lab_model.score(X_test, y_test))
svc_rgb_model = SVC(kernel = 'rbf', C=100, gamma=1)
svc_rgb_model.fit(X_train, y_train)
print(svc_rgb_model.score(X_test, y_test))
svc_lab_model = make_pipeline(
    FunctionTransformer(rgbTolab),
    StandardScaler(),
    SVC(kernel = 'rbf', C=100, gamma=0.1)
)
svc_lab_model.fit(X_train, y_train)
print(svc_lab_model.score(X_test, y_test))


