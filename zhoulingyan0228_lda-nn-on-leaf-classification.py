import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt
import seaborn as sns
data_train = pd.read_csv('../input/train.csv')
data_train.head()
data_train.drop(['id', 'species'], axis=1).describe()
data_train['species'].describe()
plt.subplots(figsize=(30,30))
corr_matrix = data_train.drop(['id', 'species'], axis=1).corr().abs()
sns.heatmap(corr_matrix);
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.995)]
feature_selected = data_train.drop(['id', 'species']+to_drop, axis=1)
plt.subplots(figsize=(30,30))
sns.heatmap(feature_selected.corr().abs());
featureScaler = StandardScaler()
featureScaler.fit(feature_selected)
feature_scaled = featureScaler.transform(feature_selected)
classifiers = [
    MLPClassifier(hidden_layer_sizes=(512,256,128), max_iter=500, tol=0.0000005),
    LinearDiscriminantAnalysis()
]
for clf in classifiers:
    print(type(clf))
    kfold = KFold(5)
    for train_indices, test_indices in kfold.split(data_train):
        clf.fit(feature_scaled[train_indices], data_train['species'].iloc[train_indices])
        print(clf.score(feature_scaled[test_indices], data_train['species'].iloc[test_indices]))
final_clf = MLPClassifier(hidden_layer_sizes=(512,256,128), max_iter=500, tol=0.0000005)
final_clf.fit(feature_scaled, data_train['species'])
data_test = pd.read_csv('../input/test.csv')
feature_test = featureScaler.transform(data_test.drop(['id']+to_drop, axis=1))
pd.concat([data_test[['id']], pd.DataFrame(final_clf.predict_proba(feature_test), columns=final_clf.classes_)], axis=1).to_csv('submission.csv', index=False)