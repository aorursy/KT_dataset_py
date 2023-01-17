# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
print('Data Size:', data.shape)
print('Any null values?:', data.isnull().sum().where(lambda row: row > 0).dropna().to_dict())
print('Feature Statistics:', '\n', data.describe().T.iloc[:,1:])

# remove highly correlated features
DROP_THRESHOLD = 0.9
corr_matrix = data.corr()
one_sided_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_cols = [col for col in one_sided_corr.columns if any(one_sided_corr[col] > DROP_THRESHOLD)]
print('Columns to drop (highly correlated):', drop_cols)
# convert categorical string labels into one-hot encoded vectors (or binary label for output) 
from sklearn.preprocessing import LabelEncoder
data['class'] = LabelEncoder().fit_transform(data['class'])
data = pd.get_dummies(data)
# scale input "data" for better linear relationship
from sklearn.preprocessing import StandardScaler
y = data['class']
data = pd.DataFrame(StandardScaler().fit_transform(data.drop('class', axis=1)), columns=data.drop('class', axis=1).columns.values)
data['class'] = y
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random

data_norm = pd.DataFrame(normalize(data), columns=data.columns.values)
kmeans = [KMeans(n_clusters=i+2, random_state=random.randrange(43)).fit(data_norm) for i in range(10)]
scores = [silhouette_score(data_norm, km.labels_, metric='cosine') for km in kmeans]
optimal_score = max(scores)
clustering_index = scores.index(optimal_score)

print('How many clusters is optimal?', clustering_index+2)
print('Clustering Score:', optimal_score)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

data_norm = pd.DataFrame(MinMaxScaler().fit_transform(data_norm), columns=data_norm.columns.values)
pca = PCA(n_components=0.95, svd_solver='full').fit(data_norm)
data_norm_pca = pca.transform(data_norm)

with plt.style.context('dark_background'):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    sns.scatterplot(ax=ax[0], x=data_norm_pca[:,0], y=data_norm_pca[:,1], hue=kmeans[clustering_index].labels_, palette='Set2')
    ax[1].plot(np.arange(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
    plt.axis('tight')
    ax[0].set_xlabel('PC0')
    ax[0].set_ylabel('PC1')
    ax[1].set_xlabel('Number of Components')
    ax[1].set_ylabel('Cumulative Explained Variance (%)')
    ax[0].get_legend().remove()
# extract most important feature variances for each cluster (by variance)
data_norm['cluster'] = kmeans[clustering_index].labels_
data_norm_cluster_means = data_norm.groupby('cluster').mean()
data_norm_feature_variances = pd.DataFrame([[col, np.var(data_norm_cluster_means[col])] for col in data_norm_cluster_means.columns[1:]], columns=['feature', 'variance'])
features = list(data_norm_feature_variances.sort_values('variance', ascending=False).head(7)['feature'].values)
feature_variances = data_norm[features + ['cluster']].melt(id_vars='cluster')

# extract feature importances with random forest classifier (cluster label is output!)
from sklearn.ensemble import RandomForestClassifier
X, y = data_norm.drop('cluster', axis=1), data_norm['cluster']
clf = RandomForestClassifier(n_estimators=100).fit(X,y)
importances = pd.DataFrame(np.array([clf.feature_importances_, X.columns]).T, columns=['importance', 'feature'])
features = list(importances.sort_values('importance', ascending=False).head(7)['feature'].values)
feature_importances = data_norm[features + ['cluster']].melt(id_vars='cluster')

with plt.style.context('dark_background'):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,10))
    sns.barplot(ax=ax[0], x='cluster', y='value', hue='variable', data=feature_importances)
    sns.barplot(ax=ax[1], x='cluster', y='value', hue='variable', data=feature_variances)
    plt.axis('tight')
    ax[0].set_title('Top Random Forest Feature Importances to k-Means Clustering Labels')
    ax[1].set_title('Top Feature Variances to k-Means Clustering Labels')
    plt.xlabel('Cluster')
# split "data" into input and output (X, y)
X = data.drop('class', axis=1)
y = data['class'] # this is the "class" feature as output
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import classification_report, roc_curve, auc

def predict(model):
    print('Best Parameters:', model.best_params_)
    print('Best CV Accuracy Score:', model.best_score_)
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = np.where(y_prob > 0.5, 1, 0)
    model.score(X_test, y_pred)

    class_rept = classification_report(y_test, y_pred)
    fp_rate, tp_rate, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fp_rate, tp_rate)
    
    print('Model Classification Report:', '\n', class_rept)
    with plt.style.context('dark_background'):
        plt.plot(fp_rate, tp_rate, color='red', linestyle='--', label=f'AUC = {roc_auc:.2f}')
        plt.axis('tight')
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

ESTIMATORS = {
    'lr':{
        'model':LogisticRegression,
        'params':[
            {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
             'penalty':['l2'], 
             'solver':['lbfgs']},
            {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 
             'penalty':['l1'], 
             'solver':['liblinear']}]},
    'nb':{
        'model':GaussianNB,
        'params':{
            'var_smoothing':np.logspace(0,-9,num=100)}},
    'svm':{
        'model':SVC,
        'params':[
            {'C':[1, 10, 100, 500, 1000], 
             'kernel':['linear', 'rbf']},
            {'C':[1, 10, 100, 500, 1000], 
             'gamma':[1, 0.1, 0.01, 0.001, 0.0001], 
             'kernel':['rbf']},
            {'C':[1, 10, 100, 500, 1000], 
             'degree':[2, 3, 4, 5, 6], 
             'kernel':['poly']}]},
    'rf':{
        'model':RandomForestClassifier,
        'params':{
            'n_estimators':range(10,100,10),
            'min_samples_leaf':range(10,100,10),
            'max_depth':range(5,15,5),
            'max_features':['auto', 'sqrt', 'log2']}},
    'tree':{
        'model':DecisionTreeClassifier,
        'params':{
            'criterion':['gini', 'entropy'],
            'max_features':['auto', 'sqrt', 'log2'],
            'min_samples_leaf':range(1,100,1),
            'max_depth':range(1,50,1)}},
    'mlp':{
        'model':MLPClassifier,
        'params':{
            'hidden_layer_sizes':range(1,200,10),
            'activation':['tanh', 'logistic', 'relu'],
            'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10],
            'max_iter':range(50,200,50)}}}

for estimator in ESTIMATORS.values():
    grid = RandomizedSearchCV(
        estimator['model'](), 
        estimator['params'], 
        cv=10, scoring='accuracy', n_jobs=-1, n_iter=20)
    grid.fit(X_train, y_train)
    predict(grid)