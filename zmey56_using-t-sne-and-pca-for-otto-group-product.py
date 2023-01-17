import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.manifold
import sklearn.decomposition
import sklearn.metrics
import numpy as np
import pandas as pd
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/train.csv")
df = data.drop("id", axis=1)
print(df.shape)
df['target'].value_counts()
df.info()
mapping = {'Class_2':2, 'Class_6':6, 'Class_8':8, 'Class_3':3, 'Class_9':9, 'Class_7':7, 'Class_4':4, 'Class_5':5, 'Class_1':1}      
#mapping = {'set': 1, 'test': 2}
df_cat = df.replace({'target': mapping})
df_cat.head()
y = df_cat['target'].values
X = df_cat.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
model_pca_trans = pca.fit_transform(X_train)
plt.figure(figsize=(20, 10))
label_color_dict = {label:idx for idx, label in enumerate(np.unique(y_train))}
cvec = [label_color_dict[label] for label in y_train]
plt.scatter(model_pca_trans[:, 0], model_pca_trans[:, 1], c=cvec, edgecolor='', alpha=0.2)
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2,  learning_rate=1000, init="random", random_state=1234).fit_transform(X_train)
plt.figure(figsize=(20, 10))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cvec, edgecolor='', alpha=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid_lr = {
    'C': np.power(10., np.arange(-4,4,1,dtype=int))
}
lr = LogisticRegression(solver='liblinear')
sg_lr = GridSearchCV(lr, param_grid_lr, scoring='neg_log_loss', cv=5)
sg_lr.fit(X_train, y_train)
# оптимальные параметры
print(sg_lr.best_params_, '\n')
print(sg_lr.best_score_, '\n')
print(sg_lr.best_estimator_, '\n')
# обучение случайного леса
param_grid_rf = {
    'max_depth': np.arange(2,48,2,dtype=int),
}
rf = RandomForestClassifier(n_estimators=50)
# поиск по сетке
sg_rf = GridSearchCV(rf, param_grid_rf, scoring='neg_log_loss', cv=5)
sg_rf.fit(X_train, y_train)
# оптимальные параметры
print(sg_rf.best_params_, '\n')
print(sg_rf.best_score_, '\n')
print(sg_rf.best_estimator_, '\n')
from sklearn import metrics
# линейная регрессия
print('LogisticRegression')
print('accuracy', metrics.accuracy_score(y_test, sg_lr.predict(X_test)))
# Случайный лес
print('RandomForestClassifier')
print('accuracy', metrics.accuracy_score(y_test, sg_rf.predict(X_test)))
pca_test = PCA(n_components=2)
pca_test.fit(X_test)
model_pca_trans_test = pca.fit_transform(X_test)
lr_pca = LogisticRegression(solver='liblinear')
sg_lr_pca = GridSearchCV(lr_pca, param_grid_lr, scoring='neg_log_loss', cv=5)
sg_lr_pca.fit(model_pca_trans, y_train)
# оптимальные параметры
print(sg_lr_pca.best_params_, '\n')
print(sg_lr_pca.best_score_, '\n')
print(sg_lr_pca.best_estimator_, '\n')
# линейная регрессия
print('LogisticRegression')
print('accuracy', metrics.accuracy_score(y_test, sg_lr_pca.predict(model_pca_trans_test)))
# обучение случайного леса
rf_pca = RandomForestClassifier(n_estimators=50)
# поиск по сетке
sg_rf_pca = GridSearchCV(rf_pca, param_grid_rf, scoring='neg_log_loss', cv=5)
sg_rf_pca.fit(model_pca_trans, y_train)
# оптимальные параметры
print(sg_rf_pca.best_params_, '\n')
print(sg_rf_pca.best_score_, '\n')
print(sg_rf_pca.best_estimator_, '\n')
# Случайный лес
print('RandomForestClassifier')
print('accuracy', metrics.accuracy_score(y_test, sg_rf_pca.predict(model_pca_trans_test)))
X_embedded_test = TSNE(n_components=2,  learning_rate=1000, init="random", random_state=1234).fit_transform(X_test)
lr_sne = LogisticRegression(solver='liblinear')
sg_lr_sne = GridSearchCV(lr_sne, param_grid_lr, scoring='neg_log_loss', cv=5)
sg_lr_sne.fit(X_embedded, y_train)
# оптимальные параметры
print(sg_lr_sne.best_params_, '\n')
print(sg_lr_sne.best_score_, '\n')
print(sg_lr_sne.best_estimator_, '\n')
# линейная регрессия
print('LogisticRegression')
print('accuracy', metrics.accuracy_score(y_test, sg_lr_sne.predict(X_embedded_test)))
# обучение случайного леса
rf_sne = RandomForestClassifier(n_estimators=50)
# поиск по сетке
sg_rf_sne = GridSearchCV(rf_sne, param_grid_rf, scoring='neg_log_loss', cv=5)
sg_rf_sne.fit(X_embedded, y_train)
# оптимальные параметры
print(sg_rf_pca.best_params_, '\n')
print(sg_rf_pca.best_score_, '\n')
print(sg_rf_pca.best_estimator_, '\n')
# Случайный лес
print('RandomForestClassifier')
print('accuracy', metrics.accuracy_score(y_test, sg_rf_sne.predict(X_embedded_test)))
