import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

%matplotlib inline
df = pd.read_csv('../input/data.csv')
df.head()
df.drop(df.columns[[0,-1]], axis = 1, inplace = True)

df.head()
mean_para = df.columns[1:11]

se_para = df.columns[11:21]

worst_para = df.columns[21:]

diagnosis = df.columns[0]
print(mean_para)

print('\n')

print(se_para)

print('\n')

print(worst_para)
corr = df[mean_para].corr()

plt.figure(figsize=(8,8))

sns.heatmap(corr, annot = True)
para_use = ['radius_mean', 'texture_mean', 'smoothness_mean', 'symmetry_mean', 'fractal_dimension_mean']
df_mean_use = pd.concat([df[diagnosis], df[para_use]], axis = 1)
plt.figure(figsize=(8,8))

sns.pairplot(df_mean_use, hue = diagnosis)
X = df[df_mean_use.columns[1:]]

y = df[df_mean_use.columns[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = SVC(kernel='linear')

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))
score = cross_val_score(clf, X = X_train, y = y_train, cv = 10)

score.mean()
parameters = [{'C':[0.1, 1, 10], 'kernel':['linear']},

              {'C':[0.1, 1, 10], 'kernel':['rbf'], 'gamma':[1, 0.1, 0.01]}

             ]
grid_sh = GridSearchCV(clf, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs= -1)
grid_sh_result = grid_sh.fit(X_train, y_train)

grid_sh_result.best_params_
clf_adj = SVC(C = 10, kernel='linear')

clf_adj.fit(X_train, y_train)

pred_adj = clf_adj.predict(X_test)

print(classification_report(y_test, pred_adj))
scale = StandardScaler()

X_scale = pd.DataFrame(scale.fit_transform(X))

X_train_scale, X_test_scale, y_train_scale, y_test_scale = train_test_split(X_scale, y, test_size=0.3, random_state=1)

grid_sh_result_scale = grid_sh.fit(X_train_scale, y_train_scale)

grid_sh_result_scale.best_params_
clf_scale = SVC(C = 1, kernel = 'rbf', gamma = 0.1)

clf_scale.fit(X_train_scale, y_train_scale)

pred_sclae = clf_scale.predict(X_test_scale)

print(classification_report(y_test, pred_sclae))
clf_xg = XGBClassifier()

clf_xg.fit(X_train, y_train)

pred_xg = clf_xg.predict(X_test)

print(classification_report(y_test, pred_xg))