import numpy as np 
import pandas as pd

# graphics
import seaborn as sns
import matplotlib.pyplot as plt

# plotly
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.express as px
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

games_source = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
print('data dimension:', games_source.shape)
print('\n')
print('column names:', list(games_source.columns))
print('\n')
print('missing values:', games_source.isnull().any().sum())
print('\n')
games_source.head(10)
# make a copy of data, set gameId to index
games = games_source.copy()
games.drop(columns='gameId', inplace=True)
games.iloc[0,:]
# heatmap
corr = games.corr()

plt.figure(figsize=(18,18))
sns.heatmap(corr, square=True)
plt.show()
games.drop(columns=['blueTotalGold', 'blueTotalExperience', 'blueTotalMinionsKilled', 'blueCSPerMin', 'blueGoldPerMin', 'blueAvgLevel'], inplace=True)
games.drop(columns=['redFirstBlood', 'redGoldDiff', 'redExperienceDiff', 'redKills', 'redDeaths', 'redAssists', 'redTotalGold', 
                    'redTotalExperience', 'redTotalMinionsKilled', 'redGoldPerMin', 'redAvgLevel', 'redCSPerMin'], inplace=True)
# heatmap
corr = games.corr()

plt.figure(figsize=(18,18))
sns.heatmap(corr, annot=True, square=True)
plt.show()
# count blueWins

plt.figure(figsize=(6,6))
sns.countplot(data=games, x='blueWins')
plt.title('Distribution of Blue Side win/lose', size=19)
plt.show()
# train, test split
X = games.iloc[:,1:]
y = games['blueWins']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
print('\n')
print('mean gold diff:', x_train['blueGoldDiff'].mean())
print('mean exp diff:', x_train['blueExperienceDiff'].mean())
# count blueWins
y_train_count = pd.DataFrame({'blueWins':y_train})
y_test_count = pd.DataFrame({'blueWins':y_test})

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,6))

sns.countplot(data=y_train_count, x='blueWins', ax=ax1)
ax1.set_title('Training set')
ax1.grid(False)

sns.countplot(data=y_test_count, x='blueWins',ax=ax2)
ax2.set_title('test set')
ax2.grid(False)

plt.suptitle('Distribution of Blue Side win/lose', size=20)
plt.show()
model = RandomForestClassifier(random_state=10)
model.fit(x_train, y_train)
model.score(x_test, y_test)
important_features = pd.DataFrame({'feature':x_train.columns, 'importance_score':model.feature_importances_})
important_features.sort_values(by='importance_score', ascending=False, inplace=True)

plt.figure(figsize=(24,6))
p = sns.barplot(data=important_features, x='feature', y='importance_score')

plt.xticks(rotation=65)
plt.show()
# # set range of parameters for tuning
# n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=79)]
# criterion = ['gini', 'entropy']
# max_features = ['auto','sqrt','log2',None]
# max_depth = [int(x) for x in np.linspace(start=5, stop=25, num=20)]
# max_depth.append(None)
# min_samples_split = [2,5,8]
# min_samples_leaf = [1, 2, 4]

# parameters = {'n_estimators' : n_estimators,
#                  'criterion' : criterion,
#                  'max_features' : max_features,
#                  'max_depth' : max_depth,
#                  'min_samples_split' : min_samples_split,
#                  'min_samples_leaf' : min_samples_split}


# rdm_for = RandomForestClassifier()
# rdm_CV = RandomizedSearchCV(rdm_for, parameters, cv=5, n_iter = 10, verbose=1)
# rdm_CV.fit(X,y)

# rdm_CV.best_params_
# # {'n_estimators': 950,
# #  'min_samples_split': 8,
# #  'min_samples_leaf': 1,
# #  'max_features': 'auto',
# #  'max_depth': 7,
# #  'criterion': 'gini'}
model = RandomForestClassifier(n_estimators=950, 
                               min_samples_split=8, 
                               min_samples_leaf=1, 
                               max_features='auto',
                              max_depth=7,
                              criterion='gini',
                              random_state=10)
model.fit(x_train, y_train)
model.score(x_test, y_test)

# logistice regression
clf = LogisticRegression()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
# set range of parameters for tuning
C = np.arange(0.00001, 5, 0.1)
penalty = ['l1', 'l2', 'elasticnet', 'none']


parameters = {'C' : C,'penalty' : penalty}


clf = LogisticRegression()
clf_CV = RandomizedSearchCV(clf, parameters, cv=5, n_iter = 10, verbose=1)
clf_CV.fit(X,y)

# clf_CV.best_params_
# {'penalty': 'l2', 'C': 4.50001}
# logistice regression after fine tuning
clf = LogisticRegression(penalty='l2', C=4.5)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
# make a copy of data, set gameId to index
games = games_source.copy()
games.drop(columns='gameId', inplace=True)

X = games.iloc[:,1:]
y = games['blueWins']


# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X_scaled)

# var ratio
variance_ratio = pd.DataFrame({'Component':range(0,25), 'Percentage_explained_variance':pca.explained_variance_ratio_})
variance_ratio

plt.figure(figsize=(24,6))
sns.barplot(data=variance_ratio, x='Component', y='Percentage_explained_variance')
plt.title('Percentage of Variance Explained', size=25)
plt.show()

# cumulative sum
cumsum_ratio = pd.DataFrame({'Component':range(0,25), 'Cumulative_sum':pca.explained_variance_ratio_.cumsum()})

plt.figure(figsize=(24,6))
sns.set_style("whitegrid")
sns.lineplot(data=cumsum_ratio, x='Component', y='Cumulative_sum')
plt.title('Cumulative Sum of Variance Explained', size=25)
plt.xticks(np.arange(0,26, step=1))
plt.show()
aa = pd.DataFrame({'feature':X.columns, 'eigenvalue': abs(pca.components_[0])})
aa.sort_values(by='eigenvalue', ascending=False).head(10)
aa = pd.DataFrame({'feature':X.columns, 'eigenvalue': abs(pca.components_[1])})
aa.sort_values(by='eigenvalue', ascending=False).head(10)
aa = pd.DataFrame({'feature':X.columns, 'eigenvalue': abs(pca.components_[2])})
aa.sort_values(by='eigenvalue', ascending=False).head(10)
aa = pd.DataFrame({'feature':X.columns, 'eigenvalue': abs(pca.components_[3])})
aa.sort_values(by='eigenvalue', ascending=False).head(10)
new_x = pca.transform(X_scaled)
new_x = pd.DataFrame(new_x)
new_x
# train, test split
X = new_x.copy()
y = games['blueWins']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
model = RandomForestClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)
