# Load packages and dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

df.head()
# check missing values and data type

df.info()
df_clean = df.copy()
# Drop some unecessary columns. e.g. blueFirstblood/redfirst blood blueEliteMonster/redEliteMonster blueDeath/redKills etc are repeated

# Based on personal experience with the game, mimion yield gold+experience, we can drop minion kill too

cols = ['gameId', 'redFirstBlood', 'redKills', 'redEliteMonsters', 'redDragons','redTotalMinionsKilled',

       'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin', 'redHeralds',

       'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin', 'blueTotalMinionsKilled']

df_clean = df_clean.drop(cols, axis = 1)
df_clean.info()
# Next let's check the relationship between parameters of blue team features

g = sns.PairGrid(data=df_clean, vars=['blueKills', 'blueAssists', 'blueWardsPlaced', 'blueTotalGold'], hue='blueWins', size=3, palette='Set1')

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
# We can see that a lot of the features are highly correlated, let's get the correlation matrix

plt.figure(figsize=(16, 12))

sns.heatmap(df_clean.drop('blueWins', axis=1).corr(), cmap='YlGnBu', annot=True, fmt='.2f', vmin=0);
# Based on the correlation matrix, let's clean the dataset a little bit more to avoid colinearity

cols = ['blueAvgLevel', 'redWardsPlaced', 'redWardsDestroyed', 'redDeaths', 'redAssists', 'redTowersDestroyed',

       'redTotalExperience', 'redTotalGold', 'redAvgLevel']

df_clean = df_clean.drop(cols, axis=1)
# Next let's drop the columns has little correlation with bluewins

corr_list = df_clean[df_clean.columns[1:]].apply(lambda x: x.corr(df_clean['blueWins']))

cols = []

for col in corr_list.index:

    if (corr_list[col]>0.2 or corr_list[col]<-0.2):

        cols.append(col)

cols
df_clean = df_clean[cols]

df_clean.head()
df_clean.hist(alpha = 0.7, figsize=(12,10), bins=5);
# train test split scale the set

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

X = df_clean

y = df['blueWins']

scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



# fit the model

clf_nb = GaussianNB()

clf_nb.fit(X_train, y_train)



pred_nb = clf_nb.predict(X_test)



# get the accuracy score

acc_nb = accuracy_score(pred_nb, y_test)

print(acc_nb)
# fit the decision tree model

from sklearn import tree

from sklearn.model_selection import GridSearchCV



tree = tree.DecisionTreeClassifier()



# search the best params

grid = {'min_samples_split': [5, 10, 20, 50, 100]},



clf_tree = GridSearchCV(tree, grid, cv=5)

clf_tree.fit(X_train, y_train)



pred_tree = clf_tree.predict(X_test)



# get the accuracy score

acc_tree = accuracy_score(pred_tree, y_test)

print(acc_tree)
# fit the model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200,300,400,500], 'max_depth': [2, 5, 10]}



clf_rf = GridSearchCV(rf, grid, cv=5)

clf_rf.fit(X_train, y_train)



pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(pred_rf, y_test)

print(acc_rf)
# fit logistic regression model

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

lm.fit(X_train, y_train)



# get accuracy score

pred_lm = lm.predict(X_test)

acc_lm = accuracy_score(pred_lm, y_test)

print(acc_lm)
# fit the model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier() 



# search the best params

grid = {"n_neighbors":np.arange(1,100)}

clf_knn = GridSearchCV(knn, grid, cv=5)

clf_knn.fit(X_train,y_train) 



# get accuracy score

pred_knn = clf_knn.predict(X_test) 

acc_knn = accuracy_score(pred_knn, y_test)

print(acc_knn)
data_dict = {'Naive Bayes': [acc_nb], 'DT': [acc_tree], 'Random Forest': [acc_rf], 'Logistic Regression': [acc_lm], 'K_nearest Neighbors': [acc_knn]}

df_c = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Accuracy Score'])

print(df_c)
# recall and precision

from sklearn.metrics import recall_score, precision_score



# params for lm 

recall_lm = recall_score(pred_lm, y_test, average = None)

precision_lm = precision_score(pred_lm, y_test, average = None)

print('precision score for naive bayes: {}\n recall score for naive bayes:{}'.format(precision_lm, recall_lm))
# params for rf

recall_rf = recall_score(pred_rf, y_test, average = None)

precision_rf = precision_score(pred_rf, y_test, average = None)

print('precision score for naive bayes: {}\n recall score for naive bayes:{}'.format(precision_rf, recall_rf))
df_clean.columns
lm.coef_
np.exp(lm.coef_)
coef_data = np.concatenate((lm.coef_, np.exp(lm.coef_)),axis=0)

coef_df = pd.DataFrame(data=coef_data, columns=df_clean.columns).T.reset_index().rename(columns={'index': 'Var', 0: 'coef', 1: 'oddRatio'})

coef_df.sort_values(by='coef', ascending=False)
# try to visualize the results using PCA

X = df_clean

y = df['blueWins']



# PCA is affected by scale, scale the dataset first

from sklearn import preprocessing 

# Standardizing the features

X = preprocessing.StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

components = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
# create visulization df

df_vis = pd.DataFrame(data = components, columns = ['pc1', 'pc2'])

df_vis = pd.concat([df_vis, df['blueWins']], axis = 1)

X = df_vis[['pc1', 'pc2']]

y = df_vis['blueWins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# refit the pca data

lm.fit(X_train, y_train)
# visualize function

from matplotlib.colors import ListedColormap

def DecisionBoundary(clf):

    X = df_vis[['pc1', 'pc2']]

    y = df_vis['blueWins']

    

    h = .02  # step size in the mesh



    # Create color maps

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])

    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    

    #Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1

    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 8))

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    

    # Plot also the training points

    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_bold)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())

    plt.show()
DecisionBoundary(lm)