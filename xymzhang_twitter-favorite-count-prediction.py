import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
plus_minus = pd.read_csv("../input/nba_2017_real_plus_minus.csv")
wiki = pd.read_csv("../input/nba_2017_player_wikipedia.csv")
twitter = pd.read_csv("../input/nba_2017_twitter_players.csv")
plus_minus['Player'] = plus_minus['NAME'].apply(lambda x: x.split(',')[0])
plus_minus.head()
new_wiki = wiki[['names','pageviews']].groupby(by='names', as_index=False).mean()
new_wiki.head()
twitter.head()
data = pd.merge(twitter, new_wiki, how='inner', left_on='PLAYER', right_on='names')
data = pd.merge(data, plus_minus, how='inner', left_on='PLAYER', right_on='Player')

data2 = data[['TWITTER_FAVORITE_COUNT','pageviews','WINS']]
data2 = data2.dropna(how='any')
sns.distplot(data2['TWITTER_FAVORITE_COUNT'],hist=False)
sns.distplot((data2['TWITTER_FAVORITE_COUNT']**(1/3)),hist=False)
sns.distplot(data2['pageviews'],hist=False)
sns.distplot(np.log(data2['pageviews']),hist=False)
sns.distplot(data2['WINS'],hist=False)
data2['log_pv'] = np.log(data2['pageviews'])
data2['new_twitter'] = data2['TWITTER_FAVORITE_COUNT']**(1/3)
final_data = data2[['log_pv','WINS','new_twitter']]
X = final_data.iloc[:,:2].as_matrix()
y = final_data.iloc[:,2:].as_matrix()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

svr = SVR()
kernel = ['poly','rbf','linear']
epsilon = [0.1,0.2,0.3]

param_grid = dict(kernel=kernel, epsilon=epsilon)

grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_svr = pd.DataFrame(grid_result.cv_results_)
result_svr.sort_values(by='mean_test_score', ascending=False)
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
print(mean_squared_error(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
n_estimators = [i for i in range(50,350,10)]
max_depth = [i for i in range(4,10,1)]

param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)

grid_search = GridSearchCV(rfr, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_rfr = pd.DataFrame(grid_result.cv_results_)
result_rfr.sort_values(by='mean_test_score', ascending=False)
best_rfr = grid_search.best_estimator_
y_pred = best_rfr.predict(X_test)
print(mean_squared_error(y_test,y_pred))
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
p = [i for i in range(1,6)]
n_neighbors = [i for i in range(2,11)]

param_grid = dict(p=p, n_neighbors=n_neighbors)

grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_result = grid_search.fit(X_train, y_train)

result_knn = pd.DataFrame(grid_result.cv_results_)
result_knn.sort_values(by='mean_test_score', ascending=False)
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print(mean_squared_error(y_test,y_pred))
