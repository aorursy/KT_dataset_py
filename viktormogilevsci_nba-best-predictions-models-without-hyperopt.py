import pandas as pd

import numpy as np



from sklearn import metrics



pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)



import pickle

import warnings

warnings.filterwarnings('ignore')
raw_game_data = pd.read_csv('../input/2012-18_teamBoxScore.csv')

raw_game_data
game_data = raw_game_data[1::2].reset_index(drop=True)

game_data.columns = map(str.upper, game_data.columns)

game_data['TEAMRSLT'] = game_data.apply(lambda x: 1 if x['TEAMRSLT'] == 'Win' else 0, axis=1)

columns = ['TEAMRSLT', 'TEAMFGM', 'TEAMFGA', 'TEAMFG%', 'TEAM3PM', 'TEAM3PA', 'TEAM3P%', 'TEAMFTM',

           'TEAMFTA', 'TEAMFT%', 'TEAMOREB', 'TEAMDREB', 'TEAMREB', 'TEAMAST', 'TEAMTOV', 'TEAMSTL',

           'TEAMBLK', 'TEAMPF', 'OPPTFGM', 'OPPTFGA', 'OPPTFG%', 'OPPT3PM', 'OPPT3PA', 

           'OPPT3P%', 'OPPTFTM', 'OPPTFTA', 'OPPTFT%', 'OPPTOREB', 'OPPTDREB', 'OPPTREB', 'OPPTAST', 'OPPTTOV',

           'OPPTSTL', 'OPPTBLK', 'OPPTPF']

game_data = game_data.filter(columns)

game_data
game_data.fillna(0, inplace = True)

game_data.isnull().sum().max()
from sklearn.decomposition import PCA



n_components = game_data.shape[1]

pca = PCA(n_components = n_components)

pca.fit(game_data)



explained_variance_ratio = pca.explained_variance_ratio_ 

cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

components_ = pca.components_ 

lst = []

for i in range (0, n_components):

    lst.append([i+1, round(explained_variance_ratio[i],6), cum_explained_variance_ratio[i]])

pca_predictor = pd.DataFrame(lst)

pca_predictor.columns = ['Component', 'Explained Variance', 'Cumulative Explained Variance']

pca_predictor
game_data_x = game_data.drop(columns=['TEAMRSLT'])

game_data_y = game_data['TEAMRSLT']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(game_data_x, game_data_y, test_size=0.2, random_state=2)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
#knn 

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)



metrics.accuracy_score(y_test, knn_pred)
#linear svm

from sklearn.svm import LinearSVC



svc_linear = LinearSVC()

svc_linear.fit(x_train, y_train)

pred_svc = svc_linear.predict(x_test)



metrics.accuracy_score(y_test, pred_svc)
#random forrest classifier

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier()

random_forest.fit(x_train, y_train)

pred_random_forest = random_forest.predict(x_test)



metrics.accuracy_score(y_test, pred_random_forest)
# Gradient Treee Boosting

from sklearn.ensemble import GradientBoostingClassifier



gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

gradient_boosting = gradient_boosting.fit(x_train, y_train)

pred_gradient_boosting = gradient_boosting.predict(x_test)



metrics.accuracy_score(y_test, pred_gradient_boosting)
# ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier



extra_tree = ExtraTreesClassifier(n_estimators=100)

extra_tree = extra_tree.fit(x_train, y_train)

pred_extra_tree = extra_tree.predict(x_test)



metrics.accuracy_score(y_test, pred_extra_tree)
# MLPClassifier

from sklearn.neural_network import MLPClassifier



MLP_classifier = MLPClassifier()

MLP_classifier = MLP_classifier.fit(x_train, y_train)

pred_MLP_classifier = MLP_classifier.predict(x_test)



metrics.accuracy_score(y_test, pred_MLP_classifier)
# SVC

from sklearn.svm import SVC



linear_SVC = SVC()

linear_SVC = linear_SVC.fit(x_train, y_train)

pred_linear_SVC = linear_SVC.predict(x_test)



metrics.accuracy_score(y_test, pred_linear_SVC)
filename = 'nba_pred_modelv1.sav'

pickle.dump(gradient_boosting, open(filename, 'wb'))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

game_data_scaled_x = pd.DataFrame(scaler.fit_transform(game_data_x))

game_data_scaled_x.columns = game_data_x.columns
pca = PCA(n_components=15)

pca = pca.fit(game_data_scaled_x)

stats_transformed = pca.fit_transform(game_data_scaled_x)

stats_transformed.shape
x_train, x_test, y_train, y_test = train_test_split(stats_transformed, game_data_y, test_size=0.2, random_state=2)
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

gradient_boosting = gradient_boosting.fit(x_train, y_train)

pred_gradient_boosting = gradient_boosting.predict(x_test)



metrics.accuracy_score(y_test, pred_gradient_boosting)
linear_SVC = SVC()

linear_SVC = linear_SVC.fit(x_train, y_train)

pred_linear_SVC = linear_SVC.predict(x_test)



metrics.accuracy_score(y_test, pred_linear_SVC)
filename = 'nba_pred_modelv1.sav'

pickle.dump(gradient_boosting, open(filename, 'wb'))

pickle.dump(pca, open('pca', 'wb'))
games_stat = raw_game_data.copy()

games_stat.columns = map(str.upper, games_stat.columns)

games_stat['TEAMRSLT'] = games_stat.apply(lambda x: 1 if x['TEAMRSLT'] == 'Win' else 0, axis=1)



id = 1;

game_ids = [];

for i in range(0, len(games_stat), 2):

    game_ids.append(id)

    game_ids.append(id)

    id += 1



games_stat['GAME_ID'] = game_ids



importance_columns = ['TEAMFGM', 'TEAMFGA', 'TEAMFG%', 'TEAM3PM', 'TEAM3PA', 'TEAM3P%', 'TEAMFTM', 

                      'TEAMFTA', 'TEAMFT%', 'TEAMAST', 'TEAMSTL', 'TEAMBLK', 'TEAMPF', 

                      'OPPTFGM', 'OPPTFGA', 'OPPTFG%', 'OPPT3PM', 'OPPT3PA', 'OPPT3P%', 'OPPTFTM',

                      'OPPTFTA', 'OPPTFT%', 'OPPTAST', 'OPPTSTL', 'OPPTBLK', 'OPPTPF']



def get_columns_mean(columns, data_frame_describe, data_frame):

    for column in columns:

        mean = data_frame_describe[column]['mean']

        data_frame[column] = round(mean, 5)

    

def get_teams_mean(game_id, home_team, away_team):

    columns = ['TEAMFGM', 'TEAMFGA', 'TEAMFG%', 'TEAM3PM', 'TEAM3PA', 'TEAM3P%', 'TEAMFTM',

           'TEAMFTA', 'TEAMFT%', 'TEAMOREB', 'TEAMDREB', 'TEAMREB', 'TEAMAST', 'TEAMTOV', 'TEAMSTL',

           'TEAMBLK', 'TEAMPF']

    

    opposite_column = ['OPPTFGM', 'OPPTFGA', 'OPPTFG%', 'OPPT3PM', 'OPPT3PA', 'OPPT3P%', 'OPPTFTM', 

            'OPPTFTA', 'OPPTFT%', 'OPPTOREB', 'OPPTDREB', 'OPPTREB', 'OPPTAST', 'OPPTTOV',  'OPPTSTL', 

            'OPPTBLK', 'OPPTPF']

        

    HOME = games_stat.loc[(games_stat['GAME_ID'] < game_id) & (games_stat['TEAMABBR'] == home_team), :][-5:]

    AWAY = games_stat.loc[(games_stat['GAME_ID'] < game_id) & (games_stat['TEAMABBR'] == away_team), :][-5:]



    HOME = HOME.filter(columns)

    AWAY = AWAY.filter(columns)



    get_columns_mean(HOME.columns, HOME.describe(), HOME)

    HOME = HOME.iloc[-1:,:]



    get_columns_mean(AWAY.columns, AWAY.describe(), AWAY)

    AWAY = AWAY.iloc[-1:,:]



    rename_column = dict()

    for i in range(len(columns)):

        rename_column[columns[i]] = opposite_column[i]

    

    AWAY.rename(columns=rename_column, inplace=True)

    

    HOME['key'] = 1

    AWAY['key'] = 1

    AWAY_HOME = pd.merge(HOME, AWAY, how='outer')

    del AWAY_HOME['key']



    AWAY_HOME = AWAY_HOME.filter(importance_columns)

    AWAY_HOME = list(AWAY_HOME.iloc[0,:])

    return AWAY_HOME
# get_teams_mean(1, 'CLE', 'NY')



games = []

results = []



for step in range(50, len(games_stat), 2):

    team_home = games_stat.iloc[step + 1]

    team_away = games_stat.iloc[step]

    game_id = team_home['GAME_ID']

    team_home_name = team_home['TEAMABBR']

    team_away_name = team_away['TEAMABBR']

    result = team_home['TEAMRSLT']



    game = get_teams_mean(game_id, team_home_name, team_away_name)

    games.append(game)

    results.append(result)
game_data_training = pd.DataFrame(np.array(games), columns=importance_columns)

game_data_result = pd.DataFrame(np.array(results), columns=['TEAMRSLT'])

game_data_result = game_data_result['TEAMRSLT']
game_data_training.fillna(0, inplace = True)

game_data_training.isnull().sum().max()



scaler = MinMaxScaler()

scaler = scaler.fit(game_data_training)

game_data_scaled_training = pd.DataFrame(scaler.transform(game_data_training))

game_data_scaled_training.columns = game_data_training.columns



pca = PCA(n_components=15)

pca = pca.fit(game_data_training)

stats_transformed = pca.fit_transform(game_data_scaled_training)

stats_transformed.shape
x_train, x_test, y_train, y_test = train_test_split(stats_transformed, game_data_result, test_size=0.2, random_state=2)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# linear svm



svc_linear = LinearSVC()

svc_linear.fit(x_train, y_train)

pred_svc = svc_linear.predict(x_test)



metrics.accuracy_score(y_test, pred_svc)
#random forrest classifier



random_forest = RandomForestClassifier()

random_forest = random_forest.fit(x_train, y_train)

pred_random_forest = random_forest.predict(x_test)



metrics.accuracy_score(y_test, pred_random_forest)
# Gradient Treee Boosting



gradient_boosting = GradientBoostingClassifier(max_depth=2)

gradient_boosting = gradient_boosting.fit(x_train, y_train)

pred_gradient_boosting = gradient_boosting.predict(x_test)



metrics.accuracy_score(y_test, pred_gradient_boosting)
# ExtraTreesClassifier



extra_tree = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2)

extra_tree = extra_tree.fit(x_train, y_train)

pred_extra_tree = extra_tree.predict(x_test)



metrics.accuracy_score(y_test, pred_extra_tree)
# MLPClassifier



MLP_classifier = MLPClassifier(hidden_layer_sizes=10000, alpha=1, max_iter=500000000)

MLP_classifier = MLP_classifier.fit(x_train, y_train)

pred_MLP_classifier = MLP_classifier.predict(x_test)



metrics.accuracy_score(y_test, pred_MLP_classifier)
# SVC



linear_SVC = SVC()

linear_SVC = linear_SVC.fit(x_train, y_train)

pred_linear_SVC = linear_SVC.predict(x_test)



metrics.accuracy_score(y_test, pred_linear_SVC)
# AdaBoostClassifier



from sklearn.ensemble import AdaBoostClassifier



adaBoost_classifier = AdaBoostClassifier()

adaBoost_classifier = adaBoost_classifier.fit(x_train, y_train)

pred_adaBoost_classifier = adaBoost_classifier.predict(x_test)



metrics.accuracy_score(y_test, pred_adaBoost_classifier)
pickle.dump(svc_linear, open('svc_linear.sav', 'wb'))

pickle.dump(random_forest, open('random_forest.sav', 'wb'))

pickle.dump(gradient_boosting, open('gradient_boosting.sav', 'wb'))

pickle.dump(extra_tree, open('extra_tree.sav', 'wb'))

pickle.dump(MLP_classifier, open('MLP_classifier.sav', 'wb'))

pickle.dump(linear_SVC, open('linear_SVC.sav', 'wb'))



pickle.dump(pca, open('pca', 'wb'))

pickle.dump(scaler, open('scaler', 'wb'))