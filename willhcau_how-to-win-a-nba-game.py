#Import required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import metrics

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

from sklearn.feature_selection import mutual_info_regression

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve
df_player = pd.read_csv("../input/nba17-18/nba_extra.csv", index_col=0)                            

df_player.head(3)
df_team_standing = pd.read_csv("../input/nba-enhanced-stats/2017-18_standings.csv", index_col=0)

df_team_standing.head(5)
#Team name abbreviations according to Wiki:

#https://en.wikipedia.org/wiki/Wikipedia:WikiProject_National_Basketball_Association/National_Basketball_Association_team_abbreviations



# Some of the team name between the tables are different, convert them so that they can match each other.

df_player.replace(to_replace='BRK', value='BKN', inplace=True)

df_player.replace(to_replace='TOT', value='TOR', inplace=True)

df_player.replace(to_replace='CHO', value='CHA', inplace=True)



df_team_standing.replace(to_replace='SA', value='SAS', inplace=True)

df_team_standing.replace(to_replace='NO', value='NOP', inplace=True)

df_team_standing.replace(to_replace='NY', value='NYK', inplace=True)

df_team_standing.replace(to_replace='GS', value='GSW', inplace=True)
# Capture the number of winning game for each team

df_team_num_of_winning = df_team_standing[['teamAbbr', 'gameWon']]

df_team_num_of_winning = df_team_num_of_winning.groupby('teamAbbr').max().sort_values(['gameWon'], ascending=False)



# Rename dataframe index

df_team_num_of_winning.index.names = ['Tm']



# Print out the top 5 teams have the most winning

df_team_num_of_winning.head(5)
df_player['win'] = df_player['Tm'].map(df_team_num_of_winning['gameWon'].to_dict())

df_player.head(5)

df_player.isnull().sum()[df_player.isnull().sum() > 0]

df_player.fillna(0, inplace=True)
le_pos = LabelEncoder()

le_pos.fit(df_player['Pos'])

df_player['Pos'] = le_pos.transform(df_player['Pos'])



le_team = LabelEncoder()

le_team.fit(df_player['Tm'])

df_player['Tm'] = le_team.transform(df_player['Tm'])
df_player_name = df_player['Player']

df_player = df_player.drop(columns=['Player'])
df_player.head(5)
f, ax = plt.subplots(figsize =(10, 10)) 

sns.heatmap(df_player.corr(), ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
df_player.corr()['win'].sort_values(ascending=False).iloc[1:].head(10)
selected_features_name = df_player.corr()['win'].sort_values(ascending=False).iloc[1:].head(10).index

corr_matrix_selected = df_player[selected_features_name]
display(corr_matrix_selected.head(5))
# Training set features & outputs

all_x = df_player.iloc[:, :-1]

all_y = df_player.iloc[:, -1]



f_test_selected = pd.DataFrame()

f_test_score = pd.DataFrame()



# Select data using f_test

def select_features(data_x, data_y, select_algo, select_size):

    k_best_selector = SelectKBest(select_algo, k=select_size)



    selected_features = k_best_selector.fit_transform(data_x, data_y)

    selected_fratures_scores = k_best_selector.scores_



    selected_features_name = []



    for selected, feature in zip(k_best_selector.get_support(), data_x.columns):

        if selected:

            selected_features_name.append(feature)



    selected_data = pd.DataFrame(selected_features, columns=selected_features_name)

    selected_fratures_scores = pd.DataFrame([selected_fratures_scores], columns=data_x.columns, index=['scores'])

    

    return selected_data, selected_fratures_scores



# Obtain selected data by f-test

f_test_selected, f_test_score = select_features(all_x, all_y, f_regression, 10)



# Obtain selected data by mutual info

mutual_info_selected, mutual_info_score = select_features(all_x, all_y, mutual_info_regression, 10)



# Print selected feature between two algorithm

print("By F-test")

print(f_test_selected.columns)

print("Bu mutual information")

print(mutual_info_selected.columns)
display(f_test_selected.head(5).style.hide_index())

display(f_test_score.T.sort_values('scores', ascending=False).head(10))
display(mutual_info_selected.head(5).style.hide_index())

display(mutual_info_score.T.sort_values('scores', ascending=False).head(10))
train_test_ration = 0.8



# Features of 4 different approaches

train_x, test_x = train_test_split(all_x, random_state=1, train_size=train_test_ration)

corr_matrix_train_x, corr_matrix_test_x = train_test_split(corr_matrix_selected, random_state=1, train_size=train_test_ration)

f_test_train_x, f_test_test_x = train_test_split(f_test_selected, random_state=1, train_size=train_test_ration)

mutual_info_train_x, mutual_info_test_x = train_test_split(mutual_info_selected, random_state=1, train_size=train_test_ration)



# Prediction target

train_y, test_y = train_test_split(all_y, random_state=1, train_size=train_test_ration)
# Create linear regression object

lr = LinearRegression(fit_intercept=True, normalize=True)

lr_corr_matrix = LinearRegression(fit_intercept=True, normalize=True)

lr_f_test = LinearRegression(fit_intercept=True, normalize=True)

lr_mutual_info = LinearRegression(fit_intercept=True, normalize=True)



# Train the model using the training sets

lr.fit(train_x, train_y)

lr_corr_matrix.fit(corr_matrix_train_x, train_y)

lr_f_test.fit(f_test_train_x, train_y)

lr_mutual_info.fit(mutual_info_train_x, train_y)
print('R2 score (All features):       %.3f' % lr.score(train_x, train_y))

print('R2 score (Correlation Matrix): %.3f' % lr_corr_matrix.score(corr_matrix_train_x, train_y))

print('R2 score (F-test):             %.3f' % lr_f_test.score(f_test_train_x, train_y))

print('R2 score (Mutual Information): %.3f' % lr_mutual_info.score(mutual_info_train_x, train_y))
# Prediction on training set to check the bias of the model

pred_y = lr.predict(train_x)

corr_matrix_pred_y = lr_corr_matrix.predict(corr_matrix_train_x)

f_test_pred_y = lr_f_test.predict(f_test_train_x)

mutual_info_pred_y = lr_mutual_info.predict(mutual_info_train_x)



print('Performance metrics over train data:')

print('(All Features)       Mean Absolute Error:', metrics.mean_absolute_error(train_y, pred_y))

print('(Correlation matrix) Mean Absolute Error:', metrics.mean_absolute_error(train_y, corr_matrix_pred_y))

print('(F-test)             Mean Absolute Error:', metrics.mean_absolute_error(train_y, f_test_pred_y))

print('(Mutual Information) Mean Absolute Error:', metrics.mean_absolute_error(train_y, mutual_info_pred_y))

print('')



# Prediction on the test to see how it variance

pred_y = lr.predict(test_x)

corr_matrix_pred_y = lr_corr_matrix.predict(corr_matrix_test_x)

f_test_pred_y = lr_f_test.predict(f_test_test_x)

mutual_info_pred_y = lr_mutual_info.predict(mutual_info_test_x)



print('Performance metrics over test data:')

print('(All Features)       Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred_y))

print('(Correlation matrix) Mean Absolute Error:', metrics.mean_absolute_error(test_y, corr_matrix_pred_y))

print('(F-test)             Mean Absolute Error:', metrics.mean_absolute_error(test_y, f_test_pred_y))

print('(Mutual Information) Mean Absolute Error:', metrics.mean_absolute_error(test_y, mutual_info_pred_y))
#Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



plot_learning_curve(lr, "The learning curve", train_x, train_y)
df_player_name = df_player_name.reset_index().drop(columns=['Rk'])

df_player_stat = df_player.reset_index().drop(columns=['Rk'])
player_index = df_player_name[df_player_name['Player'].str.contains("James Harden")].index

player_stat = df_player_stat.iloc[player_index.values,:]



pd.options.display.max_columns = None

pd.options.display.max_rows = None

display(player_stat)
# Cut of James Harden free throw attempt by half from 727 to 350

player_stat[player_stat.index == player_index.values]['FTA'] = 350



# While keeping his FT% at 0.858, so that 350 * 0.858 = 300

player_stat[player_stat.index == player_index.values]['FT'] = 300

print("If James Harden has only 350 free throw attempt, the number of game he would win: ", lr.predict(player_stat.iloc[:, :-1])[0])