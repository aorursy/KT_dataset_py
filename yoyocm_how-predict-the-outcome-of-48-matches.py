import featuretools as ft
import pandas as pd 
import numpy as np
import sqlite3
path = "../input/"  #Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)
matches_df = pd.read_sql("""SELECT * from MATCH""", conn)
teams_df = pd.read_sql("""SELECT * from TEAM""", conn)
player_attributes_df = pd.read_sql("""SELECT * from PLAYER_ATTRIBUTES""", conn)

matches_df['date'] = pd.to_datetime(matches_df['date'], format='%Y-%m-%d 00:00:00')
home_players = ["home_player_" + str(x) for x in range(1, 12)]
away_players = ["away_player_" + str(x) for x in range(1, 12)]

betting_columns = ["B365H", "B365D", "B365A"]

matches_kept_columns = ["id", "date", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"]
matches_kept_columns = matches_kept_columns + home_players
matches_kept_columns = matches_kept_columns + away_players
matches_kept_columns = matches_kept_columns + betting_columns

matches_df = matches_df[matches_kept_columns]
matches_df['goal_difference'] = matches_df['home_team_goal'] - matches_df['away_team_goal']
matches_df['home_status'] = 'D'
matches_df['home_status'] = np.where(matches_df['goal_difference'] > 0, 'W', matches_df['home_status'])
matches_df['home_status'] = np.where(matches_df['goal_difference'] < 0, 'L', matches_df['home_status'])

for player in home_players:
    matches_df = pd.merge(matches_df, player_attributes_df[["id", "overall_rating"]], left_on=[player], right_on=["id"], suffixes=["", "_" + player])
for player in away_players:
    matches_df = pd.merge(matches_df, player_attributes_df[["id", "overall_rating"]], left_on=[player], right_on=["id"], suffixes=["", "_" + player])
    
matches_df = matches_df.rename(columns={"overall_rating": "overall_rating_home_player_1"})

matches_df = matches_df[ matches_df[['overall_rating_' + p for p in home_players]].isnull().sum(axis = 1) <= 0]
matches_df = matches_df[ matches_df[['overall_rating_' + p for p in away_players]].isnull().sum(axis = 1) <= 0]

matches_df['overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].sum(axis=1)
matches_df['overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].sum(axis=1)
matches_df['overall_rating_difference'] = matches_df['overall_rating_home'] - matches_df['overall_rating_away']

matches_df['min_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].min(axis=1)
matches_df['min_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].min(axis=1)

matches_df['max_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].max(axis=1)
matches_df['max_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].max(axis=1)

matches_df['mean_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].mean(axis=1)
matches_df['mean_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].mean(axis=1)

matches_df['std_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].std(axis=1)
matches_df['std_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].std(axis=1)
for c in matches_df.columns:
    if '_player_' in c:
        matches_df = matches_df.drop(c, axis=1)
ct_home_matches = pd.DataFrame()
ct_away_matches = pd.DataFrame()

ct_matches = pd.DataFrame()

# Trick to exclude current match from statistics and do not biais predictions
ct_home_matches['time'] = matches_df['date'] - pd.Timedelta(hours=1)
ct_home_matches['instance_id'] = matches_df['home_team_api_id']
ct_home_matches['label'] = (ct_home_matches['instance_id'] == ct_home_matches['instance_id'])

# Trick to exclude current match from statistics and do not biais predictions
ct_away_matches['time'] = matches_df['date'] - pd.Timedelta(hours=1)
ct_away_matches['instance_id'] = matches_df['away_team_api_id']
ct_away_matches['label'] = (ct_away_matches['instance_id'] == ct_away_matches['instance_id'])

ct_matches = ct_home_matches.append(ct_away_matches)
es = ft.EntitySet("entityset")

es.entity_from_dataframe(entity_id="home_matches",
                        index="id",
                        time_index="date",
                        dataframe=matches_df,
                        variable_types={"home_team_api_id": ft.variable_types.Categorical,
                                              "away_team_api_id": ft.variable_types.Categorical,
                                              "home_status": ft.variable_types.Categorical,
                                              "home_team_goal":     ft.variable_types.Numeric,
                                              "away_team_goal":     ft.variable_types.Numeric})

es.entity_from_dataframe(entity_id="away_matches",
                        index="id",
                        time_index="date",
                        dataframe=matches_df,
                        variable_types={"home_team_api_id": ft.variable_types.Categorical,
                                              "away_team_api_id": ft.variable_types.Categorical,
                                              "home_status": ft.variable_types.Categorical,
                                              "home_team_goal":     ft.variable_types.Numeric,
                                              "away_team_goal":     ft.variable_types.Numeric})

es.entity_from_dataframe(entity_id="teams",
                         index="team_api_id",
                         dataframe=teams_df)

es.add_last_time_indexes()

new_relationship = ft.Relationship(es["teams"]["team_api_id"],
                                   es["home_matches"]["home_team_api_id"])
es = es.add_relationship(new_relationship)

new_relationship = ft.Relationship(es["teams"]["team_api_id"],
                                   es["away_matches"]["away_team_api_id"])
es = es.add_relationship(new_relationship)

feature_matrix, features_defs = ft.dfs(entities=es,
                                       entityset=es,
                                       cutoff_time=ct_matches,
                                       cutoff_time_in_index=True,
                                       training_window='60 days',
                                       max_depth=3,
                                       target_entity="teams",
                                       verbose=True
                                      )

print(feature_matrix)
# Recover the true datetime 
feature_matrix = feature_matrix.reset_index()
feature_matrix['time'] = feature_matrix['time'] + pd.Timedelta(hours=1)

print(feature_matrix['time'])

df_final = pd.merge(matches_df, feature_matrix, left_on=['date', 'home_team_api_id'], right_on=['time','team_api_id'], suffixes=('', '_HOME'))
df_final = pd.merge(df_final, feature_matrix, left_on=['date', 'away_team_api_id'], right_on=['time','team_api_id'], suffixes=('', '_AWAY'))
columns_to_drop = ["id", "team_fifa_api_id", "date", "team_long_name","team_long_name_AWAY", "team_short_name","team_short_name_AWAY", "home_status", "home_team_goal", "away_team_goal", "home_team_api_id", "away_team_api_id", "label_AWAY", "label", "goal_difference", 'team_api_id', 'time', 'team_api_id_AWAY', 'time_AWAY']

for c in df_final.columns:
    if 'MODE' in c:
        columns_to_drop.append(c)

y = df_final["home_status"]
df = df_final.drop(columns_to_drop, axis=1)
df = df.fillna(0)

print(df)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

# 1. Split X and y into a train and test set
X_train, X_test, y_train, y_test = train_test_split(df, y, shuffle=True, random_state=42)

# 2. Select features using RFE
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
estimator = clf
selector = RFE(estimator, 10, step=1)
selector = selector.fit(X_train, y_train)
X_train.iloc[:, selector.support_].tail()
clf.fit(selector.transform(X_train), y_train)

score = clf.score(selector.transform(X_test), y_test)
y_pred = clf.predict(selector.transform(X_test))

print(score)
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = y.unique()

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
bet = 10
earnings = 0

earnings = earnings + X_test[(y_pred == y_test) & (y_pred == 'W')]['B365H'].sum() * bet
earnings = earnings + X_test[(y_pred == y_test) & (y_pred == 'L')]['B365A'].sum() * bet
earnings = earnings + X_test[(y_pred == y_test) & (y_pred == 'D')]['B365D'].sum() * bet

earnings = earnings - len(X_test) * bet

print("You lose " + str(earnings) + "€ !")