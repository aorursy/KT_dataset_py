from IPython.core.display import display, HTML
def partnership():
    display(HTML("""<h3>Thanks for cooperation</h3>
    Many thanks to <a href="http://www.esportbetting.eu" rel="dofollow">http://www.esportbetting.eu</a> for the help which enabled
    me to complete this project.<br /> The site is a great resource for eSport odds, betting statistics, and reviews. It is mainly focused on <a href="https://esportbetting.eu/games/csgo" rel="dofollow">CS:GO betting</a>, 
    but covers other eSports such as League of Legends or Dota2 as well."""))
partnership()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import matplotlib.patches as mpatches
EVENT_PATH = '../input/cleaned-csgo-events/cs_go_events_cleaned.csv'
df = pd.read_csv(EVENT_PATH)
df.head(3)
df.groupby('year').count()['event_link'].reset_index()
years_df = df.groupby('year').count().reset_index()
years, years_count = years_df.year, years_df.event_link
plt.figure(figsize=(8, 8), dpi=80)
plt.bar(years, years_count)
plt.xlabel('Year')
plt.ylabel('Event count')
plt.title('Event count by year')
plt.show()
event_type_by_year = df.groupby(['event_type', 'year']).count().reset_index()
sns.set(style="whitegrid")
g = sns.catplot(x="year", y="event_link", hue="event_type", data=event_type_by_year, height=9, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("event count")
money_year = df.groupby('year').count().reset_index()
money_by_year = df.groupby(['year']).sum()['prize_money'].reset_index()
plt.figure(figsize=(8, 8), dpi=80)
plt.bar(money_by_year.year, money_by_year.prize_money)
plt.xlabel('Year')
plt.ylabel('Money prize - 10 mln $')
plt.title('Money prizes by year')
plt.show()
x = df.groupby(['event_type', 'year']).sum()['prize_money']/df.groupby(['event_type', 'year']).count()['event_link']
money_by_event_by_year = x.reset_index() # inplace=True - > TypeError: Cannot reset_index inplace on a Series to create a DataFrame
money_by_event_by_year.columns.values
money_by_event_by_year.columns = ['event_type', 'year', 'average_money']

sns.set(style="whitegrid")
g = sns.catplot(x="year", y="average_money", hue="event_type", data=money_by_event_by_year, height=9, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("average money prize by event")
RANKING_PATH = '../input/cleaned-csgo-rankings/cs_go_hltv_ranking_clean.csv'
MATCHES_PATH = '../input/cleaned-csgo-matches/cs_go_matches_cleaned.csv'

df = pd.read_csv(MATCHES_PATH, low_memory=False)

df.columns.values

print("Maps played: "+str(df.shape[0]))
print("Matches: "+str(len(df.match_id.unique())))
matches_type = df.best_of.value_counts().reset_index()
matches_type.columns = ['best_of_type', 'best_of_count']
matches_type = df.best_of.value_counts().reset_index()
matches_type.columns = ['best_of_type', 'best_of_count']
plt.figure(figsize=(8, 8), dpi=80)
plt.bar(matches_type.best_of_type, matches_type.best_of_count)
plt.xlabel('Best of ')
plt.ylabel('Matches count')
plt.title('Count of different matches type')
plt.show()
len(df[df.best_of == 3]) / len(df.best_of)
matches_count_2016 = len(df[df.year == 2016].match_id.unique())
matches_count_2017 = len(df[df.year == 2017].match_id.unique())
matches_count_2018 = len(df[df.year == 2018].match_id.unique())
print("Matches in 2016: " + str(matches_count_2016) )
print("Matches in 2017: " + str(matches_count_2017) )
print("Matches in 2018: " + str(matches_count_2018) )
team_count_16 = len(pd.concat([df[df.year == 2016].home_team_id, df[df.year == 2016].away_team_id]).unique())
team_count_17 =len(pd.concat([df[df.year == 2017].home_team_id, df[df.year == 2017].away_team_id]).unique())
team_count_18 =len(pd.concat([df[df.year == 2018].home_team_id, df[df.year == 2018].away_team_id]).unique())

print("Teams in 2016: " + str(team_count_16) )
print("Teams in 2017: " + str(team_count_17) )
print("Teams in 2018: " + str(team_count_18) )
round(team_count_17/team_count_16, 3)
round(matches_count_2017/matches_count_2016, 3)
maps_count = df.maps.value_counts().reset_index()
maps_count = df.maps.value_counts().reset_index()
maps_count.columns = ['map_name', 'map_count']
plt.figure(figsize=(9, 9), dpi=80)
plt.bar(maps_count.map_name, maps_count.map_count)
plt.xlabel('Map name ')
plt.ylabel('Map count')
plt.title('Number of games per map')
plt.show()
hfs = df.groupby(['fist_half_home_side','maps']).home_first_score.sum().reset_index()
ass = df.groupby(['fist_half_home_side','maps']).away_second_score.sum().reset_index()
hss = df.groupby(['fist_half_home_side','maps']).home_second_score.sum().reset_index()
afs = df.groupby(['fist_half_home_side','maps']).away_first_score.sum().reset_index()
ct_1 = hfs[hfs.fist_half_home_side == 0].home_first_score + ass[ass.fist_half_home_side == 0].away_second_score #ct
t_1 = hfs[hfs.fist_half_home_side == 1].home_first_score + ass[ass.fist_half_home_side == 1].away_second_score #t
t_2 = hss[hss.fist_half_home_side == 0].home_second_score + afs[afs.fist_half_home_side == 0].away_first_score #t
ct_2 = hss[hss.fist_half_home_side == 1].home_second_score + afs[afs.fist_half_home_side == 1].away_first_score #ct
ct_1 = ct_1.reset_index().drop('index', axis=1)
ct_2 = ct_2.reset_index().drop('index', axis=1)
t_1 = t_1.reset_index().drop('index', axis=1)
t_2 = t_2.reset_index().drop('index', axis=1)
ct_terro_percent = pd.DataFrame
ct_terro_percent = t_1 +t_2
ct_terro_percent.columns = ['terro_won_total']
ct_terro_percent['maps'] = afs[afs.fist_half_home_side == 0].maps
ct_terro_percent['ct_won_total'] = (ct_1 + ct_2)
ct_terro_percent['total_rounds'] = (ct_1 + ct_2) + (t_1 + t_2)

r = [0,1,2,3,4,5,6,7]
terro_percent = [i / j * 100 for i,j in zip(ct_terro_percent['terro_won_total'], ct_terro_percent['total_rounds'])]
ct_percent = [i / j * 100 for i,j in zip(ct_terro_percent['ct_won_total'], ct_terro_percent['total_rounds'])]
plt.figure(figsize=(9, 9), dpi=80)
barWidth = 0.9
names = ('Cache','Cobblestone','Dust2','Inferno','Mirage', 'Nuke','Overpass','Train',)
plt.bar(r, terro_percent, color='#FFFF66', edgecolor='white', width=barWidth)
plt.bar(r, ct_percent, bottom=terro_percent, color='#00BFFF', edgecolor='white', width=barWidth)
plt.xticks(r, names)
plt.xlabel("Map")
plt.ylabel("Percentage")
plt.title("Rounds won by CT or T side by map")
blue_patch = mpatches.Patch(color='#00BFFF', label='Counter terrorist')
yellow_patch = mpatches.Patch(color='#FFFF66', label='Terrorist')
plt.legend(handles=[blue_patch, yellow_patch])
plt.yticks([0,10,20,30,40,45,55,60,70,80,90,100])
plt.axhline(y=50, color="red")
plt.show()
overtime_count = df[(df.home_first_score + df.home_second_score) == (df.away_first_score + df.away_second_score)].shape[0]
adventage_won_home = df[(df.home_first_score > df.away_first_score) & ((df.home_first_score + df.home_second_score) > (df.away_first_score+ df.away_second_score))].shape[0]
adventage_won_away = df[(df.home_first_score < df.away_first_score) & ((df.home_first_score + df.home_second_score) < (df.away_first_score+ df.away_second_score))].shape[0]
total_matches = df.shape[0]
adv_won_percent = round((adventage_won_away + adventage_won_home) / total_matches *100, 2)
overtime_percent = round(overtime_count / total_matches * 100, 2)
adv_lost_percent = round(100-(adv_won_percent+overtime_percent), 2)
print("Winning the map after winning the first half occurred in " +str(adv_won_percent) + "% cases")
print("Overtime was the case in " +str(overtime_percent) + "% matches")
print("Loosing map after adventage after first half: " +str(adv_lost_percent) + "%")
# reading dataframes
df_events = pd.read_csv(EVENT_PATH)
df_matches = pd.read_csv(MATCHES_PATH, low_memory=False)
df_ranking = pd.read_csv(RANKING_PATH)
# getting event/tournament id from link
df_events.columns.values
df_events['event_id'] = df_events.event_link.str.split("/")
events_id = df_events.event_id.reset_index()
events_id.columns = ['h_index', 'event_id']
for h_index in events_id.h_index:
    df_events.loc[h_index, 'event_id'] = df_events.loc[h_index].event_id[4]
df_events.event_id.head(3)
# combining dataframes - matches with events by event id - 
event_id_list = df_events.event_id.astype(int).tolist()
df_event_matches = pd.DataFrame
df_matches.shape
# This loop is taking some time so after the first launch, I saved the data frame to CSV file

# for event_id in range(0, len(df_matches.tournament_id)-1):
#     if not df_matches.loc[event_id].tournament_id in event_id_list:
#         df_matches = df_matches.drop(event_id)
# df_matches.to_csv('matches_to_model.csv', index=False)
df_matches = pd.read_csv('../input/cs-go-matches-to-model-clean/matches_to_model.csv', low_memory=False)
df_matches.shape
df_matches = df_matches.reset_index().drop("index", axis=1)
df_matches.columns.values
df_work_data = df_matches[['best_of','fist_half_home_side','maps',
                           'year','month','home_team_id','away_team_id',
                           'ht_p1_id','ht_p2_id','ht_p3_id','ht_p4_id','ht_p5_id',
                           'at_p1_id','at_p2_id','at_p3_id','at_p4_id','at_p5_id',
                           'home_first_score','away_first_score','home_second_score','away_second_score',
                           'home_team_score','away_team_score']]
sel_rows = df_work_data[(((df_work_data.home_first_score + df_work_data.home_second_score) == 
              (df_work_data.away_first_score + df_work_data.away_second_score)) 
             & (df_work_data.best_of == 3))].index
df_work_data = df_work_data.drop(sel_rows, axis=0)
df_work_data.shape
# Target values for binary classification - home team win/lose
df_work_data['target'] = np.where(df_work_data.home_team_score > df_work_data.away_team_score, 1, 0)
df_work_data.columns.values
df_work_data.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # splits data set into Train/Test.
lb_enc = LabelEncoder()
df_work_data["maps"] = lb_enc.fit_transform(df_work_data["maps"])
df_work_data = df_work_data.drop(['home_second_score', 'away_second_score',
                    'home_team_score', 'away_team_score'], axis =1 )
df_work_data.columns.values
y = df_work_data['target']
X = df_work_data.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=123)
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import Sequential
from keras.layers import Dropout
import xgboost as xgb
import time
def XGBoost(X_train,X_test,y_train,y_test,num_rounds=500):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={
                    'tree_method':'gpu_exact',
                    'eval_metric':'error',
                    'objective':'gpu:binary:logistic'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=50,evals=[(dtest,'test')],)
xgbm = XGBoost(X_train,X_test,y_train,y_test) # For output, click on the code button
iter_ = 0 
best_error = 0
best_iter = 0
best_model = None

col_sample_rates = [1]
subsamples = [1]
etas = [0.1]
max_depths = [8]
reg_alphas = [0.004]
reg_lambdas = [0.001]
min_child_weights = [3]
ntrees = [3000]

total_models = len(col_sample_rates)*len(subsamples)*len(etas)*len(max_depths)*len(reg_alphas) *len(reg_lambdas)*len(ntrees)*len(min_child_weights)
best_error = 1
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test,label=y_test)
# determine mean y value in training
# y_mean = train[y].mean()

for col_sample_rate in col_sample_rates:
    for subsample in subsamples:
        for eta in etas:
            for max_depth in max_depths:
                for reg_alpha in reg_alphas:
                    for reg_lambda in reg_lambdas:
                        for ntree in ntrees:
                            for min_child_weight in min_child_weights:
                                tic = time.time()

                                print('---------- ---------')

                                print('Training model %d of %d ...' % (iter_ + 1, total_models))
                                print('col_sample_rate =', col_sample_rate)
                                print('subsample =', subsample)
                                print('eta =', eta)
                                print('max_depth =', max_depth)
                                print('reg_alpha =', reg_alpha)
                                print('reg_lambda =', reg_lambda)
                                print('ntree =', ntree)
                                print('min_child_weights = ', min_child_weight)

                                params = {
                                     'booster': 'gbtree',
                                     'colsample_bytree': col_sample_rate,
                                     'eta': eta,
                                     'max_depth': max_depth,
                                     'nthread': -1,
                                     'min_child_weight': min_child_weight,
                                     'reg_alpha': reg_alpha,
                                     'reg_lambda': reg_lambda,
                                     'seed': 12345,
                                     'silent': 1,
                                     'n_estimators':ntree,
                                     'tree_method':'gpu_exact',
                                     'eval_metric':'error',
                                     'objective':'gpu:binary:logistic',
                                     'subsample': subsample}

                                watchlist = [(dtrain, 'train'), (dtest, 'eval')]

                                model = xgb.train(
                                                params, 
                                                dtrain, 
                                                ntree,
                                                early_stopping_rounds=100,
                                                evals=watchlist, 
                                                verbose_eval=False)

                                print('Model %d trained in %.2f s.'  % (iter_, time.time()-tic))
                                print('Model %d best score = %.4f' % (iter_, model.best_score))

                                if model.best_score < best_error:
                                    best_error = model.best_score
                                    best_iter = iter_
                                    best_model = model 
                                    print('Best so far!!!')
                                    print('Best error =', best_error)


                                iter_ += 1

print('Best model found at iteration: %d, with error: %.4f.' % (best_iter + 1, best_error))   
model = Sequential()
model.add(Dense(19, activation="relu",input_shape=(19,)))
model.add(Dense(15,activation="tanh"))
model.add(Dense(2))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x=X_train,y=y_train,batch_size=100,epochs=5)