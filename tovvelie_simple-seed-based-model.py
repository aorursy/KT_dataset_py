import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import glob

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from scipy.optimize import minimize

import itertools
data_dict = {}

for i in glob.glob('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/*'):

    name = i.split('/')[-1].split('.')[0]

    if name != 'MTeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='cp1252')
def add_seed_num(df_input, df_seeds=data_dict['MNCAATourneySeeds']):

    

    df = df_input.copy()



    df = df.join(df_seeds.set_index(['Season', 'TeamID'])['Seed'].rename('Wseed'), on=['Season', 'WTeamID'])

    df = df.join(df_seeds.set_index(['Season', 'TeamID'])['Seed'].rename('Lseed'), on=['Season', 'LTeamID'])

    df['WRegionSeed'] = df['Wseed'].apply(lambda x: int(x[1:3]))

    df['LRegionSeed'] = df['Lseed'].apply(lambda x: int(x[1:3]))



    return df





def add_game_round(Seed_DayNum, df_rounds=data_dict['MNCAATourneySeedRoundSlots']):

    

    Seed = Seed_DayNum[:-3]

    DayNum = int(Seed_DayNum[-3:])

    GameRound = list(df_rounds[(df_rounds['Seed'] == Seed) & (df_rounds['EarlyDayNum'] <= DayNum) & (df_rounds['LateDayNum'] >= DayNum)]['GameRound'].unique()

                    )

    if len(GameRound) > 1:

        print('error')

        

    return str(GameRound[0])





def winrate_by_seed(df_input, start_year=0, end_year=0, window=5):

    

    df = df_input.copy()

    df = add_seed_num(df)

    df['SameSeed'] = (df['WRegionSeed'] == df['LRegionSeed']).astype(int)

    df = df[df['SameSeed'] == 0].copy() 

    df_win = df[['Season', 'WRegionSeed']].copy().rename(columns={'WRegionSeed': 'Seed'})

    df_win['Winrate'] = 1

    df_lose = df[['Season', 'LRegionSeed']].copy().rename(columns={'LRegionSeed': 'Seed'})

    df_lose['Winrate'] = 0

    df_concat = pd.concat((df_win, df_lose))

    years_to_predict = np.arange(start_year, end_year+1)

    result = pd.DataFrame()

    

    for year in years_to_predict:

        

        df_window = df_concat[(df_concat['Season'] < year) & (df_concat['Season'] >= year-window)].copy()

        df_window['Season'] = year

        result = pd.concat((result, df_window))

    

    return result.groupby(['Season', 'Seed'], as_index=False)['Winrate'].mean()





def winrate_by_seed_linear(df_input, start_year=0, end_year=0, window=5):

    

    df = df_input.copy()

    df = add_seed_num(df)

    df = order_teams(df)

    

    years_to_predict = np.arange(start_year, end_year+1)

    result = pd.DataFrame()

    

    for year in years_to_predict:

        

        df_window = df[(df['Season'] < year) & (df['Season'] >= year-window)].copy()

        seeds = df_window['Team1Seed'].values

        results = df_window['Result'].values

        SE = lambda x: np.sum((x[0] * seeds + x[1] - results) ** 2)

        coef = minimize(SE, (-0.05, 1), method='Nelder-Mead', tol=1e-6)['x']

        year_results = pd.DataFrame()

        year_results['Seed'] = np.arange(1,17)

        year_results['Winrate'] = year_results['Seed'].apply(lambda x: x * coef[0] + coef[1])

        year_results['Season'] = year   

        result = pd.concat((result, year_results))

    

    return result





def order_teams(df_input):

    

    df = df_input.copy()

    df['Team1'] = np.where(df.WTeamID < df.LTeamID, df.WTeamID, df.LTeamID)

    df['Team2'] = np.where(df.WTeamID > df.LTeamID, df.WTeamID, df.LTeamID)

    df['Team1Seed'] = np.where(df.WTeamID < df.LTeamID, df.WRegionSeed, df.LRegionSeed)

    df['Team2Seed'] = np.where(df.WTeamID > df.LTeamID, df.WRegionSeed, df.LRegionSeed)

    df['Result'] = np.where(df.WTeamID < df.LTeamID, 1, 0)

    

    return df





def prepare_train_dataset(start_year=2010, end_year=2014, approximation=None):

    train = data_dict['MNCAATourneyCompactResults'][(data_dict['MNCAATourneyCompactResults']['Season'] <= end_year) \

                                                    & (data_dict['MNCAATourneyCompactResults']['Season'] >= start_year)].copy()

    

    train = add_seed_num(train)

    train = order_teams(train)

    

    if approximation == 'linear':

        winrate_by_seed_train = winrate_by_seed_linear(data_dict['MNCAATourneyCompactResults'], start_year, end_year)

        

    else: 

        winrate_by_seed_train = winrate_by_seed(data_dict['MNCAATourneyCompactResults'], start_year, end_year)

        

    train = train.join(winrate_by_seed_train.set_index(['Season', 'Seed'])['Winrate'].rename('Team1Winrate'), on=['Season', 'Team1Seed'])

    train = train.join(winrate_by_seed_train.set_index(['Season', 'Seed'])['Winrate'].rename('Team2Winrate'), on=['Season', 'Team2Seed'])    

    train['WinrateDiff'] = train['Team1Winrate'] - train['Team2Winrate']

    train_X = train[['Team1Winrate', 'Team2Winrate']]

    train_y = train['Result']

    

    return train_X, train_y





def prepare_test_dataset(df_input, start_year=2015, end_year=2019):

    

    df = df_input.copy()

    

    df['Season'] = df['ID'].apply(lambda x: int(x.split('_')[0]))

    df['Team1'] = df['ID'].apply(lambda x: int(x.split('_')[1]))

    df['Team2'] = df['ID'].apply(lambda x: int(x.split('_')[2]))

    

    df_seeds=data_dict['MNCAATourneySeeds']

    

    df = df.join(df_seeds.set_index(['Season', 'TeamID'])['Seed'].rename('Team1seed'), on=['Season', 'Team1'])

    df = df.join(df_seeds.set_index(['Season', 'TeamID'])['Seed'].rename('Team2seed'), on=['Season', 'Team2'])

    df['Team1RegionSeed'] = df['Team1seed'].apply(lambda x: int(x[1:3]))

    df['Team2RegionSeed'] = df['Team2seed'].apply(lambda x: int(x[1:3]))

    

    winrate_by_seed = winrate_by_seed_linear(data_dict['MNCAATourneyCompactResults'], start_year, end_year)



    

    df = df.join(winrate_by_seed.set_index(['Season', 'Seed'])['Winrate'].rename('Team1Winrate'), on=['Season', 'Team1RegionSeed'])

    df = df.join(winrate_by_seed.set_index(['Season', 'Seed'])['Winrate'].rename('Team2Winrate'), on=['Season', 'Team2RegionSeed'])   

    

    return df['ID'], df[['Team1Winrate', 'Team2Winrate']]
df_TourneyResultsSeeds = add_seed_num(data_dict['MNCAATourneyCompactResults'])

df_TourneyResultsSeeds['GameRound'] = (df_TourneyResultsSeeds.Wseed + df_TourneyResultsSeeds.DayNum.astype(str)).apply(lambda x : add_game_round(x)).astype(int)
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))

Seeds_groupby = df_TourneyResultsSeeds.groupby(['WRegionSeed', 'LRegionSeed', 'GameRound'], as_index=False)['DayNum'].count()



for GameRound in range(1, 7):

    axes = ax[(GameRound-1) // 3, (GameRound-1) % 3]

    sns.heatmap(Seeds_groupby[Seeds_groupby['GameRound'] == GameRound].pivot(index='WRegionSeed', columns='LRegionSeed', values='DayNum') \

                .fillna(0).astype(int), annot=True, cmap='Blues', fmt='2g', ax=axes)

    axes.set_title('Round ' + str(GameRound))
print('Baseline:', -np.log(0.5))
train_X, train_y = prepare_train_dataset(start_year=2010, end_year=2014)
clf1 = LogisticRegression()

cv1 = cross_val_score(clf1, train_X.values, train_y.values, cv=5)



cv1.mean(), cv1.std()
df_TourneyResultsSeeds = order_teams(df_TourneyResultsSeeds)
plt.plot(np.arange(1,17), df_TourneyResultsSeeds.groupby('Team1Seed')['Result'].mean().values)

plt.ylabel('Winrate')

plt.xlabel('Seed')

plt.title('Winrate by seed');
seeds = df_TourneyResultsSeeds['Team1Seed'].values

results = df_TourneyResultsSeeds['Result'].values

SE = lambda x: np.sum((x[0] * seeds + x[1] - results) ** 2)

coef = minimize(SE, (-0.05, 1), method='Nelder-Mead', tol=1e-6)['x']

approximation = np.arange(1,17) * coef[0] + coef[1]

plt.plot(np.arange(1,17), df_TourneyResultsSeeds.groupby('Team1Seed')['Result'].mean().values, label='True')

plt.plot(np.arange(1,17), approximation, label='Linear approximation')

plt.legend()

plt.ylabel('Winrate')

plt.xlabel('Seed')

plt.title('Winrate by seed');
train_X_linear, train_y = prepare_train_dataset(start_year=2010, end_year=2014, approximation='linear')
clf2 = LogisticRegression()

cv2 = cross_val_score(clf2, train_X_linear.values, train_y.values, cv=5)



cv2.mean(), cv2.std()
x_s = np.array(list(itertools.product(np.linspace(0, 0.9, 100), np.linspace(0, 0.9, 100))))



clf1.fit(train_X, train_y)

y_s = clf1.predict(train_X)

y_s_proba = clf1.predict_proba(x_s)[:,1]



clf2.fit(train_X_linear, train_y)

y_s_linear = clf2.predict(train_X_linear)

y_s_proba_linear = clf2.predict_proba(x_s)[:,1]
fig, ax = plt.subplots(ncols=2, figsize=(16,4))



ax[0].scatter(x_s[:, 0], x_s[:, 1], c=y_s_proba, cmap='bwr')

ax[0].scatter(train_X.values[:, 0], train_X.values[:, 1], alpha=0.3, s=100, c=(train_y == y_s), cmap='bone')

ax[0].set_title('True winrate')

ax[0].set_xlabel('First team winrate')

ax[0].set_ylabel('Second team winrate')



ax[1].scatter(x_s[:, 0], x_s[:, 1], c=y_s_proba_linear, cmap='bwr')

ax[1].scatter(train_X_linear.values[:, 0], train_X_linear.values[:, 1], alpha=0.3, s=100, c=(train_y == y_s_linear), cmap='bone')

ax[1].set_title('Winrate linear approximation')

ax[1].set_xlabel('First team winrate')

ax[1].set_ylabel('Second team winrate');
df_submission = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
indices, dt_test = prepare_test_dataset(df_submission)
clf = LogisticRegression()

clf.fit(train_X_linear, train_y)

predictions = clf.predict_proba(dt_test)[:, 1]
df_submission['Pred'] = predictions

df_submission.to_csv('submission.csv', index=False)