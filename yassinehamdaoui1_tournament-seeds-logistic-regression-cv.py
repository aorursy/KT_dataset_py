import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data_dir = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/'

df_seeds = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')

df_tour = pd.read_csv(data_dir + 'MNCAATourneyCompactResults.csv')

df_seeds.head()

df_tour.head()

def seed_to_int(seed):

    

    """Get just the digits from the seeding. Return as int

    """

    s_int = int(seed[1:3])

    return(s_int)



df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)

df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # here the string label

df_seeds.head()
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})

df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})



df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])

df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])



df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed # difference seed



# Let's concat dataframes and show it !



df_concat.head()
# Winners 



df_wins = pd.DataFrame()

df_wins['SeedDiff'] = df_concat['SeedDiff']

df_wins['result'] = 1



# Losses

df_losses = pd.DataFrame()

df_losses['SeedDiff'] = -df_concat['SeedDiff']

df_losses['result'] = 0



# Concat them together



df_for_predictions = pd.concat((df_wins, df_losses))
df_for_predictions.head()
X_train = df_for_predictions.SeedDiff.values.reshape(-1,1)



y_train = df_for_predictions.result.values

X_train, y_train = shuffle(X_train, y_train)

logregModel = LogisticRegression()



params = {'C': np.logspace(start=-5, stop=3, num=9)}

clf = GridSearchCV(logregModel, params, scoring='neg_log_loss', refit=True)

clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

X = np.arange(-15, 15).reshape(-1, 1)

preds = clf.predict_proba(X)[:,1]
plt.plot(X, preds)

plt.xlabel('Team1 seed - Team2 seed')

plt.ylabel('P(Team1 will win)')
df_sample_sub = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')



n_test = len(df_sample_sub)



# get the year



def get_year_t1_t2(ID):

    """Return a tuple with ints `year`, `team1` and `team2`."""

    return (int(x) for x in ID.split('_'))
X_test = np.zeros(shape=(n_test, 1))



for i, row in df_sample_sub.iterrows():

    

    year, t1, t2 = get_year_t1_t2(row.ID)

    

    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]

    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]

    diff_seed = t1_seed - t2_seed

    X_test[i, 0] = diff_seed
preds = clf.predict_proba(X_test)[:,1] # predictor





clipped_preds = np.clip(preds, 0.05, 0.95) # clipped predictions



df_sample_sub.Pred = clipped_preds

df_sample_sub.head()
df_sample_sub.to_csv('Submission.csv', index=False)