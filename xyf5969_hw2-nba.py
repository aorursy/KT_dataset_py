# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
players_salary = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_salary.csv')

players_twitter = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_twitter_players.csv')

players_minus = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_real_plus_minus.csv')

team_val_elo = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_att_val_elo_with_cluster.csv')

players_endorsement = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_endorsements.csv')

players_stats = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_players_stats_combined.csv')
players_twitter.head()
players_endorsement
players_endorsement['endorsement'] = players_endorsement['ENDORSEMENT'].str[1:]

players_endorsement['salary'] = players_endorsement['SALARY'].str[1:]

players_endorsement = players_endorsement[['NAME','salary','endorsement']]

players_endorsement
players_twitter = players_twitter.rename(columns={'PLAYER':'NAME'})

players_twitter
players_TWI_SAL = pd.merge(players_twitter,players_endorsement,how='inner',on='NAME')

players_TWI_SAL
def drop(I):

    replace = ' '

    S = I.strip().split(',')

    for i in range(len(S)):

        replace += S[i]

    return float(replace)

players_TWI_SAL['endorsement'] = players_TWI_SAL['endorsement'].map(drop)

players_TWI_SAL['salary'] = players_TWI_SAL['salary'].map(drop)
players_TWI_SAL
TWI_SAL_corr = players_TWI_SAL.corr()

sns.heatmap(TWI_SAL_corr)
result = smf.ols('endorsement ~TWITTER_FAVORITE_COUNT',data=players_TWI_SAL).fit()

result.summary()
players_stats
players_minus.head()
players_minus = players_minus[['RPM','WINS']]

players_minus
players_data = pd.merge(players_stats,players_minus,how='inner',on='RPM')

players_data
players_data = players_data.rename(columns={'PLAYER':'NAME'})

players_data = players_data[['NAME','AGE','MPG','ORPM','DRPM','RPM','WINS_RPM','PIE','WINS']]

players_data
players_data_SAL = pd.merge(players_data,players_TWI_SAL,how='inner',on='NAME')

players_data_SAL
data_corr = players_data_SAL.corr()

data_corr
sns.heatmap(data_corr)
result = smf.ols('endorsement ~AGE',data=players_data_SAL).fit()

result.summary()