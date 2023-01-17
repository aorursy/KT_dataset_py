# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
all_elo = pd.read_csv('/kaggle/input/nfl-elo-ratings-from-538/nfl_elo.csv')
all_elo.head(15)
len(all_elo)
#First, remove the NaN values (only in season 2019)

all_elo = all_elo[~all_elo.score1.isna()==True]

all_elo ['predicted_tie'] = all_elo.elo1_pre == all_elo.elo2_pre

all_elo ['actual_tie'] = all_elo.score1 == all_elo.score2
all_elo [all_elo.predicted_tie == True]
all_elo [all_elo.actual_tie == True]
import plotly.express as px

ties_by_season = pd.DataFrame(all_elo.groupby(['season'])['actual_tie'].sum())

ties_by_season = ties_by_season.reset_index(drop=False)

print(ties_by_season.columns)

#fig, ax = plt.subplots()

fig = px.bar(ties_by_season, x='season', y='actual_tie',width = 600,height = 400, orientation = 'v')

fig.show()

non_ties = all_elo[~all_elo.actual_tie == True]

non_ties = non_ties[~non_ties.predicted_tie == True]
non_ties
non_ties ['team1_predicted'] = non_ties.elo1_pre > non_ties.elo2_pre

non_ties ['team1_won'] = non_ties.score1 > non_ties.score2




non_ties['correct_prediction']=non_ties.team1_predicted == non_ties.team1_won
non_ties[['team1_predicted','team1_won','correct_prediction']]
non_ties.tail()
non_ties.correct_prediction.describe().freq/len(non_ties)
non_ties_2019 = non_ties[non_ties.season==2019]

non_ties_2019.correct_prediction.describe().freq/len(non_ties_2019)

non_ties_2019.correct_prediction.describe()
non_ties['point_spread'] = (non_ties.elo2_pre - non_ties.elo1_pre)/25.0
non_ties.point_spread.describe()
non_ties[non_ties.point_spread <=-20]
non_ties['score1_adj'] = non_ties.score1 + non_ties.point_spread
non_ties['team1_won_adj'] = non_ties.score1_adj > non_ties.score2
non_ties[non_ties.team1_won != non_ties.team1_won_adj]
non_ties['correct_spread_prediction']=non_ties.team1_predicted == non_ties.team1_won_adj
non_ties.correct_spread_prediction.describe().freq/len(non_ties)