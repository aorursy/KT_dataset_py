%matplotlib inline

import numpy as np

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 5000)

from tqdm import tqdm

import plotly.express as px

import plotly.subplots as subplots

import plotly.graph_objects as go

import seaborn as sns

import os

import matplotlib.pyplot as plt

from PIL import Image



from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)
comps = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

cols = ['Id', 'Title', 'EnabledDate', 'DeadlineDate', 'FinalLeaderboardHasBeenVerified',

        'EvaluationAlgorithmName', 'RewardType', 'UserRankMultiplier', 'CanQualifyTiers', 'TotalTeams',

       'TotalCompetitors', 'TotalSubmissions']

comps = comps[cols]

comps = comps[comps.Title.str.match('covid.+global forecasting', case=False)]

teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv',  low_memory=False)

tm = pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv',  low_memory=False)

users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv',  low_memory=False)
combined = teams.loc[teams.CompetitionId.isin(comps.Id), ['Id', 'CompetitionId', 'TeamName', 'PrivateLeaderboardRank']].copy()

combined = combined.merge(tm[[ 'TeamId', 'UserId']], left_on='Id', right_on='TeamId' )

combined = combined.merge(users[['Id', 'UserName', 'DisplayName']], left_on='UserId', right_on='Id',suffixes=['Team', 'User'])



team_size = combined.groupby('TeamId')[['UserId']].nunique().reset_index()

team_size.columns = ['TeamId', 'TeamSize']

combined = combined.merge(team_size, on='TeamId')

combined = combined.merge(comps[['Id', 'Title', 'UserRankMultiplier', 'TotalTeams']], left_on='CompetitionId', right_on='Id')

combined = combined.dropna()
combined['KagglePoints'] = combined.UserRankMultiplier * 10**5 / np.sqrt(combined.TeamSize) * combined.PrivateLeaderboardRank ** (-0.75) * np.log10(1 + np.log10(combined.TotalTeams))

combined.KagglePoints = combined.KagglePoints.astype(int)
top = combined.groupby('DisplayName')[['KagglePoints']].sum().sort_values(by='KagglePoints', ascending=False).head(665).reset_index()

top[top['KagglePoints'] > 0].style.background_gradient(cmap='Reds').set_precision(0)
ranks = combined.loc[

    combined.DisplayName.isin(top.DisplayName),

    ['DisplayName', 'Title', 'PrivateLeaderboardRank']

].pivot('DisplayName', 'Title', 'PrivateLeaderboardRank')

ranks.columns = [c.split('(')[-1][:-1] for c in ranks.columns]

ranks.loc[top.DisplayName].style.background_gradient(cmap='Reds').set_precision(0)
weekly_points = combined.loc[

    combined.DisplayName.isin(top.DisplayName),

    ['DisplayName', 'Title', 'KagglePoints']

].pivot('DisplayName', 'Title', 'KagglePoints').fillna(0).round(0)

weekly_points = weekly_points.loc[top.DisplayName].astype(int)

weekly_points.columns = [c.split('(')[-1][:-1] for c in weekly_points.columns]

cm = sns.light_palette("red", as_cmap=True)

weekly_points.style.background_gradient(cmap=cm)