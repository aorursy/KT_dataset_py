import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot;
import cufflinks as cf; init_notebook_mode(connected = True); cf.go_offline()
import plotly.graph_objs as go
%matplotlib inline
nba_2017_attendance_df = pd.read_csv('../input/nba_2017_attendance.csv')
nba_2017_elo_df = pd.read_csv('../input/nba_2017_elo.csv')
nba_2017_team_valuations_df = pd.read_csv('../input/nba_2017_team_valuations.csv')
print(nba_2017_attendance_df.columns, nba_2017_elo_df.columns, nba_2017_team_valuations_df.columns)
nba_2017_team_df_one = pd.merge(nba_2017_attendance_df, nba_2017_elo_df, how='inner', on=['TEAM'])
nba_2017_team_df = nba_2017_team_df_one.merge(nba_2017_team_valuations_df, how='inner', on=['TEAM'])
nba_2017_team_df.head()
nba_2017_br_df = pd.read_csv('../input/nba_2017_br.csv')
print(nba_2017_br_df.columns)
nba_2017_pie_df = pd.read_csv('../input/nba_2017_pie.csv')
print(nba_2017_pie_df.columns)
nba_2017_salary_df = pd.read_csv('../input/nba_2017_salary.csv')
print(nba_2017_salary_df.columns)
nba_2017_real_plus_minus_df = pd.read_csv('../input/nba_2017_real_plus_minus.csv')
print(nba_2017_real_plus_minus_df.columns)
nba_2017_player_wikipedia_df = pd.read_csv('../input/nba_2017_player_wikipedia.csv')
nba_2017_twitter_players_df = pd.read_csv('../input/nba_2017_twitter_players.csv')
print(nba_2017_player_wikipedia_df.columns, nba_2017_twitter_players_df.columns)
nba_2017_team_df.head()
fig = plt.figure(figsize=(12,5))
axes0 = plt.subplot(1,3,1)
axes0 = sns.boxplot(x='CONF', y='TOTAL', data=nba_2017_team_df, palette='coolwarm')
axes0.set_title('Total Attendance')

axes1 = plt.subplot(1,3,2)
axes1 = sns.stripplot(x='CONF', y='AVG', data=nba_2017_team_df, palette='seismic')
axes1.set_title('Average Attendance')

axes2 = plt.subplot(1,3,3)
axes2 = sns.swarmplot(x='CONF', y='PCT', data=nba_2017_team_df, palette='Set1')
axes2.set_title('Percent Attendance')

plt.tight_layout()
fig = plt.figure(figsize=(8,6))
nba_2017_team_df_corr = nba_2017_team_df.corr()
sns.heatmap(nba_2017_team_df_corr, cmap='cool', linewidth=1, linecolor='white')
# It seems that values of teams have bigger influence on PCT than ELO
trace = go.Scatter3d(
    x=nba_2017_team_df['ELO'],
    y=nba_2017_team_df['VALUE_MILLIONS'],
    z=nba_2017_team_df['PCT'],
    mode='markers',
    text=nba_2017_team_df['TEAM'],
    marker=dict(
        size=12, 
        color=nba_2017_team_df['PCT'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='The Relationship Between PCT and ELO/Value',
    scene = dict(
        xaxis = dict(title='X: ELO'),
        yaxis = dict(title='Y: VALUE_MILLIONS'),
        zaxis = dict(title='Z: PCT'),
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig = plt.figure(figsize=(12,6))
axes = sns.lmplot(x='ELO', y='PCT', data=nba_2017_team_df , hue = 'CONF', palette = 'coolwarm', markers = ['o', 'v'], scatter_kws = {'s' : 50})
fig = plt.figure(figsize=(12,6))
sns.lmplot(x='VALUE_MILLIONS', y='PCT', data=nba_2017_team_df, hue='CONF', palette='coolwarm',
                  markers = ['o', 'v'], scatter_kws = {'s' : 50})
pct_value_result = smf.ols('PCT~VALUE_MILLIONS', data=nba_2017_team_df).fit()
pct_elo_result = smf.ols('PCT~ELO', data=nba_2017_team_df).fit()
print(pct_value_result.summary())
print(pct_elo_result.summary())
nba_2017_br_df.head()
nba_2017_pie_df.head()
nba_2017_real_plus_minus_df.head()
nba_2017_salary_df.head()
nba_2017_br_df.rename(columns={'Player':'PLAYER'}, inplace=True)

players = []
for item in nba_2017_real_plus_minus_df['NAME']:
    player, position = item.split(',')
    players.append(player)
nba_2017_real_plus_minus_df['PLAYER'] = players
nba_2017_real_plus_minus_df.drop(['NAME'], axis=1, inplace=True)

nba_2017_salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
nba_2017_player_one = pd.merge(nba_2017_br_df, nba_2017_pie_df, on='PLAYER', how='inner')
nba_2017_player_two = nba_2017_player_one.merge(nba_2017_real_plus_minus_df, on='PLAYER', how='inner')
nba_2017_player = nba_2017_player_two.merge(nba_2017_salary_df, on='PLAYER', how='inner')
print(nba_2017_player.columns)
nba_2017_player.head()
nba_2017_player_ability = nba_2017_player[['PLAYER', 'AGE', 'FG', 'FGA', 'FG%', '3P', '3PA',
                                          '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FTA',
                                          'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
                                          'PF', 'MPG', 'ORPM', 'DRPM', 'RPM', 'PIE', 'W', 'L']]
nba_2017_player_ability['WIN_PERCENTAGE'] = nba_2017_player_ability['W'] / (nba_2017_player_ability['W'] + nba_2017_player_ability['L'])
nba_2017_player_ability.drop(['W', 'L'], axis=1, inplace=True)
nba_2017_player_ability = nba_2017_player_ability.rename(columns={'3P%': 'THREE_PERCENT', '2P%': 'TWO_PERCENT'})
nba_2017_player_ability_cor = nba_2017_player_ability.corr()
fig = plt.figure(figsize=(12,10))
sns.heatmap(nba_2017_player_ability_cor, cmap='coolwarm', linewidth=1, linecolor='white')
pie_3p_result = smf.ols('PIE~THREE_PERCENT', data=nba_2017_player_ability).fit()
print(pie_3p_result.summary())
pie_2p_result = smf.ols('PIE~TWO_PERCENT', data=nba_2017_player_ability).fit()
print(pie_2p_result.summary())
win_3p_result = smf.ols('WIN_PERCENTAGE~THREE_PERCENT', data=nba_2017_player_ability).fit()
print(win_3p_result.summary())
win_2p_result = smf.ols('WIN_PERCENTAGE~TWO_PERCENT', data=nba_2017_player_ability).fit()
print(win_2p_result.summary())
pie_stl_result = smf.ols('PIE~STL', data=nba_2017_player_ability).fit()
print(pie_stl_result.summary())
pie_blk_result = smf.ols('PIE~BLK', data=nba_2017_player_ability).fit()
print(pie_blk_result.summary())
win_stl_result = smf.ols('WIN_PERCENTAGE~STL', data=nba_2017_player_ability).fit()
print(win_stl_result.summary())
win_blk_result = smf.ols('WIN_PERCENTAGE~BLK', data=nba_2017_player_ability).fit()
print(win_blk_result.summary())
trace = go.Scatter(
    x=nba_2017_player_ability['ORPM'],
    y=nba_2017_player_ability['MPG'],
    mode='markers',
    text=nba_2017_player_ability['PLAYER'],
    marker=dict(
        size=12,               
        color=nba_2017_player_ability['ORPM'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='NBA Players ORPM',
    scene = dict(
        xaxis = dict(title='X: ORPM'),
        yaxis = dict(title='Y: MPG'),
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace = go.Scatter(
    x=nba_2017_player_ability['DRPM'],
    y=nba_2017_player_ability['MPG'],
    mode='markers',
    text=nba_2017_player_ability['PLAYER'],
    marker=dict(
        size=12,               
        color=nba_2017_player_ability['DRPM'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='NBA Players DRPM',
    scene = dict(
        xaxis = dict(title='X: DRPM'),
        yaxis = dict(title='Y: MPG'),
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace = go.Scatter3d(
    x=nba_2017_player_ability['ORPM'],
    y=nba_2017_player_ability['DRPM'],
    z=nba_2017_player_ability['PIE'],
    mode='markers',
    text=nba_2017_player_ability['PLAYER'],
    marker=dict(
        size=12,               
        color=nba_2017_player_ability['PIE'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='The Relationship Between PIE and ORPM/DRPM',
    scene = dict(
        xaxis = dict(title='X: ORPM'),
        yaxis = dict(title='Y: DRPM'),
        zaxis = dict(title='Z: PIE'),
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace = go.Scatter3d(
    x=nba_2017_player_ability['ORPM'],
    y=nba_2017_player_ability['DRPM'],
    z=nba_2017_player_ability['WIN_PERCENTAGE'],
    mode='markers',
    text=nba_2017_player_ability['PLAYER'],
    marker=dict(
        size=12,               
        color=nba_2017_player_ability['WIN_PERCENTAGE'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='The Relationship Between WIN_PERCENTAGE and ORPM/DRPM',
    scene = dict(
        xaxis = dict(title='X: ORPM'),
        yaxis = dict(title='Y: DRPM'),
        zaxis = dict(title='Z: PIE'),
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
nba_2017_player_wikipedia_df.head()
nba_2017_player_wikipedia_df.drop(['Unnamed: 0', 'timestamps', 'wikipedia_handles'], axis=1, inplace=True)
nba_2017_player_wikipedia_df = nba_2017_player_wikipedia_df.rename(columns={'names':'PLAYER', 'pageviews':'PAGEVIEWS'})
nba_2017_player_wikipedia_avg_df = nba_2017_player_wikipedia_df.groupby(['PLAYER'], as_index=False).mean()
nba_2017_player_wikipedia_avg_df.head()
nba_2017_twitter_players_df.head()
nba_2017_player_social = pd.merge(nba_2017_twitter_players_df, nba_2017_player_wikipedia_avg_df,
                                 on='PLAYER', how='inner')
nba_2017_player_social_power = nba_2017_player_social.merge(nba_2017_player_ability[['PLAYER','PIE','WIN_PERCENTAGE']],
                                                           on='PLAYER', how='inner')
nba_2017_player_social_power.head()
nba_2017_player_social_power_cor = nba_2017_player_social_power.corr()
fig = plt.figure(figsize=(12,8))
sns.heatmap(nba_2017_player_social_power_cor, cmap='coolwarm', linewidth=1, linecolor='white')
sns.jointplot(x='PIE', y='TWITTER_FAVORITE_COUNT', data=nba_2017_player_social_power, kind='reg')
sns.jointplot(x='PIE', y='TWITTER_RETWEET_COUNT', data=nba_2017_player_social_power, kind='reg')
win_twitter_result = smf.ols('WIN_PERCENTAGE~TWITTER_FAVORITE_COUNT', data=nba_2017_player_social_power).fit()
print(win_twitter_result.summary())
win_wikipedia_result = smf.ols('WIN_PERCENTAGE~PAGEVIEWS', data=nba_2017_player_social_power).fit()
print(win_wikipedia_result.summary())


