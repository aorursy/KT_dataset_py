import numpy as np
import csv 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sqlalchemy import create_engine
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture as GMM

import warnings
warnings.filterwarnings('ignore')
%%time
NBA = pd.read_csv('/kaggle/input/nba-players-stats-20142015/players_stats.csv')
NBAc = NBA[pd.notnull(NBA['BMI'])]
print(NBA.columns)
duplicateRowsDF = NBA[NBA.duplicated()]
duplicateRowsDF
LAL_PLAYER = NBA.loc[NBA['Team']=='LAL']

LAL_PLAYER = LAL_PLAYER[['Name', 'Team', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']]

LAL_PLAYER.index = range(LAL_PLAYER.shape[0])

t_min, t_pts, t_reb, t_ast, t_stl, t_blk, t_tov =  LAL_PLAYER[['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']].sum()

attr_list = []

for i in LAL_PLAYER.index:
    
    p_name, p_team, p_min, p_pts, p_reb, p_ast, p_stl, p_blk, p_tov = LAL_PLAYER.iloc[i]
    
    attr_list.append([p_name, 'MIN', p_min/t_min])
    attr_list.append([p_name, 'PTS', p_pts/t_pts])
    attr_list.append([p_name, 'REB', p_reb/t_reb])
    attr_list.append([p_name, 'AST', p_ast/t_ast])
    attr_list.append([p_name, 'STL', p_stl/t_stl])
    attr_list.append([p_name, 'BLK', p_blk/t_blk])
    attr_list.append([p_name, 'TOV', p_tov/t_tov])
    
LAL_DATA = pd.DataFrame(attr_list, columns=['Name', 'Contribute_name', 'Values'])
import altair as alt

bars = alt.Chart(LAL_DATA).mark_bar().encode(
    x=alt.X('sum(Values)', stack='zero'),
    y=alt.Y('Contribute_name'),
    color=alt.Color('Name')
)

text = alt.Chart(LAL_DATA).mark_text(dx=-15, dy=3, color='white').encode(
    x=alt.X('sum(Values):Q', stack='zero'),
    y=alt.Y('Contribute_name'),
    detail='Name',
    text=alt.Text('sum(Values):Q', format='.4f')
)

(bars + text).properties(width=800, height=300)
NBA.head(10)
NBA.Age.plot.hist(rwidth=0.90,color= "g");
NBA.plot.scatter("Games Played", "PTS", alpha=0.5, color= "m");
height_weight_info = NBA[['Name', 'Height', 'Weight', 'Pos']]

height_weight_info.head(10)
alt.Chart(height_weight_info).mark_circle(size=20).encode(
    x = 'Height',
    y = 'Weight',
    color = 'Pos',
    tooltip = ['Name', 'Height', 'Weight', 'Pos']
).properties(
    width=600, 
    height=600
).interactive()

three_points_rate = NBA[['Team', '3P%']]

team_three_points_rate = three_points_rate.groupby('Team').mean()

team_three_points_rate['Team']  = team_three_points_rate.index

team_three_points_rate.index = [i for i in range(30)]

team_three_points_rate.head(10)
team_data = alt.Chart(team_three_points_rate).mark_bar(
    color='lightblue'
).encode(
    x = 'Team',
    y = '3P%'
)
mean_rate = alt.Chart(team_three_points_rate).mark_rule(
    color='green'
).encode(
    y = 'mean(3P%)'
)
(team_data + mean_rate).properties(width=600)
three_points_rate = NBA[['Team', '3PA']]

team_three_points_rate = three_points_rate.groupby('Team').mean()

team_three_points_rate['Team']  = team_three_points_rate.index

team_three_points_rate.index = [i for i in range(30)]

team_three_points_rate.head(10)
team_data = alt.Chart(team_three_points_rate).mark_bar(
    color='lightgreen'
).encode(
    x = 'Team',
    y = '3PA'
)
mean_rate = alt.Chart(team_three_points_rate).mark_rule(
    color='red'
).encode(
    y = 'mean(3PA)'
)
(team_data + mean_rate).properties(width=600)
three_points_rate = NBA[['Team', '3PM']]

team_three_points_rate = three_points_rate.groupby('Team').mean()

team_three_points_rate['Team']  = team_three_points_rate.index

team_three_points_rate.index = [i for i in range(30)]

team_three_points_rate.head(10)
team_data = alt.Chart(team_three_points_rate).mark_bar(
    color='lightyellow'
).encode(
    x = 'Team',
    y = '3PM'
)
mean_rate = alt.Chart(team_three_points_rate).mark_rule(
    color='red'
).encode(
    y = 'mean(3PM)'
)
(team_data + mean_rate).properties(width=600)
PTS_points_rate = NBA[['Team', 'PTS']]

team_PTS_points_rate = PTS_points_rate.groupby('Team').mean()

team_PTS_points_rate['Team']  = team_PTS_points_rate.index

team_PTS_points_rate.index = [i for i in range(30)]

team_PTS_points_rate.head(5)
team_data = alt.Chart(team_PTS_points_rate).mark_bar(
    color='lightgreen'
).encode(
    x = 'Team',
    y = 'PTS'
)
mean_rate = alt.Chart(team_PTS_points_rate).mark_rule(
    color='red'
).encode(
    y = 'mean(PTS)'
)
(team_data + mean_rate).properties(width=600)
NBA.plot.scatter
NBA.plot.scatter("EFF", "PTS", alpha=0.5, color= "r", figsize=(13,5))
plt.xlabel('Efisiensi Pemain')
plt.ylabel('Point')
msk = np.random.rand(len(NBAl)) < 0.8
train = NBAl[msk]
test = NBAl[~msk]

plt.figure(figsize=(13,5))
plt.scatter(train.EFF, train.PTS, alpha=0.5, color='blue')
plt.xlabel("Efisiensi Pemain")
plt.ylabel("Point")
plt.show()


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['EFF']])
train_y = np.asanyarray(train[['PTS']])
regr.fit (train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.figure(figsize=(13,5))
plt.scatter(train.EFF, train.PTS, alpha=0.5, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Efisiensi Pemain")
plt.ylabel("Point")
Boston = NBA.Team == 'BOS'
Detroit = NBA.Team == 'DET'
Cleveland = NBA.Team == 'CLE'
Orlando = NBA.Team == 'ORL'
Toronto = NBA.Team == 'TOR'
Washington = NBA.Team == 'WAS'
Philadelphia = NBA.Team == 'PHI'
Milwaukee = NBA.Team == 'MIL'
NewYork = NBA.Team == 'NYK'
Atlanta = NBA.Team == 'ATL'
Charlote = NBA.Team == 'CHA'
Chicago = NBA.Team == 'CHI'
Indiana = NBA.Team == 'IND'
Miami = NBA.Team == 'MIA'

dfeast = NBA[Boston | Detroit | Cleveland | Orlando | Toronto | Washington | Philadelphia | 
            Milwaukee | NewYork | Atlanta | Charlote | Chicago | Indiana | Miami]
maxt = dfeast.groupby(['Team'])['PTS'].transform(max) == dfeast['PTS']
dfmax = dfeast[maxt]
dfmax.sort_values("EFF", ascending=False)
import plotly.offline as py
import plotly.graph_objs as go

tr1 = go.Bar(
                 x = dfmax['Name'],
                 y = dfmax['EFF'],
                 name = 'Effisiensi',
                 marker = dict(color='crimson',
                              line = dict(color='rgba(0,0,0)', width=1)),
                 text = dfmax.Team)

tr2 = go.Bar(
                 x = dfmax['Name'],
                 y = dfmax['PTS'],
                 name = 'Points',
                 marker = dict(color='rgba(0, 1, 255, 1)',
                              line = dict(color='rgba(0,0,0)', width=1)),
                 text = dfmax.Team)
dn = [tr1, tr2]
layoutnew = go.Layout(barmode='group', title='Effisiensi Tertinggi Pemain pada TIM Regional Timur Terhadap Points')
fig = go.Figure(data=dn, layout=layoutnew)
fig.update_layout(barmode='stack')
py.iplot(fig)
engine= create_engine('sqlite:///:memory:')

dfeast.to_sql('data_table', engine) 
dfeastrt= pd.read_sql_query('SELECT SUM("EFF"), Team FROM data_table group by Team', engine)
K = dfeastrt['SUM("EFF")']
engine= create_engine('sqlite:///:memory:')

dfeast.to_sql('data_table', engine) 
dfeastrp= pd.read_sql_query('SELECT SUM("PTS"), Team FROM data_table group by Team', engine)
dfeastrp.insert(1, 'SEFF', K)
dfeastrp.rename(
    columns={
        'SUM("PTS")': "SPTS"
    },
    inplace=True)
dfeastrp['Rank'] = dfeastrp['SPTS'].rank(method='dense', ascending=False)
dfeastrp.sort_values("SPTS", ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=dfeastrp['Team'], y=dfeastrp['SEFF'], fill='tozeroy',name = 'Total Team Efficiency'))
fig.add_trace(go.Scatter(x=dfeastrp['Team'], y=dfeastrp['SPTS'], fill='tonexty',name = 'Number of Team Points'))

fig.show()
engine= create_engine('sqlite:///:memory:')

dfeast.to_sql('data_table', engine) 
gp= pd.read_sql_query('SELECT "Games Played" FROM data_table group by Team', engine)

dfeastrp.insert(3, 'GP', gp)
dfeastgpp = dfeastrp['SPTS'] / 82
dfeastgpp
# dfeastrp['GP'].max()
fig = {
        'data': [ 
             {
                'values' : dfeastgpp,
                'labels' : dfeastrp['Rank'],
                'domain' : {'x': [0, 1]},
                'name' : 'Points / Game',
                'hoverinfo' : 'label+percent+name',
                'hole' : 0.3,
                'type' : 'pie'
              },
             ],
         'layout' : {
                     'title' : 'คะแนนเฉลี่ยต่อเกมที่ได้รับจากทีมตามการจัดอันดับ',
                     'annotations' : [
                                        { 'font' : {'size' : 20},
                                          'showarrow' : False,
                                          'text' : ' ',
                                          'x' : 0.20,
                                          'y' : 1
                                         },
                                      ]    
                     }
        }
py.iplot(fig)
dfatl = dfeast[dfeast['Team'] == "ATL"]
dfatl
NBAdel1 =NBAc.Name != 'Sim Bhullar'
NBAdel2 = NBAc.Name != 'Jerrelle Benimon'
NBAcl = NBAc[NBAdel1 & NBAdel2]
NBAcl["TRB/MIN"] = NBAcl["REB"]/NBAcl["MIN"] 
NBAcl["AST/MIN"] = NBAcl["AST"]/NBAcl["MIN"]

fig, ax = plt.subplots()

x_var="AST/MIN"
y_var="TRB/MIN"

colors = {'SG':'blue', 'PF':'red', 'PG':'green', 'C':'purple', 'SF':'orange'}

ax.scatter(NBAcl[x_var], NBAcl[y_var], c=NBAcl['Pos'].apply(lambda x: colors[x]), s = 10)

# set a title and labels
ax.set_title('NBA Dataset')
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
NBAn = NBAcl[["AST/MIN","TRB/MIN"]]

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(NBAn)
print(kmeans.cluster_centers_)
d0=NBAn[y_kmeans == 0]
d1=NBAn[y_kmeans == 1]
d2=NBAn[y_kmeans == 2]
d3=NBAn[y_kmeans == 3]
d4=NBAn[y_kmeans == 4]

plt.scatter(d0[x_var], d0[y_var], s = 10, c = 'blue', label = 'D0')
plt.scatter(d1[x_var], d1[y_var], s = 10, c = 'green', label = 'D1')
plt.scatter(d2[x_var], d2[y_var], s = 10, c = 'red', label = 'D2')
plt.scatter(d3[x_var], d3[y_var], s = 10, c = 'purple', label = 'D3')
plt.scatter(d4[x_var], d4[y_var], s = 10, c = 'orange', label = 'D4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
d0[x_var]='SG'
d1[x_var]='PF'
d2[x_var]='PG'
d3[x_var]='C'
d4[x_var]='SF'

NBAlist = pd.concat([d0[x_var], d1[x_var], d2[x_var], d3[x_var], d4[x_var]])
NBAcluster = (NBAc[["Name", "Team", "Pos"]])
NBAcluster

NBAcl["TRB/MIN"]
NBAcl["AST/MIN"]

NBAcluster.insert(2, 'TRBMIN', NBAcl["TRB/MIN"])
NBAcluster.insert(3, 'ASTMIN', NBAcl["AST/MIN"])
NBAcluster.insert(5, 'Next Pos', NBAlist)
NBAcluster