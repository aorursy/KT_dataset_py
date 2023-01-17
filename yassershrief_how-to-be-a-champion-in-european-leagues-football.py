import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/extended-football-stats-for-european-leagues-xg/understat.com.csv')
df = df.rename(index=int, columns={'Unnamed: 0': 'league', 'Unnamed: 1': 'year'}) 
df.head()
#Leagues Numbers 
df['league'].value_counts()
f = plt.figure(figsize=(25,12))
ax = f.add_subplot(3,2,1)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Bundesliga') & (df['position'] <= 4)], ax=ax)
ax = f.add_subplot(3,2,2)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'EPL') & (df['position'] <= 4)], ax=ax)
ax = f.add_subplot(3,2,3)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'La_liga') & (df['position'] <= 4)], ax=ax)
ax = f.add_subplot(3,2,4)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Serie_A') & (df['position'] <= 4)], ax=ax)
ax = f.add_subplot(3,2,5)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'Ligue_1') & (df['position'] <= 4)], ax=ax)
ax = f.add_subplot(3,2,6)
plt.xticks(rotation=45)
sns.barplot(x='team', y='pts', hue='year', data=df[(df['league'] == 'RFPL') & (df['position'] <= 4)], ax=ax)
outlier_teams = ['Wolfsburg', 'Schalke 04', 'Leicester', 'Villareal', 'Sevilla', 'Lazio',
                 'Fiorentina', 'Lille', 'Saint-Etienne', 'FC Rostov', 'Dinamo Moscow']

# Removing unnecessary for our analysis columns 
df_xg = df[['league', 'year', 'position', 'team', 'scored', 'xG', 'xG_diff', 'missed',
            'xGA', 'xGA_diff', 'pts', 'xpts', 'xpts_diff']]
# Checking if getting the first place requires fenomenal execution
first_place = df_xg[df_xg['position'] == 1]

# Get list of leagues
leagues = df['league'].drop_duplicates()
leagues = leagues.tolist()

# Get list of years
years = df['year'].drop_duplicates()
years = years.tolist()
bu=first_place[first_place['league']=='Bundesliga']
bu
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bu["year"],bu["pts"],label='Points')
ax.bar(bu["year"],bu["xpts"],label='expected Points',alpha=0.8)
ax.legend()
ax.set_xlabel('year')
ax.set_ylabel('points')
ax.set_title('Comparing Actual and Expected Points for Winner Team in Bundesliga')
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bu["year"],bu["missed"],label='Actual goals missed')
ax.bar(bu["year"],bu["xGA"],label='expected goals')
ax.legend()
ax.set_xlabel('year')
ax.set_ylabel('points')
ax.set_title('Comparing actual goals missed and expected goals against for Winner Team in Bundesliga')
plt.show()
# and from this table we see that Bayern dominates here totally, even when they do not play well
bu2=df_xg[(df_xg['position'] <= 2)&(df_xg['league']=='Bundesliga')].sort_values(by=['year','pts'], ascending=False)
bu2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=bu2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

la=first_place[first_place['league']=='La_liga']
la
def points(df,league):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["year"],df["pts"],label='Points')
    ax.bar(df["year"],df["xpts"],label='expected Points',alpha=0.8)
    ax.legend()
    ax.set_xlabel('year')
    ax.set_ylabel('points')
    ax.set_title('Comparing Actual and Expected Points for Winner Team in '+league)
    plt.show()
points(la,'La liga')
# comparing with runner-up
la2=df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'La_liga')].sort_values(by=['year','xpts'], ascending=False)
la2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=la2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ep=first_place[first_place['league'] == 'EPL']
ep
points(ep,'EPL')
# comparing with runner-ups
ep2=df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'EPL')].sort_values(by=['year','xpts'], ascending=False)
ep2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=ep2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
li=first_place[first_place['league'] == 'Ligue_1']
li
points(li,'Ligue_1')
# comparing with runner-ups
li2=df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'Ligue_1')].sort_values(by=['year','xpts'], ascending=False)
li2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=li2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
se=first_place[first_place['league'] == 'Serie_A']
se
points(se,'Serie_A')
# comparing to runner-ups
se2=df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'Serie_A')].sort_values(by=['year','xpts'], ascending=False)
se2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=se2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
rf=first_place[first_place['league'] == 'RFPL']
rf
points(rf,'RFPL')
# comparing to runner-ups
ep2=df_xg[(df_xg['position'] <= 2) & (df_xg['league'] == 'RFPL')].sort_values(by=['year','xpts'], ascending=False)
ep2
fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x='year',y='pts',hue='team',data=ep2)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Creating separate DataFrames per each league
EPL = df_xg[df_xg['league'] == 'EPL']
print(EPL)
EPL.describe()

def print_records_antirecords(df):
  print('Presenting some records and antirecords: \n')
  for col in df.describe().columns:
    if col not in ['index', 'year', 'position']:
      team_min = df['team'].loc[df[col] == df.describe().loc['min',col]].values[0]
      year_min = df['year'].loc[df[col] == df.describe().loc['min',col]].values[0]
      team_max = df['team'].loc[df[col] == df.describe().loc['max',col]].values[0]
      year_max = df['year'].loc[df[col] == df.describe().loc['max',col]].values[0]
      val_min = df.describe().loc['min',col]
      val_max = df.describe().loc['max',col]
      print('The lowest value of {0} had {1} in {2} and it is equal to {3:.2f}'.format(col.upper(), team_min, year_min, val_min))
      print('The highest value of {0} had {1} in {2} and it is equal to {3:.2f}'.format(col.upper(), team_max, year_max, val_max))
      print('='*100)
# replace EPL with any league you want
print_records_antirecords(EPL)
#sns.set_palette(['blue','red','green','yellow','purple'])
hue_colors = {2018:'b',2017:'g',2016:'r',2015:'c',2014:'m'}
g=sns.relplot(x='position',y='xG_diff',hue='year',data=EPL,kind='line',palette=hue_colors,
            height=6,aspect=3)
g.fig.suptitle('Comparing xG gap between positions',fontsize=20)

plt.show()
#sns.set_palette(['blue','red','green','yellow','purple'])
hue_colors = {2018:'b',2017:'g',2016:'r',2015:'c',2014:'m'}
g=sns.relplot(x='position',y='xGA_diff',hue='year',data=EPL,kind='line',palette=hue_colors,
            height=6,aspect=3)
g.fig.suptitle('Comparing xGA gap between positions',fontsize=20)

plt.show()
#sns.set_palette(['blue','red','green','yellow','purple'])
hue_colors = {2018:'b',2017:'g',2016:'r',2015:'c',2014:'m'}
g=sns.relplot(x='position',y='xpts_diff',hue='year',data=EPL,kind='line',palette=hue_colors,
            height=6,aspect=3)
g.fig.suptitle('Comparing xPTS gap between positions',fontsize=20)

plt.show()
# Check mean differences
def league_mean(df):
    m=df.groupby('year')[['xG_diff', 'xGA_diff', 'xpts_diff']].mean()
    return m 

league_mean(EPL)
# Check median differences
def league_median(df):
    me=df.groupby('year')[['xG_diff', 'xGA_diff', 'xpts_diff']].median()
    return me 

league_median(EPL)