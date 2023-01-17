import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
        
        
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/ipl/matches.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.drop("umpire3",axis =1, inplace=True)
df.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)

sns.countplot(x = 'season' , data = df)
sns.countplot(x = 'winner' , data = df)
fav_cities = df['city'].value_counts().reset_index()
fav_cities.columns = ['city','count']
sns.barplot(x = 'count',y = 'city', data = fav_cities[:10])
fav_stadium = df['venue'].value_counts().reset_index()
fav_stadium.columns = ['venue','count']
sns.barplot(x = 'count',y = 'venue', data = fav_stadium[:10])
toss = df.toss_decision.value_counts()
labels = (np.array(toss.index))
sizes = (np.array((toss / toss.sum())*100))
colors = ['red', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()
          
plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='toss_decision', data=df)
plt.xticks(rotation='vertical')
plt.show()

num_of_wins = (df.win_by_wickets>0).sum()
num_of_loss = (df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['green', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage of team batting second")
plt.show()
df["field_win"] = "win"
df["field_win"].loc[df['win_by_wickets']==0] = "loss"
plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='field_win', data=df)
plt.xticks(rotation='vertical')
plt.show()
MOM = df.player_of_match.value_counts()[:10]
labels = np.array(MOM.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(MOM), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")

plt.show()
bestump_df = pd.melt(df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])

best_ump = bestump_df.value.value_counts()[:10]
labels = np.array(best_ump.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(best_ump), width=width, color='r')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Umpires")

plt.show()
df['toss_winner_is_winner'] = 'no'
df['toss_winner_is_winner'].loc[df.toss_winner == df.winner] = 'yes'
temp_series = df.toss_winner_is_winner.value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss winner is match winner")
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(x='toss_winner', hue='toss_winner_is_winner', data=df)
plt.xticks(rotation='vertical')
plt.show()





