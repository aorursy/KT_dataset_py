# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
players = pd.read_excel('/kaggle/input/ipl-data-set/Players.xlsx')

teams = pd.read_csv("/kaggle/input/ipl-data-set/teams.csv")

deliveries = pd.read_csv("/kaggle/input/ipl-data-set/deliveries.csv")

matches = pd.read_csv("/kaggle/input/ipl-data-set/matches.csv",parse_dates=['date'])

teamwise_home_and_away = pd.read_csv("/kaggle/input/ipl-data-set/teamwise_home_and_away.csv")

most_runs_average_strikerate = pd.read_csv("/kaggle/input/ipl-data-set/most_runs_average_strikerate.csv")
players.head()
teams.head()
deliveries.head()
matches.head()
teamwise_home_and_away.head()
most_runs_average_strikerate.head()
print("Number of teams:- ",teams['team1'].nunique())

teams['team1'].unique()
deliveries.info()
matches.info()
#Replacing the Full names by short names

matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',

                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',

                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']

                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
print("No. of Umpires 1: ",matches['umpire1'].nunique())

print("No. of Umpires 2: ",matches['umpire2'].nunique())

print("No. of Umpires 3: ",matches['umpire3'].nunique())

ump_set1= set(matches['umpire1'].unique())

ump_set2=set(matches['umpire2'].unique())

ump_set3=set(matches['umpire3'].unique())

all_set = ump_set1.intersection(ump_set2)

all_set=all_set.intersection(ump_set3)

print("Umpires who umpired as 1st, 2nd & 3rd umpires: ",all_set)

print("Total Umpires:- ",len(all_set))
#Number of matches umpired by each

import matplotlib.pyplot as plt

import seaborn as sns

plt.subplots(figsize=(14,6))

ax=matches['umpire1'].value_counts().plot.bar(width=0.9,color=sns.color_palette('pastel',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()),(p.get_x()+0.15,p.get_height()+1))

plt.xlabel("umpires ", fontsize=14)

plt.ylabel("Counts ", fontsize=14)

plt.title("Umpires-1 who have umpires most (from highest to lowest)", fontsize=20)

plt.show()

plt.subplots(figsize=(14,6))

ax=matches['umpire2'].value_counts().plot.bar(width=0.9,color=sns.color_palette('pastel',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.xlabel("Umpires", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.title("Umpires-2 who have umpired most (from highest to lowest)", fontsize=20)

plt.show()
plt.subplots(figsize=(14,6))

ax=matches['umpire3'].value_counts().plot.bar(width=0.9,color=sns.color_palette('pastel',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.xlabel("Umpires", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.title("Umpires-3 who have umpired most (from highest to lowest)", fontsize=20)

plt.show()
plt.subplots(figsize=(10,6))

ax=matches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.title("Teams that won the toss (from highest to lowest)", fontsize=20)

plt.xlabel("Teams", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
plt.subplots(figsize=(10,6))

sns.countplot(x='Season',hue='toss_decision',data=matches,palette=sns.color_palette('bright',20))

plt.title("decision to field or bat across seasons ")

plt.show()
MOM = matches['player_of_match'].tolist()

def CountFrequency(MOM): 

    freq = {} 

    names=[]

    win=[]

    for item in MOM: 

        if (item in freq): 

            freq[item] += 1

        else: 

            freq[item] = 1

    for key, value in freq.items(): 

        names.append(key)

        win.append(value)

    dic={"NAMES":names,"No. of MOM": win}

    return dic

data=CountFrequency(MOM)

PoM=pd.DataFrame.from_dict(data)

PoM
df_sorted=PoM.nlargest(10, ['No. of MOM']) 

df_sorted
plt.figure(figsize=(10,6),dpi=200)

plt.bar(df_sorted['NAMES'],df_sorted['No. of MOM'])

plt.xticks(df_sorted['NAMES'], rotation=70,fontsize=10,fontweight='bold')

plt.yticks(np.arange(0,23),fontsize=11)

plt.title('Player of the match', fontsize=14,fontweight='bold')

plt.ylabel('no. of wins', fontsize=12,fontweight='bold')

plt.legend()

plt.grid()

plt.show()
print("Total number of Cities played: ",matches['city'].nunique())

print("Total number of Venues played: ",matches['venue'].nunique())
plt.figure(figsize=[14,7],dpi=250)

plt.xticks(rotation=90)

sns.countplot('venue',data=matches)

ax=plt.gca()

ax.set_xlabel('Grounds')

ax.set_ylabel('Count')

plt.title("Venues played ")

plt.show()
per=matches.pivot_table(index=['winner','Season'],aggfunc='size')

per=pd.DataFrame(per)

per.rename(columns={0:'wins'},inplace=True)

per.reset_index(inplace=True)

CSK=per[per['winner'] == 'CSK']

RCB=per[per['winner']=='RCB']

MI=per[per['winner']=='MI']

KKR=per[per['winner']=='KKR']

KXIP=per[per['winner']=='KXIP']

RR=per[per['winner']=='RR']

DD=per[per['winner']=='DD']
fig = plt.figure(figsize=(14,12))

#  subplot #1

plt.subplot(331)

plt.title('Chennai Super King', fontsize=14)

plt.plot(CSK['Season'],CSK['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)

#  subplot #2

plt.subplot(332)

plt.title('Royal Challanger Bangalore', fontsize=14)

plt.plot(RCB['Season'],RCB['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)



#  subplot #3

plt.subplot(333)

plt.title('Mumbai Indians', fontsize=14)

plt.plot(MI['Season'],MI['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)



#  subplot #4

plt.subplot(334)

plt.title('Kolkata Knight Riders', fontsize=14)

plt.plot(KKR['Season'],KKR['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)



#  subplot #5

plt.subplot(335)

plt.title('Rajasthan Royals', fontsize=14)

plt.plot(RR['Season'],RR['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)

#subplot 6

plt.subplot(336)

plt.title('Kings XI Punjab', fontsize=14)

plt.plot(KXIP['Season'],KXIP['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)



#subplot 7

plt.subplot(337)

plt.title('Delhi Daredevils', fontsize=14)

plt.plot(DD['Season'],DD['wins'],'-o')

plt.xticks(ticks=np.arange(13),labels= DD['Season'], rotation=70)

plt.tight_layout()

plt.show()

#Total number of matches

print(len(matches[matches['team1']=='CSK']) + len(matches[matches['team2']=='CSK']))

print(len(matches[matches['team1']=='MI']) + len(matches[matches['team2']=='MI']))

print(len(matches[matches['team1']=='KKR']) + len(matches[matches['team2']=='KKR']))

print(len(matches[matches['team1']=='RR']) + len(matches[matches['team2']=='RR']))

print(len(matches[matches['team1']=='RCB']) + len(matches[matches['team2']=='RCB']))

print(len(matches[matches['team1']=='KXIP']) + len(matches[matches['team2']=='KXIP']))

print(len(matches[matches['team1']=='SRH']) + len(matches[matches['team2']=='SRH']))

print(len(matches[matches['team1']=='DD']) + len(matches[matches['team2']=='DD']))
Stat=deliveries.groupby('batting_team')['total_runs'].sum()
print('chennai average score :',  Stat['CSK']/162)

print('Mumbai average score :',  Stat['MI']/184)

print('Kolkata average score :',  Stat['KKR']/177)

print('Rajasthan average score :',  Stat['RR']/144)

print('Banglore average score :',  Stat['RCB']/175)

print('punjab average score :',  Stat['KXIP']/175)

print('Hyderabad average score :',  Stat['SRH']/106)

print('Delhi average score :',  Stat['DD']/157)