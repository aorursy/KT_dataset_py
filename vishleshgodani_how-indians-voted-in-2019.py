# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

df.head()
# Drop the unrequired columns

df.drop(columns=['SYMBOL','GENERAL\nVOTES','POSTAL\nVOTES','OVER TOTAL ELECTORS \nIN CONSTITUENCY','TOTAL ELECTORS'],inplace=True)
# Renaming of COLUMNS

df.rename(columns={'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY':'VOTE PERCENTAGE','TOTAL\nVOTES':'TOTAL VOTES','CRIMINAL\nCASES':'CRIMINAL CASES'},inplace=True)
# Creating copy of DataFrame excluding NOTA records

df_exclude_NOTA = df.copy()

df_exclude_NOTA.dropna(inplace=True)
df_exclude_NOTA.head()
# Education Qualification of Candidates for LOK SABHA - 2019



ax = df_exclude_NOTA.EDUCATION.value_counts().plot.bar(

figsize=(12,4),

color = 'green',

fontsize =14    

    

)



ax.set_title('Education Qualification of Candidates for LOK SABHA - 2019',fontsize=18)

ax.set_y_label('Number of candidates',fontsize=16)

sns.despine()
df[(df['EDUCATION']=='Illiterate')&(df['WINNER']==1)]
ax=df_exclude_NOTA.PARTY.value_counts().head(20).plot.bar(

figsize=(18,5),

color = '#2A89A1',

fontsize=12)



ax.set_title('Number of Seats Contested by PARTIES (TOP 20)',fontsize=20)

ax.set_ylabel('Number of Seats',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



sns.despine(left=True,bottom=True)

def win_percent_convertor(party):

    total_contested_seats = df[df['PARTY']==party].shape[0]

    total_seats_won = df[(df['PARTY']==party)&(df['WINNER']==1)].shape[0]

    win_percent = (total_seats_won/total_contested_seats)*100

    return win_percent

party_win_percent = {}



for party in df['PARTY'].unique():

    party_win_percent[party] = win_percent_convertor(party)

    

party_win_percent_series = pd.Series(party_win_percent)  



party_win_percent_series
# Seat Conversion Rate PARTYWISE



ax=party_win_percent_series.sort_values(ascending=False).head(36).plot.bar(

figsize=(17,5),

color='blue'    

)



ax.set_title('Seat Conversion Rate',fontsize=20)

ax.set_xlabel('Political Parties',fontsize=14)

ax.set_ylabel('Win Percentage',fontsize=14)



sns.despine(bottom=True,left=True)
ax=df_exclude_NOTA['PARTY'][(df_exclude_NOTA['WINNER']==1)].value_counts().head(20).plot.bar(

figsize=(16,4),

color='green'

)



ax.set_title('Number of Seats WON by Parties (TOP 20)',fontsize=20)

ax.set_ylabel('Seats Won',fontsize=16)

ax.set_xlabel('Political Parties',fontsize=16)



top_20_parties = pd.Series(df_exclude_NOTA['PARTY'].value_counts().head(21))

top_20_parties = top_20_parties.index.drop(['IND'])
df_partiwise_seats_comparison = pd.DataFrame(columns=df_exclude_NOTA.columns)



for count,party in enumerate(df['PARTY']):

    if party in top_20_parties:

        df_partiwise_seats_comparison = df_partiwise_seats_comparison.append(df.loc[count],ignore_index=True)
plt.figure(figsize=(17,6))

ax = sns.countplot(x='PARTY',hue='WINNER',data=df_partiwise_seats_comparison,palette='Set1')

ax.set_title('Comparison of Seats Won and Lost by Parties (TOP 20 PARTIES)',fontsize=20)

ax.legend(['Seats Lost','Seats Won'],loc='upper right',frameon=False),

ax.set_xlabel('Political Parties',fontsize=16)

ax.set_ylabel('Number of Seats',fontsize=16)
