# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.
#Let's see a sample
df.sample()
df=df[['ID','Name','Age','Nationality','Overall','Potential','Preferred Foot','International Reputation','Skill Moves','Jersey Number',]]
# Important Features
df.sample()
#let's see the Players Age Histogram PLot
plt.hist(df.Age)
#As you can see, Players above Age 40 are few and I would like to remove them.
df.Age.value_counts()
df = df[df.Age<=40]
df.Age.max()
plt.hist(df.Age)
df.sample()
#Now we can see that , Overall Rating , Potential Rating are a scale out of 100
df.describe()
df.groupby('Age')['Overall'].agg(np.max)
df[(df.Age == 16) & (df.Overall == 64)]
df['total_score'] = (df.Overall/20 + df.Potential/20 + df['International Reputation'] + df['Skill Moves'])*5
df.total_score
df.sample(2)
#Now, The Best Players are as follows
best = df.groupby('Age')['total_score'].agg(np.max).reset_index()
age_scores = list(zip(best.Age,best.total_score))
best_players = pd.DataFrame(columns=df.columns)
for index,i in enumerate(age_scores):
    best_players = best_players.append(df[(df.Age == i[0] ) & (df.total_score == i[1])],ignore_index=True )
best_players
sss = best_players.Nationality.value_counts().reset_index(name='counts')
sn.barplot(x="index", y="counts", data=sss)
plt.xticks(rotation=60)
#Top 5 players with best total_score
best_players.sort_values(by='total_score')[::-1][:5]
Jersey = best_players['Jersey Number'].value_counts().reset_index(name='counts')
sn.barplot(x="index", y="counts", data=Jersey)
plt.xticks(rotation=60)
df_after_edit = pd.read_csv("../input/data.csv")
df_after_edit.sample(3)

