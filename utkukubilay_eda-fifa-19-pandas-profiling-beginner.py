# Import Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling
df = pd.read_csv("../input/data.csv", delimiter = ",")

df.head()
df.columns
df.drop(['Unnamed: 0','Photo','Flag','Club Logo', "Loaned From"],axis=1,inplace=True)
plt.figure(figsize=(14,11))

sns.heatmap(df.isnull(), cbar=False);
df.isnull().sum().sort_values(ascending=False)
pandas_profiling.ProfileReport(df)
def convert_str_to_numeric(value):

    value = value.replace("€","")

    

    if value.endswith("M"):

        return float(value.split("M")[0]) * 1000000

    

    elif value.endswith("K"):

        return float(value.split("K")[0]) * 1000

    

df['Value'] = df['Value'].apply(convert_str_to_numeric)

df['Wage'] = df['Wage'].apply(convert_str_to_numeric)
df.describe().T
#take same colums



columns = ['Value','Age','Overall','Potential','Special','Release Clause',

        'International Reputation','Weak Foot','Skill Moves','Crossing',

       'Finishing', 'ShortPassing', 'Dribbling',

       'LongPassing', 'BallControl',  'Reactions', 'Balance', 'ShotPower',

       'Jumping', 'Penalties']

plt.figure(figsize=(18,13))

sns.heatmap(df[columns].corr(),annot = True);

sns.clustermap(df[columns].corr(),annot = False);
plt.figure(figsize = (10,5))

pl = sns.boxplot(x=df['Club'][1:50], y=df['Overall'], palette='hls');

pl.set_title(label='Overall - Club');

plt.xlabel('Club')

plt.ylabel('Overall')

plt.xticks(rotation = 80)

plt.ylim(85, 95)
clubs = ("Juventus", "Chelsea", "FC Barcelona" ,"Beşiktaş JK", "Real Madrid") # <3 Beşiktaş <3



df_col = df[['Name','Overall','Club']]

group_clubs = df_col.groupby('Club').groups



for i in clubs:

    indices = group_clubs[i]

    print(df_col.loc[[*indices]].sort_values(by='Overall',ascending=False).head(5))
sns.jointplot(df['Age'],df['Overall'],joint_kws={'s':5});
sns.jointplot(x=df['Overall'], y=df['Wage']);
df["Age_categ"]=pd.cut(df.Age,[15,20,25,30,35,40,45],labels=["15-20", "20-25", "25-30", "30-35", "35-40", "40-45"])   
sns.catplot("Age_categ", "Overall", data=df, kind="violin", hue= "Preferred Foot",split=True);
sns.catplot("Position", "Overall", data=df, kind="box").fig.set_size_inches(11,7)
df.groupby(by='Nationality')['Overall'].mean().sort_values(ascending=False).head(10)