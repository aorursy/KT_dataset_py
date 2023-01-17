#   Processing

import pandas as pd

import numpy as np

np.set_printoptions(threshold=np.nan)

import re

#   Visuals

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize']=(20,10)
df = pd.read_csv("../input/cereal.csv",sep=',')
df.head()
df.count()
df.isnull().sum()
ax = sns.barplot(x="rating", y="name", data=df.sort_values('rating',ascending=False)[:15])

_ = ax.set(xlabel='Rating', ylabel='Name',title = "The best rated cereals")
ax = sns.lmplot(size=12, x="rating", y="calories", data=df)

_ = ax.set(xlabel='Rating', ylabel='Calories',title = "Relation between calories and rating")
df.dtypes
# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in df.columns if col not in ['rating'] if df[col].dtype=='float64' or 

         df[col].dtype=='int64']

labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(df[col].values, df.rating.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,12))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

plt.show()
ax = sns.pairplot(df[list(corr_df[-3:]['col_labels'])], kind="reg", size=4);
df['protein/calories']=df['protein']/df['calories']
dfBestRating = df[df['rating']>50]
ax = sns.lmplot(x='protein/calories', # Horizontal axis

           y='rating', # Vertical axis

           data=dfBestRating, # Data source

           size = 15,

           line_kws={'color': 'orange'})



plt.title('Protein/calories vs Rating')

# Set x-axis label

plt.xlabel('Protein/calories ratio')

# Set y-axis label

plt.ylabel('Rating')





def label_point(x, y, val, ax):

    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)

    for i, point in a.iterrows():

        ax.text(point['x']+.001, point['y'], str(point['val']))



label_point(dfBestRating['protein/calories'], dfBestRating['rating'], dfBestRating['name'], plt.gca()) 