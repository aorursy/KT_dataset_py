import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

import datetime
sns.set(style="darkgrid")



df = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')

df['year'] = pd.to_datetime(df['date']).dt.year

df['armed_gp'] = df['armed'].apply(lambda x: True if x != 'unarmed' else False)

df.columns
def uni_pic(df, column, order = True, x_rotation = 0):

    

    fig, ax = plt.subplots(figsize=(20,5))

    

    if order:

        sns.countplot(x = column, data = df, order = df[column].value_counts().index)

        

    if not order:

        sns.countplot(x = column, data = df)

        

    ax.set_xlabel(ax.get_xlabel(), fontsize=20)

    ax.set_ylabel("No. of Death", fontsize=20)



    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, fontsize=12)

    ax.set_yticklabels(ax.get_yticks().astype('int'),fontsize=12)



    return plt.show()
uni_pic(df, 'manner_of_death', order = True, x_rotation = 0)    
uni_pic(df, 'armed', order = True, x_rotation = 90) 
uni_pic(df, 'age', order = False, x_rotation = 90) 
uni_pic(df, 'gender', order = True, x_rotation = 0) 
uni_pic(df, 'race', order = True, x_rotation = 0)    
uni_pic(df, 'state', order = True, x_rotation = 90) 
uni_pic(df, 'signs_of_mental_illness', order = True, x_rotation = 0) 
uni_pic(df, 'threat_level', order = True, x_rotation = 0) 
uni_pic(df, 'flee', order = True, x_rotation = 0) 
uni_pic(df, 'body_camera', order = True, x_rotation = 0) 
df_dt = df.groupby(['year'])['id'].count().reset_index()



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_dt, x='year', y = 'id')



ax.set_xlabel('Year', fontsize=20)

ax.set_ylabel('No. of Death', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.0f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



plt.show()
df_bdy0 = df.groupby(['year'])['body_camera'].apply(lambda x: x.sum()/x.count()).reset_index()



df_bdy = df.groupby(['year','race'])['body_camera'].apply(lambda x: x.sum()/x.count()).reset_index()

df_bdy = df_bdy[df_bdy.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_bdy0, x='year', y = 'body_camera', color='grey', linewidth=2.5, label = 'AllRaces').lines[0].set_linestyle("--")

sns.lineplot(data = df_bdy, x='year', y = 'body_camera', hue = 'race', linewidth=2)



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Percentage with Camera in Use', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.2f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



plt.legend(fontsize = 14)

plt.show()
obs = df[df.manner_of_death == 'shot and Tasered'].groupby(['year'])['manner_of_death'].count()

pop = df.groupby(['year'])['manner_of_death'].count()

df_sub0 = (obs/pop).reset_index()



obs = df[df.manner_of_death == 'shot and Tasered'].groupby(['year','race'])['manner_of_death'].count()

pop = df.groupby(['year','race'])['manner_of_death'].count()

df_sub = (obs/pop).reset_index()

df_sub = df_sub[df_sub.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_sub0, x='year', y = 'manner_of_death', color='grey', linewidth=2.5, label = 'AllRaces').lines[0].set_linestyle("--")

sns.lineplot(data = df_sub, x='year', y = 'manner_of_death', hue = 'race', linewidth = 2)



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Percentage of Shoot-and-Tasered to Death', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.2f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



ax.legend(fontsize = 14)

plt.show()
df_sub = df[df.manner_of_death == 'shot and Tasered'].groupby(['year','race'])['manner_of_death'].count().reset_index()

df_sub = df_sub[df_sub.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_sub, x='year', y = 'manner_of_death', hue = 'race', linewidth = 2)



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('No. of Death', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.0f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



ax.legend(fontsize = 14)

plt.show()
df_sub0 = df.groupby(['year'])['signs_of_mental_illness'].apply(lambda x: x.sum()/x.count()).reset_index()



df_sub = df.groupby(['year','race'])['signs_of_mental_illness'].apply(lambda x: x.sum()/x.count()).reset_index()

df_sub = df_sub[df_sub.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_sub0, x='year', y = 'signs_of_mental_illness', color='grey', linewidth=2.5, label = 'AllRaces').lines[0].set_linestyle("--")

sns.lineplot(data = df_sub, x='year', y = 'signs_of_mental_illness', hue = 'race', linewidth=2)



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Percentage of Fatal Shooting with Signs of Mental Illness', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.2f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



plt.legend(fontsize = 14)

plt.show()
sub0 = df[(df.signs_of_mental_illness == True) & df.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.boxplot(x='year', y='age', data=sub0, hue='race')



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Age', fontsize=20)



#xlabels = ['%i'%i for i in ax.get_xticks()]

#ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.0f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



ax.legend(fontsize = 14, title='Races')

plt.show()
df_sub0 = df.groupby(['year'])['age'].mean().reset_index()



df_sub = df.groupby(['year','race'])['age'].mean().reset_index()

df_sub = df_sub[df_sub.race.isin(['W', 'B', 'H'])]



fig, ax = plt.subplots(figsize=(20,5))

sns.lineplot(data = df_sub0, x='year', y = 'age', color='grey', linewidth=2.5, label = 'AllRaces').lines[0].set_linestyle("--")

sns.lineplot(data = df_sub, x='year', y = 'age', hue = 'race', linewidth=2)



ax.set_xlabel(ax.get_xlabel(), fontsize=20)

ax.set_ylabel('Average Age', fontsize=20)



xlabels = ['%i'%i for i in ax.get_xticks()]

ax.set_xticklabels(xlabels, rotation=0, fontsize=12)

ylabels= ['{:,.2f}'.format(x) for x in ax.get_yticks()]

ax.set_yticklabels(ylabels,fontsize=12)



plt.legend(fontsize = 14)

plt.show()
armed = pd.get_dummies(df.armed_gp, prefix='Armed:')

flee = pd.get_dummies(df.flee, prefix='Flee:')

threat = pd.get_dummies(df.threat_level, prefix='Threat:')

df_sub = pd.concat([armed, flee, threat], axis=1)



sns.set(style="white")

# Compute the correlation matrix

corr = df_sub.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure

figure, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



xlabels = [x for x in df_sub.columns]

ax.set_xticklabels(xlabels, rotation=90, fontsize=12)

ylabels= [x for x in df_sub.columns]

ax.set_yticklabels(ylabels,fontsize=12)



#ax.set_title('Correlations among Attack, Threat & Flee', fontsize=20)



plt.show()
race = pd.get_dummies(df.race, prefix='Races:')

armed = pd.get_dummies(df.armed_gp, prefix='Armed:')

flee = pd.get_dummies(df.flee, prefix='Flee:')

threat = pd.get_dummies(df.threat_level, prefix='Threat:')

df_sub = pd.concat([race, armed, flee, threat], axis=1)



sns.set(style="white")

# Compute the correlation matrix

corr = df_sub.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure

figure, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



xlabels = [x for x in df_sub.columns]

ax.set_xticklabels(xlabels, rotation=90, fontsize=12)

ylabels= [x for x in df_sub.columns]

ax.set_yticklabels(ylabels,fontsize=12)



#ax.set_title('Correlations among Attack, Threat & Flee', fontsize=20)



plt.show()