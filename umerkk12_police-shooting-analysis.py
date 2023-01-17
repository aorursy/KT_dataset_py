import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
df = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
df.head()
df.info()
#find out null values
df.isnull()
#heatmap creation will tell exactly what coulmns are missing values the most.
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
df['armed'].unique()
df.drop('armed', inplace=True, axis=1)
sns.heatmap(df.isnull())
df['flee'].unique()
df[df['flee'].isnull()].count()
df['flee'].fillna('Other', inplace=True)
df['flee'].unique()
sns.heatmap(df.isnull())
df[df['age'].isnull()].count()
df['age'].groupby(df['race']).mean()
def impude_age (cols):
    age = cols[0]
    race = cols[1]
    if pd.isnull(age):
        if race == 'A':
            return 37
        elif race == 'B':
            return 32
        elif race == 'H':
            return 34
        elif race == 'N':
            return 31
        elif race == 'O':
            return 33
        else:
            return 40
    else:
        return age
        
df['age']=df[['age','race']].apply(impude_age, axis=1)
sns.heatmap(df.isnull())
df['race'].fillna('Unknown', inplace=True)
sns.heatmap(df.isnull())
df.dropna(inplace=True)
sns.heatmap(df.isnull())
df
df['age'].iplot(kind='histogram', color='red', opacity = 0.6, histfunc = 'sum', bins = 100)
ty = sns.countplot(x='race', data = df, hue= 'gender', palette = 'Set2', alpha = 0.6)
fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty.set_title('Cases based on race and Gender')
ty.set_ylabel('Number of Cases')
new = pd.get_dummies(df['manner_of_death'])
df = pd.concat([df, new], axis = 1)
df
state_wise = df['manner_of_death'].groupby(df['state']).count().sort_values(ascending = False)[0:10]
state_wise = pd.DataFrame(state_wise)
ds = state_wise.reset_index()
by_states = sns.barplot(x='state', y='manner_of_death', data = ds, palette = 'magma')
fig=plt.gcf()
fig.set_size_inches(13,10)
sns.despine(left=True)
by_states.set_title('Cases based on States')
by_states.set_ylabel('Number of Cases')
city_wise = df['manner_of_death'].groupby(df['city']).count().sort_values(ascending = False)[0:10]
cw = city_wise.reset_index()
by_city = sns.barplot(x='city', y='manner_of_death', data = cw, palette = 'coolwarm_r')
fig=plt.gcf()
fig.set_size_inches(13,10)
sns.despine(left=True)
by_city.set_title('Cases based on States')
by_city.set_ylabel('Number of Cases')

age_25 = df[(df['age']<25) & (df['shot'] == 1)]
age_25 = age_25.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()
age_35 = df[(df['age']>25) & (df['shot'] == 1) & (df['age']< 35)]
age_35 = age_35.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_45 = df[(df['age']>35) & (df['shot'] == 1) & (df['age']< 45)]
age_45 = age_45.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_55 = df[(df['age']>45) & (df['shot'] == 1) & (df['age']< 55)]
age_55 = age_55.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_65 = df[(df['age']>55) & (df['shot'] == 1) & (df['age']< 65)]
age_65 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_75 = df[(df['age']>65) & (df['shot'] == 1) & (df['age']< 75)]
age_75 = age_75.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_85 = df[(df['age']>75) & (df['shot'] == 1) & (df['age']< 85)]
age_85 = age_85.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_95 = df[(df['age']>85) & (df['shot'] == 1) & (df['age']< 95)]
age_95 = age_95.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

sns.scatterplot(x='state', y='shot', data=age_25, s = 80, label= 'Below 25')
sns.scatterplot(x='state', y='shot', data=age_35 , s = 80,label= '25-35' )
sns.scatterplot(x='state', y='shot', data=age_45, s = 80,label= '35-45')
sns.scatterplot(x='state', y='shot', data=age_55,s = 80,label= '45-55')
sns.scatterplot(x='state', y='shot', data=age_65, s = 80,label= '55-65')
sns.scatterplot(x='state', y='shot', data=age_75 ,s = 80,label= '65-75')
sns.scatterplot(x='state', y='shot', data=age_85, s = 80,label= '75-85')
ty = sns.scatterplot(x='state', y='shot', data=age_95, s = 80, label= '85-95')
fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty.set_title('Number of people Shot in top States by Age group')
ty.set_ylabel('Number of Shoots')
ty.set_xlabel('States')
age_25 = df[(df['age']<25) & (df['shot'] == 0)]
age_25 = age_25.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_35 = df[(df['age']>25) & (df['shot'] == 0) & (df['age']< 35)]
age_35 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_45 = df[(df['age']>35) & (df['shot'] == 0) & (df['age']< 45)]
age_45 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_55 = df[(df['age']>45) & (df['shot'] == 0) & (df['age']< 55)]
age_55 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_65 = df[(df['age']>55) & (df['shot'] == 0) & (df['age']< 65)]
age_65 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_75 = df[(df['age']>65) & (df['shot'] == 0) & (df['age']< 75)]
age_75 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_85 = df[(df['age']>75) & (df['shot'] == 0) & (df['age']< 85)]
age_85 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

age_95 = df[(df['age']>85) & (df['shot'] == 0) & (df['age']< 95)]
age_95 = age_65.groupby('state').count()['shot'].sort_values(ascending=False)[0:10].reset_index()

sns.scatterplot(x='state', y='shot', data=age_25, s = 80, label= 'Below 25')
sns.scatterplot(x='state', y='shot', data=age_35 , s = 80,label= '25-35' )
sns.scatterplot(x='state', y='shot', data=age_45, s = 80,label= '35-45')
sns.scatterplot(x='state', y='shot', data=age_55,s = 80,label= '45-55')
sns.scatterplot(x='state', y='shot', data=age_65, s = 80,label= '55-65')
sns.scatterplot(x='state', y='shot', data=age_75 ,s = 80,label= '65-75')
sns.scatterplot(x='state', y='shot', data=age_85, s = 80,label= '75-85')
ty = sns.scatterplot(x='state', y='shot', data=age_95, s = 80, label= '85-95')
fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty.set_title('Number of people Shot and Tasered in top States by Age group')
ty.set_ylabel('Number of Shoots')
ty.set_xlabel('States')
female_crimes = df[df['gender']=='F']
f_c = female_crimes['gender'].groupby(df['race']).count().sort_values(ascending=False)
f_c = pd.DataFrame(f_c)
f_c = f_c.reset_index()
ty= sns.barplot(x= 'race', y= 'gender', data=f_c, alpha=0.5)
fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty.set_title('Female Crime according to Race')
ty.set_ylabel('Number of Cases')


female_crimes = df['gender'] == 'F'
f_c_1 = df[female_crimes]['gender'].groupby(df['state']).count().sort_values(ascending=False)[0:10]
f_c_1 = pd.DataFrame(f_c_1)
f_c_1 = f_c_1.reset_index()
ty_1 = sns.barplot(x= 'state', y= 'gender', data=f_c_1, alpha=0.5, palette= 'coolwarm_r')

fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty_1.set_title('Female Crime according to States')
ty_1.set_ylabel('Number of Cases')

female_crimes_2 = df['gender'] == 'F'
f_c_2 = df[female_crimes]['gender'].groupby(df['city']).count().sort_values(ascending=False)[0:10]
f_c_2 = pd.DataFrame(f_c_2)
f_c_2 = f_c_2.reset_index()
ty_2 = sns.barplot(x= 'city', y= 'gender', data=f_c_2, alpha=0.5, palette= 'magma_r')

fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty_2.set_title('Female Crime according to Cities')
ty_2.set_ylabel('Number of Cases')

illness_by_race = df[df['signs_of_mental_illness'] == True].groupby(df['race']).count()['signs_of_mental_illness'].sort_values(ascending=False)
illness_by_race = pd.DataFrame(illness_by_race)
illness_by_race = illness_by_race.reset_index()
ty_2 = sns.barplot(x= 'race', y= 'signs_of_mental_illness', data=illness_by_race, alpha=0.8, palette= 'Spectral')

fig=plt.gcf()
fig.set_size_inches(13,5)
sns.despine(left=True)
ty_2.set_title('Mental Illness Shooting cases by Race (Total Reported = 1215)')
ty_2.set_ylabel('Number of Cases')
ty_2.set_xlabel('Race')

df[df['signs_of_mental_illness'] == True].count()