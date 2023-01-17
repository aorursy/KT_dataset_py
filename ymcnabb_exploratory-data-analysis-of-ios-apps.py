### My code is very much inspired by YouHan Lee (https://www.kaggle.com/youhanlee) and adityapatil 
### (https://www.kaggle.com/adityapatil673). Cheers, guys!

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set(font_scale=1.8)
import plotly.offline as py
import plotly.figure_factory as ff

from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import missingno as msno
import random

import os
print(os.listdir("../input"))
# Import App data
df_app = pd.read_csv('../input/AppleStore.csv')
# Import App description
df_desc = pd.read_csv('../input/appleStore_description.csv')
# Check if there are any null data in the app data frame
df_app.isnull().sum()
# Check if there are any null data in the description data frame 
df_desc.isnull().sum()
# Check the first few rows of both data frames
print(df_app.head())
print(df_desc.head())
# Delete 0th column
df_app = df_app.iloc[:, 1:]
# Merge the two data frames on `id`, essentially adding the app_desc column to df_app
df_app['app_desc'] = df_desc['app_desc']
print(df_app.head())
# Adding a column of App size in MB for readability
df_app['size_bytes_in_MB'] = df_app['size_bytes'] / (1024 * 1024.0)
# Adding a 'paid' column, where 1 is 'paid' and 0 is 'free'
df_app['paid'] = df_app['price'].apply(lambda x: 1 if x > 0 else 0)
df_app['paid'].value_counts().plot.bar()
plt.xticks(np.arange(2), ('Free', 'Paid'),rotation=0)
plt.xlabel('Type of App')
plt.ylabel('Count')
plt.title('Number of free vs. paid Apps')
plt.show()
# Divide the free and paid apps into two data frames
df_app_paid = df_app[df_app['paid'] == 1]
df_app_free = df_app[df_app['paid'] == 0]

print(df_app_paid['price'].describe(percentiles=[.05,.25, .5, .75, .95, .99]))

plt.figure(figsize=(14,5))
plt.style.use('seaborn-muted')
plt.hist(df_app_paid['price'], bins = 100, density=True)
plt.xlabel('Price of App')
plt.ylabel('Probability')
plt.margins(0.02)
plt.show()
#fact generator 
print ('Number of super expensive Apps: ' + str(sum(df_app.price > 50)))
print (' -  which is around ' + str(sum(df_app.price > 50)/len(df_app.price)*100) +
       " % of the total Apps")
print (' Thus we will dropping the following apps')
df_app_paid_outliers = df_app[df_app.price>50][['track_name','price','prime_genre','user_rating']]
df_app_paid_outliers
# Removing outliers
df_app_paid = df_app[((df_app.price<50) & (df_app.price>0))]
print('Now the max price of any app in new data is : ' + str(max(df_app_paid.price)))
print('Now the min price of any app in new data is : ' + str(min(df_app_paid.price)))
print(df_app_paid['price'].describe(percentiles=[.05,.25, .5, .75, .95, .99]))

plt.figure(figsize=(14,5))
plt.style.use('seaborn-muted')
plt.hist(df_app_paid['price'], bins = 25, log=True)
plt.xlabel('Price of App')
plt.ylabel('Probability')
plt.margins(0.02)
plt.show()
k2, p = stats.normaltest(df_app_paid['price'])
alpha = 1e-3
print("p = {:g}".format(p))
if p < alpha: # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
print('There are {} App genres'.format(len(df_app.prime_genre.unique())), 'and they are as follows:')
print('Here is the breakdown per category:')
print(df_app.prime_genre.value_counts())
cnt_per_gnr = df_app['prime_genre'].value_counts()
prcnt_per_gnr = ['{:.2f}%'.format(100 * (value / cnt_per_gnr.sum())) for value in cnt_per_gnr.values]

trace = go.Bar(
    x = cnt_per_gnr.index,
    y = cnt_per_gnr.values,
    text = prcnt_per_gnr,
    opacity = 0.7
)
data = [trace]

layout = go.Layout(
    title = 'Apps per genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Count'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# Free apps
free_gnr_cnts = df_app_free['prime_genre'].value_counts()
free_gnr_prcnt = ['{:.2f}%'.format(100 * (value / free_gnr_cnts.sum())) for value in free_gnr_cnts.values]

trace1 = go.Bar(
    x = free_gnr_cnts.index,
    y = free_gnr_cnts.values,
    text = free_gnr_prcnt,
    opacity = 0.7,
    name='Free'
)

paid_gnr_cnts = df_app_paid['prime_genre'].value_counts()
paid_gnr_prcnt = ['{:.2f}%'.format(100 * (value / paid_gnr_cnts.sum())) for value in paid_gnr_cnts.values]

trace2 = go.Bar(
    x = paid_gnr_cnts.index,
    y = paid_gnr_cnts.values,
    text = paid_gnr_prcnt,
    opacity = 0.7,
    name='Paid'
)

data = [trace1, trace2]

layout = go.Layout(
    title = 'Apps per genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Count'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
yrange = [0,25]
fsize =15

plt.figure(figsize=(15,10))
plt.suptitle("Price range (up to $25) of Apps per Genre (incl. four most popular genres)")

plt.subplot(4,1,1)
plt.xlim(yrange)
paid_games = df_app_paid[df_app_paid.prime_genre=='Games']
sns.stripplot(data=paid_games,y='price',jitter= True , orient ='h',size=6,color='#eb5e66')
plt.title('Games',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,2)
plt.xlim(yrange)
paid_ent = df_app_paid[df_app_paid.prime_genre=='Entertainment']
sns.stripplot(data=paid_ent,y='price',jitter= True ,orient ='h',size=6,color='#ff8300')
plt.title('Entertainment',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,3)
plt.xlim(yrange)
paid_edu = df_app_paid[df_app_paid.prime_genre=='Education']
sns.stripplot(data=paid_edu,y='price',jitter= True ,orient ='h' ,size=6,color='#20B2AA')
plt.title('Education',fontsize=fsize)
plt.xlabel('') 

plt.subplot(4,1,4)
plt.xlim(yrange)
paid_pv = df_app_paid[df_app_paid.prime_genre=='Photo & Video']
sns.stripplot(data=paid_pv,y='price',jitter= True  ,orient ='h',size=6,color='#b84efd')
plt.title('Photo & Video',fontsize=fsize)
plt.xlabel('') 

plt.show()
# Reducing the number of genre categories

s = df_app.prime_genre.value_counts().index[:4]
def categ(x):
    if x in s:
        return x
    else : 
        return "Other"

df_app['broad_genre']= df_app.prime_genre.apply(lambda x : categ(x))

# Redoing the two data frames
df_app_free = df_app[df_app['paid'] == 0]
df_app_paid = df_app[((df_app.price<50) & (df_app.price>0))]
plt.hist(df_app['user_rating'], bins = 12, density=True)
plt.xlabel('App rating')
plt.ylabel('Probability')
plt.margins(0.02)
plt.show()

df_app['user_rating'].describe(percentiles=[.05,.25, .5, .75, .95, .99])
mean_rating_per_gnr_free = df_app_free[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)
#text1 = ['{:.2f}%'.format(100 * (value / cnt_srs1.sum())) for value in cnt_srs1.values]

mean_rating_per_gnr_paid = df_app_paid[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)
#text2 = ['{:.2f}%'.format(100 * (value / cnt_srs2.sum())) for value in cnt_srs2.values]

overall_mean_rating_free = pd.Series(index=mean_rating_per_gnr_free.index)
# Boradcast the overall mean across genres to plot one line
overall_mean_rating_free.iloc[:] = df_app_free['user_rating'].mean()

overall_mean_rating_paid = pd.Series(index=mean_rating_per_gnr_paid.index)
# Boradcast the overall mean across genres to plot one line
overall_mean_rating_paid.iloc[:] = df_app_paid['user_rating'].mean()

overall_mean_rating = pd.Series(index=mean_rating_per_gnr_paid.index)
# Boradcast the overall mean across genres to plot one line
overall_mean_rating.iloc[:] = df_app['user_rating'].mean()


trace1 = go.Bar(
    x = mean_rating_per_gnr_free.index,
    y = mean_rating_per_gnr_free.values,
    opacity = 0.7,
    name='Free'
)


trace2 = go.Bar(
    x = mean_rating_per_gnr_paid.index,
    y = mean_rating_per_gnr_paid.values,
    opacity = 0.7,
    name='Paid'
)

trace3 = go.Scatter(
    x = overall_mean_rating_free.index,
    y = overall_mean_rating_free.values,
    opacity = 0.7,
    name='Overall mean free'
)

trace4 = go.Scatter(
    x = overall_mean_rating_paid.index,
    y = overall_mean_rating_paid.values,
    opacity = 0.7,
    name='Overall mean paid'
)

trace5 = go.Scatter(
    x = overall_mean_rating.index,
    y = overall_mean_rating.values,
    opacity = 0.7,
    name='Overall mean'
)

data = [trace1, trace2, trace3, trace4, trace5]

layout = go.Layout(
    title = 'Mean rating per genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Mean rating',
        range=[0, 5]
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sns.lmplot(x= 'paid', y='user_rating', data=df_app, x_estimator=np.mean)
plt.ylim(0,5)
plt.xticks([0,1],('Free', 'Paid'))
plt.xlabel('Type of App')
plt.ylabel('Rating')
plt.margins(0.02)
plt.show()
plt.hist(df_app['sup_devices.num'], bins = 40, density=True)
plt.xlabel('Number of supported devices')
plt.ylabel('Probability')
plt.margins(0.02)
plt.show()
sns.lmplot(x='sup_devices.num', y='user_rating', data=df_app, x_jitter=.05)
plt.xlabel('Number of supported devices')
plt.ylabel('Mean rating')
plt.margins(0.02)
plt.show()
sns.lmplot(data=df_app,
           x='sup_devices.num',y='user_rating',size=4, aspect=2,col_wrap=2,hue='broad_genre',
           col='broad_genre',fit_reg=False)
plt.show()
print('Here are the summary statistics for the number of supported languages:')
print(df_app['lang.num'].describe())
print('There are {} cases of 0 supported languages.'.format(len(df_app[df_app['lang.num']==0])))
# First: correct the 0 languages to 1 language in `lang.num'
df_app.loc[df_app['lang.num'] == 0, 'lang.num'] = 1
print('There are {} cases of 0 supported languages.'.format(len(df_app[df_app['lang.num']==0])))
print('There are {} cases of 1 supported language.'.format(len(df_app[df_app['lang.num']==1])))
# Second: look at the new descriptive statistics:
print('Here are the new summary statistics for the number of supported languages:')
print(df_app['lang.num'].describe(percentiles=[.05,.25, .5, .75, .95, .99]))

plt.figure(figsize=(14,5))
plt.hist(df_app['lang.num'], bins = 75, density=True)
plt.xlabel('Number of supported languages')
plt.ylabel('Probability')
plt.margins(0.02)
plt.show()
sns.lmplot(data=df_app,
           x='lang.num',y='user_rating',size=4, aspect=2,col_wrap=2,hue='broad_genre',
           col='broad_genre',fit_reg=False)
plt.show()
print('Here are the summary statistics for App size (in MB):')
print(df_app['size_bytes_in_MB'].describe(percentiles=[.05,.25, .5, .75, .95, .99]))
plt.figure(figsize=(14,5))
plt.hist(df_app['size_bytes_in_MB'], bins = 100, log=True)
plt.xlabel('App size (MB)')
plt.ylabel('Log frequency')
plt.margins(0.02)
plt.show()
sns.lmplot(x='size_bytes_in_MB', y='user_rating', data=df_app, aspect=2.5)
plt.xlabel('App size')
plt.ylabel('Mean rating')
plt.margins(0.02)
plt.show()
k2, p = stats.normaltest(df_app['size_bytes_in_MB'])
alpha = 1e-3
print("p = {:g}".format(p))
if p < alpha: # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
sns.lmplot(data=df_app,
           x='size_bytes_in_MB',y='user_rating',size=4, aspect=2,col_wrap=2,hue='broad_genre',
           col='broad_genre',fit_reg=False)
plt.show()