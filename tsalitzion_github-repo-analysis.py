import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import seaborn as sns

from ggplot import *

plt.style.use('default')

git_df = pd.read_csv("../input/TopStaredRepositories.csv", parse_dates=['Last Update Date'], dayfirst=True)

git_df.head()
git_df.info()
git_df_max = git_df['Number of Stars'].str.contains('k').all()

git_df_max
git_df['Number of Stars']=git_df['Number of Stars'].str.replace('k','').astype(float)
git_df.head()
git_df.tail()
git_df['Number of Stars'].describe()
popular_repos= git_df[git_df['Number of Stars'] > 13.0]

len(popular_repos)
popular_repos.head(8)
popular_repos.tail(8)
# classifying repositories according to the popularity

classified_repos=[]

for i in range(8,300,7):

    x = git_df[(git_df['Number of Stars'] >= i) & (git_df['Number of Stars'] <(i+7.0))]

    classified_repos.append(len(x))
indexes = []



for i in range (8000,300000, 7000):

    x = '[' + str(i) +','+ (str(i+7000)) + ')'

    indexes.append(x)

divided_repos = pd.Series(data=classified_repos, index=indexes)

divided_repos.plot(kind='bar', figsize=(15,10), color=['red'],legend=True, label='Number of repositories')
x=git_df['Language'].value_counts()

x.head()

#p = ggplot(aes(x='index',y='count'), data =x) + geom_point(color='coral') + geom_line(color='red')

#print(p)
%matplotlib inline

plt.figure()

x.plot(kind='barh',figsize=(15,10),grid=True, label='Number of repositories',legend='No of repos',title='No of repositories vs language used')

%matplotlib inline

x[:5].plot.pie(label="Division of the top 5 languages",fontsize=10,figsize=(10,10),legend=True)
%matplotlib inline

x[:20].plot.pie(label="Division of the top 20 languages",fontsize=10,figsize=(10,10),legend=True)
#git_df['Number of Stars']=git_df['Number of Stars'].str.replace('k','').astype(float)

nonull_df = git_df[['Tags','Number of Stars']].dropna()

tags_list = nonull_df['Tags'].str.split(',')
tags_list.head()
initial = nonull_df['Tags'].str.split(',')

a = []

for item in initial:

       a = a+item

wc_text = ' '.join(a)



%matplotlib inline

wordcloud = WordCloud(background_color='black',width=800, height=400).generate(wc_text)

plt.figure(figsize=(25,10), facecolor='k')

plt.imshow(wordcloud, interpolation='bilinear')

plt.tight_layout(pad=0)

plt.axis("off")
web_dev_count = 0

tags = ['javascript', 'css', 'html', 'nodejs', 'bootstrap','react', 'react-native', 'rest-api', 'rest', 'web-development','typescript','coffeescript']

for item in tags_list:

    if set(tags).intersection(item):

        web_dev_count+=1

web_dev_count
machine_data_count=0

mach=[]

tags=['machine-learning', 'jupyter','jupter-notebook', 'tensorflow','data-science','data-analytics']

for item in tags_list:

    if set(tags).intersection(item):

        machine_data_count+=1

        mach.append(item)

machine_data_count
mobile_dev_count=0

tags=['android','sdk','ios','swift','mobile','react','macos','windows']

for item in tags_list:

    if set(tags).intersection(item):

        mobile_dev_count+=1

mobile_dev_count
linux_dev_count=0

linux=[]

tags=['linux','unix','bash','shell','cli','bsd']

for item in tags_list:

    if set(tags).intersection(item):

        linux_dev_count+=1

        linux.append(item)

linux_dev_count
hardware_dev_count=0

hardware=[]

tags=['hardware','iot','smart','system','system-architecture','cloud']

for item in tags_list:

    if set(tags).intersection(item):

        hardware.append(item)

        hardware_dev_count+=1

hardware_dev_count
domain_series=pd.Series(index=['Web Development','Data Science and Machine Learning','Mobile Development','Linux and Shell Programming','System hardware and IOT'],

                        data=[web_dev_count,machine_data_count,mobile_dev_count,linux_dev_count,hardware_dev_count])
domain_series
%matplotlib inline

fig_domain=domain_series.plot(lw=2,kind='barh',figsize=(20,10),color=['green'],grid=True,title='Domain-wise repository analysis',

                              )

fig_domain.set(xlabel="Number of repositories", ylabel="Domain Name")
nonull_df['CountTag']=0

for i in range(0,489,1):

    nonull_df['CountTag'].iloc[i] = len(list(nonull_df['Tags'].iloc[i].split(',')))

nonull_df['CountTag'].corr(nonull_df['Number of Stars'])
python_tags = git_df[git_df['Language'] == 'Python'][['Username', 'Repository Name', 'Description', 'Tags']]
python_tags