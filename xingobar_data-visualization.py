# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
nrows = 1000000

df_train = pd.read_csv('../input/clicks_train.csv',nrows=nrows)

df_test = pd.read_csv('../input/clicks_test.csv')
df_train.head()
train_count = df_train.groupby(['display_id'])['ad_id'].count().value_counts()

test_count = df_test.groupby(['display_id'])['ad_id'].count().value_counts()



train_size = train_count / np.sum(train_count)

test_size = test_count / np.sum(test_count)

del train_count,test_count



fig,ax = plt.subplots(figsize=(8,6))



sns.barplot(train_size.index,train_size.values,color='gray',ax=ax,label='train')

sns.barplot(test_size.index,test_size.values,color='#80cbc4',ax=ax,label='test')

plt.legend()

plt.xlabel('Number of ad ')

plt.ylabel('Proportion of set')
ad_count = df_train.groupby(['ad_id'])['ad_id'].count()



for i in [10,50,100,1000]:

    print('Ads that appear less than {} times :{}% '.format(i,round((ad_count < i).mean() * 100,2)))



plt.figure(figsize=(8,6))

plt.hist(ad_count.values,bins=50,log=True);

plt.title('AD Distribution')

plt.xlabel('Numer of times ad appeared' )

plt.ylabel('Log Count of Ad')

plt.show()
category = pd.read_csv('../input/documents_categories.csv',nrows=nrows)
category.head()
category_count = category.groupby('category_id')['confidence_level'].count().sort_values()



for i in [5000,10000,15000,20000]:

    print('category that appeared less than {} times: {}%'.format(i,round((category_count <i).mean() * 100,2)))



plt.figure(figsize=(8,6))

plt.hist(category_count.values,bins=50,log=True)

plt.title('Category Distribtuion')

plt.xlabel('Document Category')

plt.ylabel('Total Occurrence')

plt.show()

del category_count
entity = pd.read_csv('../input/documents_entities.csv',nrows=nrows)
entity.head()
entity_count = entity.groupby('entity_id')['confidence_level'].count().sort_values()



for i in [50,100,150,500]:

    print('entity that appeared less than {} times : {}'.format(i,round((entity_count < i).mean()*100,2)))



plt.figure(figsize=(8,6))

plt.hist(entitiy_count.values,bins=50,log=True)

plt.title('Entity Distribution')

plt.xlabel('Document Entity')

plt.ylabel('Total Occurrence')

plt.show()

del entity_count
meta = pd.read_csv('../input/documents_meta.csv',nrows=nrows)
meta.head()
plt.title('Publisher Distribution')

sns.distplot(meta['publisher_id'].dropna())
sns.distplot(meta['source_id'].dropna())
topics = pd.read_csv('../input/documents_topics.csv',nrows=nrows)
topics.head()
topics_count = topics.groupby('topic_id')['confidence_level'].count().sort_values()



for i in [1500,2000,3500,4500,6000]:

    print('topic that appeared less than {} times : {}'.format(i,round((topics_count < i).mean() *100,2)))



plt.figure(figsize=(8,6))

plt.hist(topics_count.values,bins=50,log=True);

plt.title('Topic Distribution')

plt.xlabel('Topics')

plt.ylabel('Log Count Topics')

plt.show()
event = pd.read_csv('../input/events.csv')
event.head()
event.platform = event.platform.astype(str)

event_count = event.platform.value_counts()

fig,ax = plt.subplots(figsize=(12,4))

sns.barplot(event_count.index,event_count.values,ax=ax)

plt.title('Platform Count')
uuid_counts = event.groupby('uuid')['uuid'].count().sort_values()



print(uuid_counts.tail())



for i in [2, 5, 10]:

    print('Users that appear less than {} times: {}%'.format(i, round((uuid_counts < i).mean() * 100, 2)))

    

plt.figure(figsize=(12, 4))

plt.hist(uuid_counts.values, bins=50, log=True)

plt.xlabel('Number of times user appeared in set', fontsize=12)

plt.ylabel('log(Count of users)', fontsize=12)

plt.show()