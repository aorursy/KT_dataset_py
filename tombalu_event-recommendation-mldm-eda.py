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
train_df = pd.read_csv('/kaggle/input/event-recommendation-engine-challenge/train.csv')
train_df.head()
#check missing values
train_df.isnull().sum()
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
train_df.shape
# To
train_df.event.unique()
# Find no of users interested in each event
train_df.event[train_df['interested']==1].value_counts()
chart = sns.countplot(train_df.event,
              order=train_df.event.value_counts().iloc[:10].index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
# No. of events
train_df.event[train_df['invited']==1].value_counts()

plt.figure(figsize=(10,8))
plt.title('Barplot of Activity')
sns.countplot(train_df.event)
plt.xticks(rotation=90)
sns.heatmap(train_df.corr())
events = pd.read_csv('/kaggle/input/event-recommendation-engine-challenge/events.csv.gz', compression = 'gzip')
events.head()
events.shape
events.describe()
events.event_id.value_counts().sum()
events.isnull().sum()
print(events.city.value_counts())

event_attendees = pd.read_csv('/kaggle/input/event-recommendation-engine-challenge/event_attendees.csv.gz', compression= 'gzip')
event_attendees.head()
event_attendees.yes[0]
# Change string of attendees into list (yes attribute)
for i in range(event_attendees.shape[0]):
    attendees = list()
    attendees = str(event_attendees.yes[i]).split()
    event_attendees.yes[i]=attendees
    print(event_attendees.event[i], attendees)
event_attendees.head()
# Change string of attendees into list (maybe attribute)
for i in range(event_attendees.shape[0]):
    attendees = list()
    attendees = str(event_attendees.maybe[i]).split()
    event_attendees.maybe[i]=attendees
#     print(event_attendees.event[i], attendees)

# Change string of attendees into list (no attribute)
for i in range(event_attendees.shape[0]):
    attendees = list()
    attendees = str(event_attendees.no[i]).split()
    event_attendees.no[i]=attendees
#     print(event_attendees.event[i], attendees)    

# Change string of attendees into list (invited attribute)
for i in range(event_attendees.shape[0]):
    attendees = list()
    attendees = str(event_attendees.invited[i]).split()
    event_attendees.invited[i]=attendees
#     print(event_attendees.event[i], attendees)

users = pd.read_csv('/kaggle/input/event-recommendation-engine-challenge/users.csv')
users.head()
users.shape
users.isnull().sum()
users.location.value_counts()
import seaborn as sns
# plt.figure(figsize=(10,5))

chart = sns.countplot(users.location,
              order=users.location.value_counts().iloc[:10].index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
users.locale.value_counts()
chart = sns.countplot(users.locale,
              order=users.locale.value_counts().iloc[:10].index)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

user_friends = pd.read_csv('../input/event-recommendation-engine-challenge/user_friends.csv.gz', compression = 'gzip')
user_friends.head()
# Change string of friends into list
for i in range(user_friends.shape[0]):
    friends = list()
    friends = str(user_friends.friends[i]).split()
    user_friends.friends[i]=friends
user_friends.head()
user_friends.shape
