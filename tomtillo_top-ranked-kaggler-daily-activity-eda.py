from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import warnings

warnings.filterwarnings("ignore")



import pandas as pd 

import time

import datetime



import seaborn as sns 

import os



import matplotlib.pyplot as plt



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# A function to convert day of week (number) to String 

def get_weekday(dow): 

    dow_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

    return dow_dict[dow]
df_activity = pd.read_csv('/kaggle/input/top-ranked-kaggle-user-activity-1-1000-ranks/USER_ACTIVITY.csv')

df_activity.sample(7)
df_activity.dtypes # check the type 

df_activity['date']= pd.to_datetime(df_activity['date']) # convert to datetime format 

today_= datetime.datetime.strptime(str(datetime.date.today()), "%Y-%m-%d")

yearback_ = datetime.datetime.strptime(str(datetime.date.today().replace(year=today_.year - 1)), "%Y-%m-%d")

activity_year = df_activity[df_activity['date'] >= yearback_]
activity_year['month'] = activity_year.date.apply(lambda x: x.month)

activity_year['wk_day'] = activity_year.date.apply(lambda x:get_weekday(x.weekday()))
activity_year.sample(5)
df_month_grouping = activity_year.groupby('month').sum()

df_month_grouping = df_month_grouping.reset_index().sort_values(by = 'month')

sns.barplot(data = df_month_grouping, x='month',y='comments')

plt.show();
sns.barplot(data = df_month_grouping, x='month',y='submissions')

plt.show();
df_user_grouping = activity_year.groupby('username').sum().reset_index()

df_user_grouping.head()
sns.barplot(data = df_user_grouping.sort_values(by = 'submissions', ascending  = False).head(10), \

            x='submissions',y='username')

plt.show();
sns.barplot(data = df_user_grouping.sort_values(by = 'comments', ascending  = False).head(10), x='comments',y='username')

plt.show();
sns.barplot(data = df_user_grouping.sort_values(by = 'scripts', ascending  = False).head(10), x='scripts',y='username')

plt.show();
#Removing the user - 'kerneler'

df_user_grouping =df_user_grouping[df_user_grouping['username'] != 'kerneler']
sns.barplot(data = df_user_grouping.sort_values(by = 'scripts', ascending  = False).head(10), x='scripts',y='username')

plt.show();