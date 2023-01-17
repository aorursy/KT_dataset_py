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
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime
data = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')

data.head()
data.shape
data.info()
data = data.loc[data['content_duration'].str.contains('hour')|data['content_duration'].str.contains('min')]

data.shape
data['content_duration'].unique()
def text2cas(cas):

    cas1 = cas.replace('hour', '')

    cas2 = cas1.replace('s', '')

    cas3 = cas2.replace('min', '')

    return float(cas3)
data['content_duration'] = data['content_duration'].apply(text2cas)
data['content_duration'].unique()

data.info()
data['price'].unique()
def free_course(course):

    course = course.replace('Free', '0')

    return course

data['price'] = data['price'].apply(free_course)

data['price'].unique()

data['price'] = data['price'].apply(lambda x: float(x))
data['price'].unique()
data['published_timestamp'] = pd.to_datetime(data['published_timestamp'], format = '%Y-%m-%dT%H:%M:%SZ')
data.head()
#splitting the published_timestamp to just date

data['date'] = pd.to_datetime(data['published_timestamp'].dt.date, format = '%Y-%m-%d')
data.head()
categorical = [var for var in data.columns if data[var].dtype == 'object']
print('the categorical variables are: \n\n', categorical)
#we have to fix the is_paid column

data['is_paid'] = data['is_paid'].replace({'TRUE':'True', 'FALSE':'False'})

data['is_paid'].unique()
for var in categorical:

    print(data[var].value_counts())
ax = data['is_paid'].value_counts().plot(kind = 'bar', figsize= (10,8))

ax.set_title("Amount every course in udemy is paid", fontsize = 22, fontweight = 'bold')

ax.set_xlabel('Paid')
ax = data['subject'].value_counts().plot(kind ='bar',

                                        figsize = (10,5),

                                        color = 'g',

                                        width = 0.6,

                                        alpha = 0.7)
ax = data['level'].value_counts().plot(kind ='bar',

                                        figsize = (10,5),

                                        color = 'm',

                                        width = 0.6,

                                        alpha = 0.7)

ax.set_xlabel('Levels', fontsize = 20)
grouped = data.groupby(['level', 'subject'])

grouped_pct = grouped['course_id']

grouped_pct.agg('describe')
#explore numerical data

numerical = [var for var in data.columns if data[var].dtype != 'object']

print(numerical)
(data.corr()*100).round(1)

ax = plt.matshow(data.corr(), cmap = 'Reds')

plt.colorbar(ax)
ax = sns.boxplot(data['price'], orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribuition from courses', fontsize  = 24)
ax = sns.distplot(data['price'])

ax.figure.set_size_inches(12,6)

ax.set_xlabel('Prices')

ax.set_title('Distribution', fontsize = 22)
ax = sns.boxplot(x = 'level', y = 'price', data = data, orient = 'v', width =0.3)

ax.figure.set_size_inches(12,6)

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribution of price with level', fontsize  = 24)

ax.set_xlabel('Level', fontsize= 20)
ax = sns.boxplot(x = 'subject', y = 'price', data = data, orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribution of price with subject', fontsize  = 24)

ax.set_xlabel('Subject', fontsize= 20)
ax = sns.pairplot(data, y_vars = 'price', x_vars = ['num_subscribers', 'num_reviews', 'num_lectures','content_duration'], height =5, kind = 'reg')

ax.fig.suptitle('Scatter plot with variables', fontsize =20 , y = 1.05)