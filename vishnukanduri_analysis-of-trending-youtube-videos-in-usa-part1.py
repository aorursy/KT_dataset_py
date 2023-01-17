# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#reading the input CSV file and storing it in a dataframe

df = pd.read_csv("../input/youtube-new/USvideos.csv")
#shape of the dataframe (dataset)

df.shape
#statistical information about the data

df.describe()
#checking for null values. It says increase the count by 1 whenever you find a null value in each column

df.isnull().sum()
#number of unique values (no duplicates are counted)

df.nunique()
#data type of each column

df.dtypes
#datatype and number of non-null values in each column

df.info()
#prints first five rows of the dataframe

df.head()
#defining the size of the figure

plt.figure(figsize = (20,4))



#Important note: 1st digit in subplot indicates the number of rows in the plot, 2nd digit indicates the number of columns in the plot, 3rd digit indicates the index

#So, we use 14 as we want one row and 4 columns.

plt.subplot(141)

p1 = sns.distplot(df['likes'], color='green')

plt.xticks(rotation='vertical')

p1.set_title('Distribution of number of likes')



plt.subplot(142)

p2 = sns.distplot(df['dislikes'], color='red')

p2.set_title('Distribution of number of dislikes')



plt.subplot(143)

p3 = sns.distplot(df['views'], color='blue')

p3.set_title('Distribution of number of views')



plt.subplot(144)

p4 = sns.distplot(df['comment_count'], color='yellow')

p4.set_title('Distribution of number of comments')



plt.subplots_adjust(wspace = 0.4)

plt.xticks(rotation='vertical')

plt.show()
#defining the size of the figure

plt.figure(figsize = (20,4))



df['likes_log'] = np.log(df['likes']+1)

df['dislikes_log'] = np.log(df['dislikes']+1)

df['views_log'] = np.log(df['views']+1)

df['comments_log'] = np.log(df['comment_count']+1)



plt.subplot(141)

p1 = sns.distplot(df['likes_log'], color='green')

p1.set_title('Log distribution of number of likes')



plt.subplot(142)

p2 = sns.distplot(df['dislikes_log'], color='red')

p2.set_title('Log distribution of number of dislikes')



plt.subplot(143)

p3 = sns.distplot(df['views_log'], color='blue')

p3.set_title('Log distribution of number of views')



plt.subplot(144)

p4 = sns.distplot(df['comments_log'], color='yellow')

p4.set_title('Log distribution of number of comments')



plt.show()
#adding a column with all entries 'None'

df['likesl'] = 'None'

df['dislikesl'] = 'None'



#dropping one column

df.drop('likesl', axis=1)



#dropping multiple columns (columns keyword is optional). Both the below statements yield the same result

df.drop(['dislikesl', 'likesl'], axis=1)

df.drop(columns=['dislikesl', 'likesl'], axis=1)
df.head()
df.head()
#Percentage of likes, dslikes and comments

df['like_rate'] =  df['likes'] / df['views'] * 100

df['dislike_rate'] =  df['dislikes'] / df['views'] * 100

df['comment_rate'] =  df['comment_count'] / df['views'] * 100
#defining the size of the figure

plt.figure(figsize = (20,4))



plt.subplot(131)

p1 = sns.distplot(df['like_rate'], color='green')

plt.xticks(rotation='vertical')

p1.set_title('Like rate')



plt.subplot(132)

p2 = sns.distplot(df['dislike_rate'], color='red')

p2.set_title('Dislikes rate')



plt.subplot(133)

p3 = sns.distplot(df['comment_rate'], color='blue')

p3.set_title('Views rate')



plt.subplots_adjust(wspace = 0.4)

plt.xticks(rotation='vertical')

plt.show()
#Countplots

# sns.set(style="darkgrid")

# ax = sns.countplot(x=df['like_rate'], data=df)
plt.figure(figsize = (20,6))

plt.subplot(121)

plt.scatter(df['like_rate'], df['dislike_rate'])

plt.subplot(122)

plt.scatter(df['like_rate'], df['comment_rate'])
#Correlation between views and likes

plt.figure(figsize = (20,10))

plt.scatter(df['views'], df['likes'])

plt.xlabel('Views', fontsize=26)

plt.ylabel('Likes', fontsize=26)

plt.xticks(size = 15)

plt.yticks(size = 15)
df['likes'].corr(df['views'])
df.corr(method='pearson')
sns.heatmap(df.corr())
df.corr(method ='kendall') 
plt.figure(figsize = (20,4))



plt.subplot(121)

p1 = sns.distplot(df['likes_log'], color='green')

plt.xticks(rotation='vertical')

p1.set_title('Distribution of number of likes')



plt.subplot(122)

p3 = sns.distplot(df['views_log'], color='blue')

p3.set_title('Distribution of number of views')



# plt.subplots_adjust(wspace = 0.4)

# plt.xticks(rotation='vertical')

plt.show()