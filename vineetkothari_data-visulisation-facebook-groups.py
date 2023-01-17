# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#reading comments/posts/like/member 

comment_df = pd.read_csv('../input/comment.csv')

like_df = pd.read_csv('../input/like.csv')

member_df = pd.read_csv('../input/member.csv')

post_df = pd.read_csv('../input/post.csv')
#print the file in a tabular format with the help of head method similar to unix head and tail commands

comment_df.head()
like_df.head()
post_df.head()
member_df.head()
#which user was active initally accordint to the timestamp

comment_df['timeStamp'].head()
#Set df['date'] as the index and delete the column

comment_df['timeStamp'] =pd.to_datetime(comment_df.timeStamp)

comment_df.index = comment_df['timeStamp']

comment_df.head()
comment_df['2014'].head()
#sort out data according to hours

times = pd.DatetimeIndex(comment_df.timeStamp)

comment_df['hour']=times.hour

comment_df.head()
#arranging data

com_count = comment_df.groupby('pid').count()['hour']

data = post_df.join(com_count,on='pid', rsuffix='c')[['name','likes', 'shares', 'hour', 'gid']]

data.columns = ['name', 'likes','shares', 'hour','gid']

data.head()
#replacing NAN values with zeros

data.fillna(0,inplace=True)

data.head()
#ploting a graph b/w hour/likes/name

hours=data['hour'].head()

likes=data['likes'].head()

name=data['name'].head()

label=[]

#truncating names to fit in graph

for i in name:

    label.append(i[0:5])

print (label)
#plotting a scater plot for likes and name

plt.scatter(hours,likes)

#label each plot

for label_names,hour_count,likes_count in zip(label,hours,likes):

    plt.annotate(label_names,  

    xy=(hour_count,likes_count),#put the label with its point

    xytext=(5,-5),                   #but slightly offset

    textcoords='offset points')

    

plt.title("Daily hour vs likes")

plt.ylabel("# of hours")

plt.xlabel("# of likes")

plt.show()
#name having max likes

g1 = data.head().groupby(['name','likes'], sort=True)['likes'].max()

#create a line chart , likes on x axis 

g1
plt.xticks(likes, label)

plt.plot(likes,hours)

plt.show()
