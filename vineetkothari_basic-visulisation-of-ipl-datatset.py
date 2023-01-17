# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read CSV (comma-separated) file into DataFrame return result: DataFrame or TextParser

delieveries_df=pd.read_csv("../input/deliveries.csv")

matches_df=pd.read_csv("../input/matches.csv")
#return top n columns DataFrame.head(n=5)

delieveries_df.head()
matches_df.head()
#dataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False, **kwargs)

delieveries_sum=delieveries_df.groupby('over',axis=0).sum()

delieveries_sum.head()
#np.arange(0,n) 

over=np.arange(1,21)

total_sum=delieveries_sum.total_runs
#realtion ship between runs and overs 

plt.plot(over,total_sum)

plt.xlabel('overs')

plt.ylabel('runs')

plt.title('over vs runs')

plt.show()
plt.plot(over,total_sum,'ro')

plt.xlabel('overs')

plt.ylabel('runs')

plt.title('over vs runs')

plt.show()
# red dashes, blue squares and green triangles

#Normalising total_sum

plt.bar(over,total_sum)

plt.show()
#group according o batman

delieveries_batsman=delieveries_df.groupby('batsman').sum()

delieveries_batsman.head()
#ploting a graph to see max 

#DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

batsman_maximum_runs=delieveries_batsman.sort_values('total_runs',ascending=False)

total=batsman_maximum_runs['total_runs'].values[0:10]

over=(batsman_maximum_runs['over'].values)[0:10]

name_batsman=batsman_maximum_runs.index.values

labels=name_batsman[0:10]
fig, ax = plt.subplots() #helps to create various plots inside plots

ax.plot(total,over, 'ro')#plotting a scatter plot

ax.plot(total, over)#ploting a line plot

ax.bar(total,over)#ploting a bar plot 

ax.set_title('Total runs vs over')

plt.xlabel('total')

plt.ylabel('overs')

plt.xticks(total, labels, rotation='vertical')#setting names as label to x axis

plt.show()