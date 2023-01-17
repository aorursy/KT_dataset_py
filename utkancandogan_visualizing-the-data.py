# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(font_scale=1.5)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





df = pd.read_csv('../input/HR_comma_sep.csv')

df.rename(columns = {'sales':'position'}, inplace = True)

print(df.head())



print(df.describe())

# Any results you write to the current directory are saved as output.
print(sum(df['Work_accident']==0))
fig,ax = plt.subplots(4,2,figsize=(15,15))

plt.tight_layout( h_pad = 2.5)

# sns.boxplot(data=df, x ='left', y= 'satisfaction_level' ,orient='v',ax=ax[0,0])

# sns.boxplot(data=df, x ='left', y= 'last_evaluation' ,orient='v',ax=ax[0,1])

# sns.boxplot(data=df, x ='left', y= 'number_project' ,orient='v',ax=ax[1,0])

# sns.boxplot(data=df, x ='left', y= 'average_montly_hours' ,orient='v',ax=ax[1,1])

# sns.boxplot(data=df, x ='left', y= 'time_spend_company' ,orient='v',ax=ax[2,0])



# sns.boxplot(data=df, x ='left', y= 'Work_accident' ,orient='v',ax=ax[2,1])

# sns.boxplot(data=df, x ='left', y= 'promotion_last_5years' ,orient='v',ax=ax[3,0])



sns.countplot(data=df, x ='left', hue= 'Work_accident' ,orient='v',ax=ax[0,0])

sns.countplot(data=df, x ='left', hue= 'promotion_last_5years' ,orient='v',ax=ax[0,1])

sns.countplot(data=df, x ='salary', hue= 'left' ,orient='v',ax=ax[1,0])









sns.kdeplot( df[df['left']==1]['time_spend_company'], shade=1, ax=ax[1,1],label='left = 1',legend='Work_accident' )

sns.kdeplot( df[df['left']==0]['time_spend_company'], shade=1, ax=ax[1,1],label='left = 0' ,legend='Work_accident' )

ax[1,1].set_title('KDE for time_spend_company')





sns.kdeplot( df[df['left']==1]['last_evaluation'], shade=1, ax=ax[2,0],label='left = 1',legend='Work_accident' )

sns.kdeplot( df[df['left']==0]['last_evaluation'], shade=1, ax=ax[2,0],label='left = 0' ,legend='Work_accident' )

ax[2,0].set_title('KDE for last_evaluation')



sns.kdeplot( df[df['left']==1]['satisfaction_level'], shade=1, ax=ax[2,1],label='left = 1',legend='Work_accident' )

sns.kdeplot( df[df['left']==0]['satisfaction_level'], shade=1, ax=ax[2,1],label='left = 0' ,legend='Work_accident' )

ax[2,1].set_title('KDE for satisfaction_level')



sns.kdeplot( df[df['left']==1]['number_project'], shade=1, ax=ax[3,0],label='left = 1',legend='Work_accident' )

sns.kdeplot( df[df['left']==0]['number_project'], shade=1, ax=ax[3,0],label='left = 0' ,legend='Work_accident' )

ax[3,0].set_title('KDE for number_project')



sns.kdeplot( df[df['left']==1]['average_montly_hours'], shade=1, ax=ax[3,1],label='1')

sns.kdeplot( df[df['left']==0]['average_montly_hours'], shade=1, ax=ax[3,1],label='0')

ax[3,1].set_title('KDE for average_montly_hour')

plt.figure(figsize=(10, 10))

sns.heatmap(df.corr(),linewidths=2)

sns.plt.title('Heatmap of Correlation Matrix')

sns.lmplot(data = df, x='satisfaction_level',y='last_evaluation',hue='left',fit_reg=False,size=10)
