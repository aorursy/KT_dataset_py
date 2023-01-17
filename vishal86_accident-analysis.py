# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/Traffic accidents by month of occurrence 2001-2014.csv')

df2=pd.read_csv('../input/Traffic accidents by time of occurrence 2001-2014.csv')
df1.columns
df_grouped=df1.groupby('TYPE').sum()

print(df_grouped)



df_grouped_1=df_grouped[[ u'JANUARY', u'FEBRUARY', u'MARCH', u'APRIL', u'MAY', u'JUNE',

       u'JULY', u'AUGUST', u'SEPTEMBER', u'OCTOBER', u'NOVEMBER', u'DECEMBER',

       u'TOTAL']]
df_transposed=df_grouped_1.T

df_transposed['total_accident']=df_transposed.sum(axis=1)

df_transposed.sort_values('total_accident')
df_transposed.plot(kind='bar',stacked=True)

df1[['STATE/UT','TOTAL']].groupby('STATE/UT').sum().sort_values(by='TOTAL',ascending=False)[:10]

# IT seems TN is having leading accident toll 
df1[['STATE/UT','TOTAL']].groupby('STATE/UT').sum().sort_values(by='TOTAL',ascending=False).plot(kind='bar')
print(df2[[u'TYPE', u'0-3 hrs. (Night)', u'3-6 hrs. (Night)',

       u'6-9 hrs (Day)', u'9-12 hrs (Day)', u'12-15 hrs (Day)',

       u'15-18 hrs (Day)', u'18-21 hrs (Night)', u'21-24 hrs (Night)']].groupby('TYPE').sum())

df2[[u'TYPE', u'0-3 hrs. (Night)', u'3-6 hrs. (Night)',

       u'6-9 hrs (Day)', u'9-12 hrs (Day)', u'12-15 hrs (Day)',

       u'15-18 hrs (Day)', u'18-21 hrs (Night)', u'21-24 hrs (Night)']].groupby('TYPE').sum().plot(kind='bar')
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(18,14))

df2.groupby(['STATE/UT','YEAR']).sum()[['Total']]

pd.pivot_table(data=df2,index='STATE/UT',columns='YEAR',values='Total',aggfunc='sum').plot(kind='bar',stacked=True)