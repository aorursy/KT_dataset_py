# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import seaborn as sns
from sklearn import preprocessing
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.style.use('ggplot')
%matplotlib inline
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output

input_df = pd.read_csv("../input/arrests.csv",sep=',')

# Any results you write to the current directory are saved as output.
old_columns = input_df.columns

US_immigrants =  input_df.loc[input_df['Border'] == 'United States']

year = [i for i in range(2000,2017)]
all_immigrants_column =  [cname for cname in old_columns if '(All Illegal Immigrants)' in cname ]
mex_immigrants_column = [cname for cname in old_columns if '(Mexicans Only)' in cname ]

all_immigrants = US_immigrants[all_immigrants_column].values.tolist()
mex_immigrants = US_immigrants[mex_immigrants_column].values.tolist()
other_immigrants = C = [a - b for a, b in zip(all_immigrants[0], mex_immigrants[0])]

df = [year,all_immigrants,mex_immigrants]


us_immigrants_df = pd.DataFrame({"year":year,
                                "total_immigrants":all_immigrants[0],
                                "total_mexicans":mex_immigrants[0],
                                "other":other_immigrants})

print(us_immigrants_df)
sns.factorplot('year','total_immigrants',kind='bar',data=us_immigrants_df,size=4, aspect=2)
us_immigrants_df.plot(x='year',y=['total_mexicans','other'],kind='bar', stacked=True,alpha=0.5,figsize=(9,5))
fig, ax = plt.subplots(figsize=(10,5))
old_columns = input_df.columns
required_columns = [cname for cname in old_columns if '(Mexicans Only)' not in cname]

US_immigrants =  input_df[required_columns].loc[input_df['Sector'] == 'All']

US_immigrants = US_immigrants.drop(['State/Territory','Sector'],1)

new_cname = [ cname.replace(' (All Illegal Immigrants)','') for cname in  US_immigrants.columns]

US_immigrants.columns = new_cname

normalize_column = [cname for cname in new_cname if 'Border' not in cname]

for cname in normalize_column:
    US_immigrants[cname]  = US_immigrants[cname].apply(lambda x: math.log(x))

US_immigrants = US_immigrants.set_index('Border')
US_immigrants.columns.names = ['year']

sns.heatmap(US_immigrants, linewidths=.5,annot=True)
old_columns = input_df.columns

required_columns = [cname for cname in old_columns if '(Mexicans Only)' not in cname]

US_immigrants =  input_df[required_columns].loc[(input_df['Sector'] != 'All') & (input_df['Sector'] != 'United States')]

US_immigrants = US_immigrants.drop(['State/Territory','Border'],1)

new_cname = [ cname.replace(' (All Illegal Immigrants)','') for cname in  US_immigrants.columns]

US_immigrants.columns = new_cname

normalize_column = [cname for cname in new_cname if 'Sector' not in cname]

for cname in normalize_column:
    US_immigrants[cname]  = US_immigrants[cname].apply(lambda x: math.log(x))

US_immigrants = US_immigrants.set_index('Sector')
US_immigrants.columns.names = ['year']

sns.heatmap(US_immigrants, linewidths=.5)