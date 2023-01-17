# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Read input

df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

print (df.shape)

df[:5]
## Create Cost Dataframe that calculates the cost for every family's options

cost = df[['family_id']]

cost['choice_0'] = 0

cost['choice_1'] = 50

cost['choice_2'] = 50 + df.n_people*9

cost['choice_3'] = 100 + df.n_people*9

cost['choice_4'] = 200 + df.n_people*9

cost['choice_5'] = 200 + df.n_people*18

cost['choice_6'] = 300 + df.n_people*18

cost['choice_7'] = 300 + df.n_people*36

cost['choice_8'] = 400 + df.n_people*36

cost['choice_9'] = 500 + df.n_people*(36+199)

cost['otherwise'] = 500 + df.n_people*(36+398)



df['otherwise']= 101 # create this column to match the column list in cost df



cost[:5]
# Create Day Cost dataframe that calculates the cost if the family is assigned any day. if it cannot be assigned then use inf.



day_cost = np.zeros((5000,101))

day_cost[:,:] = np.inf

for fid,days,costs in zip(df.family_id,df[cost.columns[1:]].values, cost[cost.columns[1:]].values):

    for d,c in zip(days, costs):

        day_cost[fid,d-1]=c

day_cost = pd.DataFrame(day_cost, columns = ['day_{}'.format(i) for i in range(1,102)])

day_cost['n_people'] = df.n_people

day_cost[:5]
total_cost = 0

assigned = np.array([-1 for _ in range(5001)])  # default to -1 if not assigned

assigned_ppl = np.zeros(101)  # To update the number of people assigned everyday

while(len(day_cost)>0):

    print ('\n', '*'*100)

    print ('\nNum families remaining :', len(day_cost))

    for d in range(1,101):

        temp_df = day_cost[np.isfinite(day_cost['day_{}'.format(d)])].sort_values('day_{}'.format(d))

        cumsum = temp_df.n_people.cumsum()

        last_idx = np.where(cumsum>40)[0] # in every batch/loop upto 65 people get assigned to a day. You can play with this number and improve score

        if len(last_idx)>1:

            last_idx = last_idx[1]

        else:

            last_idx = len(cumsum)

        idx_fam = list(cumsum[:last_idx].index)

        assigned[idx_fam] = d

        c=temp_df.loc[idx_fam]['day_{}'.format(d)].sum()

        print('Day {} cost : {}'.format(d,c), end='\t\t')

        total_cost+=c

        assigned_ppl[d] = assigned_ppl[d]+day_cost.n_people.loc[idx_fam].sum()

        day_cost = day_cost.drop(idx_fam)

print ('Total Cost : ', total_cost)

day_cost[:5]
assigned_ppl[1:]

pd.DataFrame({'family_id':range(5000),'assigned_day':assigned[:5000]}).to_csv('submission.csv',index=False)