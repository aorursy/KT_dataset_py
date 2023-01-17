import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

act_df = pd.read_csv('../input/act_train.csv',sep=',')



sns.countplot(x='outcome',data=act_df)

sns.plt.show()
sns.countplot(x='activity_category',data=act_df,hue='outcome')

sns.plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(30, 20)

h = sns.countplot(x='char_1',data=act_df,hue='outcome',ax=ax)

h.set_xticklabels(h.get_xticklabels(),rotation=50)

sns.plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(30, 20)

people_df = pd.read_csv('../input/people.csv',sep=',')

group_based_ppl_count = people_df.groupby(['group_1']).count().sort_values(by='people_id',ascending=[0])

group_based_ppl_count = group_based_ppl_count.reset_index()

group_based_ppl_count = group_based_ppl_count.ix[:20,]

g = sns.barplot(x='group_1',y='people_id',data=group_based_ppl_count,ax=ax)



#merging people and activity 



people_df = pd.read_csv("../input/people.csv",sep=',',parse_dates=['date'])

activity_df = pd.read_csv("../input/act_train.csv",sep=',',parse_dates=['date'])



def sanitizepeople():

    sn_fileds= ["char_1","group_1","char_2","date","char_3","char_4","char_5","char_6","char_7","char_8","char_9"]

    for filed  in sn_fileds:



        if "group" in filed:

            people_df[filed] = people_df[filed].str.lstrip('group ').astype(np.float)

        elif "char_" in filed:

            people_df[filed] = people_df[filed].fillna("-999")

            people_df[filed] = people_df[filed].str.lstrip('type ').astype(np.float)

        else:

            people_df['year'] = people_df[filed].dt.year

            people_df['month'] = people_df[filed].dt.month

            people_df['day'] = people_df[filed].dt.day



    people_df1 = people_df.drop(['date'],axis=1)



    return people_df1



def sanitizeactivity():

    sn_fileds= ["date","activity_category","char_1","char_2","char_3","char_4","char_5","char_6","char_7","char_8","char_9","char_10"]

    for filed  in sn_fileds:



        if "char_" in filed or "activity" in filed:

            activity_df[filed] = activity_df[filed].fillna("-999")

            activity_df[filed] = activity_df[filed].str.lstrip('type ').astype(np.float)

        else:

            activity_df['year'] = activity_df[filed].dt.year

            activity_df['month'] = activity_df[filed].dt.month

            activity_df['day'] = activity_df[filed].dt.day



    activity_df1 = activity_df.drop(['date'],axis=1)



    return activity_df1





people_nrm_df = sanitizepeople()

activity_nrm_df = sanitizeactivity()

j_df = pd.merge(people_nrm_df,activity_nrm_df,how='left',on='people_id',left_index='True')





fig, ax = plt.subplots()

fig.set_size_inches(30, 20)



j_top20grp_grpby = j_df.groupby(['group_1']).sum().sort_values(by='outcome',ascending=[0])

j_top20grp_grpby = j_top20grp_grpby.reset_index()



top20group = j_top20grp_grpby['group_1'].astype(np.int).tolist()

top20group = top20group[:50]



j_top20grp_df = j_df.loc[j_df['group_1'].isin(top20group)]



j_top20grp_df = j_top20grp_df[['group_1','outcome']]



h = sns.countplot(x='group_1',data=j_top20grp_df,hue='outcome',ax = ax)

h.set_xticklabels(h.get_xticklabels(),rotation=50)

sns.plt.show()