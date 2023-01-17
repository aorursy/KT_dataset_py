import numpy as np

import pandas as pd

from IPython.display import  display

from collections import defaultdict

from itertools import combinations

pd.set_option('display.max_colwidth',-1)
train_df=pd.read_csv('../input/train.csv')

train_df.head(2)
# only duplicated questions

ddf=train_df[train_df.is_duplicate==1]

print('Duplicated questions shape:',ddf.shape)

ddf.head(2)
# get all duplicated questions

clean_ddf1=ddf[['qid1','question1']].drop_duplicates()

clean_ddf1.columns=['qid','question']

clean_ddf2=ddf[['qid2','question2']].drop_duplicates()

clean_ddf2.columns=['qid','question']

all_dqdf=clean_ddf1.append(clean_ddf2,ignore_index=True)

print(all_dqdf.shape)

all_dqdf.head(2)
# groupby qid1, and then we get all the combinations of id in each group

dqids12=ddf[['qid1','qid2']]

df12list=dqids12.groupby('qid1', as_index=False)['qid2'].agg({'dlist':(lambda x: list(x))})

print(len(df12list))

d12list=df12list.values

d12list=[[i]+j for i,j in d12list]

# get all the combinations of id, like (id1,id2)...

d12ids=set()

for ids in d12list:

    ids_len=len(ids)

    for i in range(ids_len):

        for j in range(i+1,ids_len):

            d12ids.add((ids[i],ids[j]))

print(len(d12ids))
# the same operation of qid2

dqids21=ddf[['qid2','qid1']]

display(dqids21.head(2))

df21list=dqids21.groupby('qid2', as_index=False)['qid1'].agg({'dlist':(lambda x: list(x))})

print(len(df21list))

ids2=df21list.qid2.values

d21list=df21list.values

d21list=[[i]+j for i,j in d21list]

d21ids=set()

for ids in d21list:

    ids_len=len(ids)

    for i in range(ids_len):

        for j in range(i+1,ids_len):

            d21ids.add((ids[i],ids[j]))

len(d21ids)
# merge two set

dids=list(d12ids | d21ids)

len(dids)
# let's define union-find function

def indices_dict(lis):

    d = defaultdict(list)

    for i,(a,b) in enumerate(lis):

        d[a].append(i)

        d[b].append(i)

    return d



def disjoint_indices(lis):

    d = indices_dict(lis)

    sets = []

    while len(d):

        que = set(d.popitem()[1])

        ind = set()

        while len(que):

            ind |= que 

            que = set([y for i in que 

                         for x in lis[i] 

                         for y in d.pop(x, [])]) - ind

        sets += [ind]

    return sets



def disjoint_sets(lis):

    return [set([x for i in s for x in lis[i]]) for s in disjoint_indices(lis)]
# split data into groups, so that each question in each group are duplicated

did_u=disjoint_sets(dids)

new_dids=[]

for u in did_u:

    new_dids.extend(list(combinations(u,2)))

len(new_dids)
new_ddf=pd.DataFrame(new_dids,columns=['qid1','qid2'])

print('New duplicated shape:',new_ddf.shape)

display(new_ddf.head(2))
# merge with all_dqdf to get question1 description

new_ddf=new_ddf.merge(all_dqdf,left_on='qid1',right_on='qid',how='left')

new_ddf.drop('qid',inplace=True,axis=1)

new_ddf.columns=['qid1','qid2','question1']

new_ddf.drop_duplicates(inplace=True)

print(new_ddf.shape)

new_ddf.head(2)
# the same operation with qid2

new_ddf=new_ddf.merge(all_dqdf,left_on='qid2',right_on='qid',how='left')

new_ddf.drop('qid',inplace=True,axis=1)

new_ddf.columns=['qid1','qid2','question1','question2']

new_ddf.drop_duplicates(inplace=True)

print(new_ddf.shape)

new_ddf.head(2)
# is_duplicate flag

new_ddf['is_duplicate']=1

new_ddf.head(2)
# let random select 10 rows to check the result

new_ddf.sample(10)
# the orininal duplicated pairs count:

print(len(all_dqdf))

# after we generate more data, then the duplicated pairs count:

print(len(new_ddf))
new_ddf[new_ddf.qid1 == 290521]
new_ddf[new_ddf.qid2== 290521]
new_ddf
del train_df['id']

new = train_df.append(new_ddf)

new = new.drop_duplicates()
a = sum(new['is_duplicate'] == 1) * 0.31931551750964987

b = sum(new['is_duplicate'] == 1) * 0.31931551750964987  + sum(new['is_duplicate'] == 0) * (2-0.31931551750964987)

a / b
(0.17426) / (sum(new['is_duplicate'] == 1) / len(new))
2 * 0.17426 * 255027 / (283274 - 0.17426 * 283274 +  0.17426 * 255027)
2-0.3193155
a = 377811 * 0.24937809 *2

b = 255027 * 1.750621*2
a/(a+b)
sum(new['is_duplicate'] == 1), sum(new['is_duplicate'] == 0)
import csv

new.to_csv('new.csv', quoting = csv.QUOTE_ALL)
new