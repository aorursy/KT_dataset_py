import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')        

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')



train_transaction = train_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD']]

train_identity = train_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]

test_transaction = test_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD']]

test_identity = test_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]



train_transaction = train_transaction.merge(train_identity, how='left', left_on='TransactionID', right_on='TransactionID')

test_transaction = test_transaction.merge(test_identity, how='left', left_on='TransactionID', right_on='TransactionID')



total = train_transaction.copy()

total = total[total.ProductCD=='C']

del train_transaction

del test_transaction
#same functions as in my previous kernel about CardID

import itertools

import math

import networkx as nx



#function to create keys based on multiple columns

def create_key(df, cols, name_new_col):

    '''

    df: pandas dataframe

    cols: list of columns composing the key

    name_new_col: name given to the new column

    '''

    df.loc[:,name_new_col] = ''

    for col in cols:

        df.loc[:,name_new_col] = df.loc[:,name_new_col] + df.loc[:,col].astype(str)

    return df  



def truncate(f, n):

    return math.floor(f * 10 ** n) / 10 ** n  



def merge(list1, list2): 

    merged_list = [[p1, p2] for idx1, p1 in enumerate(list1)  

    for idx2, p2 in enumerate(list2) if idx1 == idx2] 

    return merged_list  



def find_groups(df, groupingcriteria):   

    a=[]

    liste_sameamount = df.groupby(groupingcriteria)['TransactionID'].apply(list).tolist()

    res = [list(map(a.append, map(list,zip(i, i[1:] + i[:1])))) for i in liste_sameamount]

    return a
groups = pd.read_csv('../input/cardid/groups.csv')

groups = groups.set_index('TransactionID')

dictgroups = groups['groups'].to_dict()

total['cardID'] = total['TransactionID'].map(dictgroups)
total['day'] = total['TransactionDT']/(3600*24)

total['daytrunc'] = total['day'].apply(lambda x: truncate(x,1))

total['dayround'] = total['day'].apply(lambda x: round(x,1))

total['TransactionAmtround'] = total['TransactionAmt'].apply(lambda x: round(x,3))
total1 = total[['TransactionID','TransactionAmt','TransactionAmtround','id_19','id_20','daytrunc','dayround','day']].copy()

total1 = create_key(total1, ['TransactionAmtround','id_19','id_20'],'firstgroupcriteriaC')
import gc

timeframe = total1.dayround.unique().tolist()

group_list_C_criteria1 = []



for frame in timeframe:

    if frame%50==0:

        print('day',frame)

        gc.collect()

    

    subset = total1[total1['dayround']==frame].copy()

    if len(subset)==1:

        group_list_C_criteria1.append(subset['TransactionID'].tolist())

    else:

        group_list_C_criteria1.extend(find_groups(subset, 'firstgroupcriteriaC'))
timeframe = total1.daytrunc.unique().tolist()



for frame in timeframe:

    if frame%50==0:

        print('day',frame)

        gc.collect()

    

    subset = total1[total1['daytrunc']==frame].copy()

    if len(subset)==1:

        group_list_C_criteria1.append(subset['TransactionID'].tolist())

    else:

        group_list_C_criteria1.extend(find_groups(subset, 'firstgroupcriteriaC'))
print(len(group_list_C_criteria1))

group_list_C_criteria1 = [list(tupl) for tupl in {tuple(item) for item in group_list_C_criteria1 }]

print(len(group_list_C_criteria1))
group_list_C_criteria1[:10]
L = group_list_C_criteria1

G = nx.Graph()



G.add_nodes_from(sum(L, []))

q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]

for i in q:

          G.add_edges_from(i)



group_list = [list(i) for i in nx.connected_components(G)]



myDict = {}



for i in range(0,len(group_list)):

    for element in group_list[i]:

        name='group'+str(i)

        myDict[element] = name

    

groupsCAmtid1920 = pd.DataFrame.from_dict(myDict, orient='index').reset_index()

groupsCAmtid1920.columns=['TransactionID','groupsCAmtid1920']
groupsCAmtid1920.head(5)
total1 = total[['TransactionID','id_19','id_20','id_31','DeviceInfo','daytrunc','dayround','day']].copy()

total1 = create_key(total1, ['id_19','id_20','id_31','DeviceInfo'],'secondgroupcriteriaC')

total1 = total1[(total1['id_20'].isna()==False) & (total1['id_19'].isna()==False) & (total1['id_31'].isna()==False)]

# this key is too indulgent if we don't get rid of missing id_19 and id_20 as many are missing, but try your experiments to find the best combination
import gc

timeframe = total1.dayround.unique().tolist()

group_list_C_criteria1 = []



for frame in timeframe:

    if frame%50==0:

        print('day',frame)

        gc.collect()

    

    subset = total1[total1['dayround']==frame].copy()

    if len(subset)==1:

        group_list_C_criteria1.append(subset['TransactionID'].tolist())

    else:

        group_list_C_criteria1.extend(find_groups(subset, 'secondgroupcriteriaC'))

    

## Second on Truncated data

timeframe = total1.daytrunc.unique().tolist()



for frame in timeframe:

    if frame%50==0:

        print('day',frame)

        gc.collect()

    

    subset = total1[total1['daytrunc']==frame].copy()

    if len(subset)==1:

        group_list_C_criteria1.append(subset['TransactionID'].tolist())

    else:

        group_list_C_criteria1.extend(find_groups(subset, 'secondgroupcriteriaC'))       
print(len(group_list_C_criteria1))

group_list_C_criteria1 = [list(tupl) for tupl in {tuple(item) for item in group_list_C_criteria1 }]

print(len(group_list_C_criteria1)) 
L = group_list_C_criteria1

G = nx.Graph()



G.add_nodes_from(sum(L, []))

q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]

for i in q:

    G.add_edges_from(i)



group_list = [list(i) for i in nx.connected_components(G)]



groupsCid192031Device = pd.DataFrame.from_dict(myDict, orient='index').reset_index()

groupsCid192031Device.columns=['TransactionID','groupsCid192031Device']
total1 = total[['TransactionID','day','cardID']].copy()



groupC1 = groupsCAmtid1920.copy()

groupC2 = groupsCid192031Device.copy()



total1 = total1.merge(groupC1, how='left',left_on='TransactionID',right_on='TransactionID')

total1 = total1.merge(groupC2, how='left',left_on='TransactionID',right_on='TransactionID')



#imputation

total1['imputecol'] = [i for i in range(0,len(total1))]

total1.loc[total1.groupsCid192031Device.isna(), 'groupsCid192031Device'] = total1.loc[total1.groupsCid192031Device.isna(), 'imputecol']
groups_C_final = []

groups_C_final.extend(find_groups(total1, 'cardID'))

print('group1done')



groups_C_final.extend(find_groups(total1, 'groupsCAmtid1920'))

print('group2done')



groups_C_final.extend(find_groups(total1, 'groupsCid192031Device'))

print('group3done')



print(len(groups_C_final))

groups_C_final = [list(tupl) for tupl in {tuple(item) for item in groups_C_final }]

print(len(groups_C_final))
L = groups_C_final

G = nx.Graph()



G.add_nodes_from(sum(L, []))

q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]

for i in q:

    G.add_edges_from(i)



group_list = [list(i) for i in nx.connected_components(G)]



myDict = {}



for i in range(0,len(group_list)):

    for element in group_list[i]:

        name='group'+str(i)

        myDict[element] = name

    

groupsCuser = pd.DataFrame.from_dict(myDict, orient='index').reset_index()

groupsCuser.columns=['TransactionID','groupsCuser']

groupsCuser.to_csv('groupsCuser.csv',index=False)
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')



train_transaction = train_transaction[['TransactionID','TransactionDT','TransactionAmt','ProductCD','isFraud']]

train_identity = train_identity[['TransactionID','id_19','id_20','id_31','DeviceInfo']]



total = train_transaction.merge(train_identity, how='left', left_on='TransactionID', right_on='TransactionID')



total = total[total.ProductCD=='C']

del train_transaction
#cardID

groups = pd.read_csv('../input/cardid/groups.csv')

groups = groups.set_index('TransactionID')

dictgroups = groups['groups'].to_dict()

total['cardID'] = total['TransactionID'].map(dictgroups)



#New User group

total = total.merge(groupsCuser, how='left',left_on='TransactionID',right_on='TransactionID')
total['CardIDcount'] = total['cardID'].map(total.cardID.value_counts())

total['CardID_fraud_sum'] = total.groupby('cardID')['isFraud'].sum()



total['UserIDcount'] = total['groupsCuser'].map(total.groupsCuser.value_counts())

total['UserID_fraud_sum'] = total.groupby('groupsCuser')['isFraud'].sum()
total[total.isFraud==1].to_csv('checkgroups.csv',index=False)