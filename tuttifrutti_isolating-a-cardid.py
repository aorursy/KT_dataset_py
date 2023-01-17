import pandas as pd

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')        

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = train_transaction.merge(train_identity, how='left', left_on='TransactionID',right_on='TransactionID')

del train_identity
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



train_transaction['day'] = train_transaction['TransactionDT']/(3600*24)

train_transaction['D1minusday'] = (train_transaction['D1']-train_transaction['day']).replace(np.nan, -9999).map(int)

colsID = ['card1','card2','card3','card4','card5','card6','D1minusday','ProductCD']

train_transaction = create_key(train_transaction, colsID, 'cardID_D1')
train_transaction['cardID_D1'].value_counts()
len(train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=="C")])
train_transaction['V307'] = train_transaction['V307'].fillna(0)

train_transaction['V307plus'] = train_transaction['V307']+train_transaction['TransactionAmt']
train_transaction.loc[(train_transaction.TransactionID==3030465),'V307'].values[0] == train_transaction.loc[(train_transaction.TransactionID==3026025),'V307plus'].values[0]
train_transaction['V307rtrunc'] = train_transaction['V307'].apply(lambda x: truncate(x,3))

train_transaction['V307round'] = train_transaction['V307'].apply(lambda x: round(x,3))

train_transaction['V307plusround'] = train_transaction['V307plus'].apply(lambda x: round(x,4))

train_transaction['V307plusroundtrunc'] = train_transaction['V307plusround'].apply(lambda x: truncate(x,3))

train_transaction['V307plusround'] = train_transaction['V307plus'].apply(lambda x: round(x,3))

train_transaction['V307trunc2'] = train_transaction['V307'].apply(lambda x: truncate(x,2))

train_transaction['V307plustrunc2'] = train_transaction['V307plus'].apply(lambda x: truncate(x,2))

train_transaction['TransactionAmttrunq'] = train_transaction['TransactionAmt'].apply(lambda x: round(x,3))
#the card group of interest for this example

card_group = train_transaction[train_transaction.cardID_D1=='16136204.0185.0visa138.0debit108C']



list1 = card_group['V307plusroundtrunc'].tolist()

list2 = card_group['V307rtrunc'].tolist()

kv = []

res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

list1 = card_group.iloc[[i[0] for i in res]]['TransactionID'].tolist()

list2 = card_group.iloc[[i[1] for i in res]]['TransactionID'].tolist()

liste_existstrun = merge(list1, list2)
liste_existstrun
L = liste_existstrun

G = nx.Graph()



G.add_nodes_from(sum(L, []))

q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]

for i in q:

    G.add_edges_from(i)

group_list = [list(i) for i in nx.connected_components(G)]

group_list
def find_groups(aa):

    group_list = []

    

    #get the couples by existstrun

    list1 = aa['V307plusroundtrunc'].tolist()

    list2 = aa['V307rtrunc'].tolist()

    kv = []

    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()

    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()

    liste_existstrun = merge(list1, list2)





    #get the couples by existsroundtrunc

    list1 = aa['V307plusroundtrunc'].tolist()

    list2 = aa['V307round'].tolist()

    kv = []

    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()

    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()

    liste_existsroundtrunc = merge(list1, list2)



    #get the couples by existsroundtrunc

    list1 = aa['V307plusround'].tolist()

    list2 = aa['V307round'].tolist()

    kv = []

    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()

    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()

    liste_existsroundround = merge(list1, list2)





    #get the couples by existsroundtrunc

    list1 = aa['V307trunc2'].tolist()

    list2 = aa['V307plustrunc2'].tolist()

    kv = []

    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()

    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()

    liste_existstrunc2 = merge(list1, list2)





    #get the couples by existsamount

    list1 = aa['TransactionAmttrunq'].tolist()

    list2 = aa['V307round'].tolist()

    kv = []

    res = [[list(filter(lambda z: list1[z]==x, range(len(list1)))),list(filter(lambda z: list2[z]==x, range(len(list2))))] for x in list1 if x in list2] #find the pairs

    res= [list(map(kv.append,map(list,(itertools.product(*sublist))))) for sublist in res] #drop duplicates from list of list

    res = list(map(list, set(map(lambda i: tuple(i), kv)))) #create list of couple indexes

    list1 = aa.iloc[[i[0] for i in res]]['TransactionID'].tolist()

    list2 = aa.iloc[[i[1] for i in res]]['TransactionID'].tolist()

    liste_existsamount = merge(list1, list2)



    #get by exact same amount

    a=[]

    liste_sameamount = aa.groupby('TransactionAmt')['TransactionID'].apply(list).tolist()

    res = [list(map(a.append, map(list,zip(i, i[1:] + i[:1])))) for i in liste_sameamount]



    group_list.extend(liste_existstrun)

    group_list.extend(liste_existsroundtrunc)

    group_list.extend(liste_existsamount)

    group_list.extend(liste_existsroundround)

    group_list.extend(liste_existstrunc2)



    group_list.extend(a)



    L = group_list

    G = nx.Graph()

    G.add_nodes_from(sum(L, []))

    q = [[(s[i],s[i+1]) for i in range(len(s)-1)] for s in L]

    for i in q:

        G.add_edges_from(i)

    group_list = [list(i) for i in nx.connected_components(G)]

    return group_list
groups_found = find_groups(card_group)

groups_found