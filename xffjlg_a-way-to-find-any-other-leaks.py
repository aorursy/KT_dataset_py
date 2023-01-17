import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
# Horizontal search, to find more columns
# reference:
# https://www.kaggle.com/johnfarrell/giba-s-property-extended-extended-result
def bf_search_a(df_new, df_cand):
    cnt = 0
    head_curr = df_new.values[1:, 0]
    tail_curr = df_new.values[:-1, -1]
    while True:
        for c in df_cand.columns:
            if c in df_new:
                continue
            elif np.all(
                df_cand[c].iloc[:-1].values==head_curr
            ) and len(df_cand[c].unique())>1:
                df_new.insert(0, c, df_cand[c].values)
                head_curr = df_new.values[1:, 0]
#                 print(c, 'found head!', 'new shape', df_new.shape)
                cnt += 1
                break
            elif np.all(
                df_cand[c].iloc[1:].values==tail_curr
            ) and len(df_cand[c].unique())>1:
                df_new[c] = df_cand[c].values
                tail_curr = df_new.values[:-1, -1]
#                 print(c, 'found tail!', 'new shape', df_new.shape)
                cnt += 1
                break
            else:
                continue
        if cnt==0:
            break
        else:
            cnt = 0
            continue
    return df_new
# vertically Search,to find more rows
def df_append(df, index, train, target='target', col='col', index_h={}):
    index_h.add(index)
    df = df.append(train[[target,col]].loc[index])
    last_value = df[target].tail(1).values[0]
    col_index = list(train[train[col]==last_value].index)
    uni_col_index = list(set(col_index) - index_h)

    if uni_col_index:
        if df.shape[0] < 2:
            df = df_append(df, min(uni_col_index), train, target, col, index_h)  
        for col_index_i in uni_col_index:
            uni_col_index.remove(col_index_i)
            df_c = df.copy()
            bf_search_a(df_c, train.iloc[df_c.index, 2:])
            if df_c.shape[1] >= 4:
                print(df_c.index, df_c.shape)
                df = df_append(df, col_index_i, train, target, col, index_h)
            else:
                df = df.drop(df.index[-1])
                df = df.append(train[[target,col]].loc[col_index_i])
    return df
## run to search
# for col in [f for f in train.columns if f not in['ID','target']]:  # use this to search in all columns
for col in [f for f in train.columns if f not in['ID','target']][1:3]:
    print('#################### new col:     ',col)
#     for i in range(train.shape[0]):  # use this to search in all rows
    for i in range(50):
        ff = pd.DataFrame()
        index_h = set()
        df_append(ff, i, train, 'target', col, index_h)