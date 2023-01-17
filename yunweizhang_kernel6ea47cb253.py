# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_bankacct=pd.read_csv("/kaggle/input/ungrd-rd2-auo/bank_accounts.csv", dtype={'userid':str, 'bank_account':str})
df_creditcards=pd.read_csv("/kaggle/input/ungrd-rd2-auo/credit_cards.csv", dtype={'userid':str, 'credit_card':str})
df_devices=pd.read_csv("/kaggle/input/ungrd-rd2-auo/devices.csv", dtype={'userid':str, 'device':str})
df_orders=pd.read_csv("/kaggle/input/ungrd-rd2-auo/orders.csv", dtype={'orderid':str, 'buyer_userid':str, 'seller_userid':str})
print(df_bankacct.head())
print(df_creditcards.head())
print(df_devices.head())
print(df_orders.head())
df_bankacct2=df_bankacct
df_creditcards2=df_creditcards
df_devices2=df_devices
df_orders2=df_orders


df_linked_user_bank_acct=pd.merge(df_bankacct, df_bankacct2, on='bank_account')
filt=df_linked_user_bank_acct.userid_x==df_linked_user_bank_acct.userid_y
df_linked_user_bank_acct=df_linked_user_bank_acct[~filt].drop(columns={'bank_account'})


df_linked_user_creditcards=pd.merge(df_creditcards, df_creditcards2, on='credit_card')
filt=df_linked_user_creditcards.userid_x==df_linked_user_creditcards.userid_y
df_linked_user_creditcards=df_linked_user_creditcards[~filt].drop(columns={'credit_card'})

df_linked_user_devices=pd.merge(df_devices, df_devices, on='device')
filt=df_linked_user_devices.userid_x==df_linked_user_devices.userid_y
df_linked_user_devices=df_linked_user_devices[~filt].drop(columns={'device'})

df_direct=pd.concat([df_linked_user_bank_acct,df_linked_user_creditcards, df_linked_user_devices]).drop_duplicates().dropna()


df_1=df_direct
#df_1['Not Done']=True
df_2=df_direct
len1=0
len2=1
while len1!=len2:
    len1=len(df_1)
    df_linked_user=pd.merge(df_1,df_2, how='left', left_on='userid_y', right_on='userid_y').drop_duplicates()
    df_linked_user.columns=['userid_x','userid_y','userid_z']
    df_linked_user=pd.melt(df_linked_user, id_vars='userid_x', value_name='userid_y').drop_duplicates()
    print(df_linked_user.tail())

    df_linked_user=df_linked_user[['userid_x','userid_y']]
    df_linked_user=df_linked_user[df_linked_user.userid_x!=df_linked_user.userid_y]
    df_linked_user=df_linked_user.drop_duplicates().dropna()
    print(df_linked_user.tail())

    df_1=df_linked_user
    len2=len(df_linked_user)

# df_1=df_direct.copy()
# df_1['Not_Done']=True
# df_2=df_direct.copy()
# df_linked_user= pd.DataFrame(columns=['userid_x','userid_y','Not_Done'])

# while not(df_1.empty):
#     len1=len(df_1)
#     print(len(df_1))
#     df_combined=pd.merge(df_1,df_2, how='left', left_on='userid_y', right_on='userid_y').drop_duplicates()
#     df_combined.columns=['userid_x','userid_y','Not_Done','userid_z']
#     df_combined=pd.melt(df_combined, id_vars=['userid_x','Not_Done'], value_name='userid_y').drop_duplicates()
#     df_combined['Not_Done']=(df_combined['variable']=='userid_z')
#     df_combined=df_combined[['userid_x','userid_y','Not_Done']]
#     df_combined=df_combined[df_combined.userid_x!=df_combined.userid_y]
#     df_linked_user= pd.concat([df_linked_user, df_combined])
#     prioritize_not_done_false= df_linked_user.groupby(['userid_x','userid_y']).Not_Done.transform(min)
#     df_linked_user = df_linked_user.loc[df_linked_user.Not_Done == prioritize_not_done_false]
#     df_linked_user=df_linked_user.drop_duplicates().dropna()
#     print(df_linked_user)
#     df_1=df_linked_user[df_linked_user.Not_Done==True]
    



# df_link_index= dict(list(df_direct.groupby('userid_x')))

# for x in df_link_index:
#     df_link_index[x]=list(df_link_index[x].userid_y)

    
# counter=0

# for x in df_link_index:
#     y_list=df_link_index[x].copy()

#     while len(y_list)!=0:
#         y=y_list.pop()
#         if y in df_link_index:
#             for z in df_link_index[y]:
#                 if ((z not in df_link_index[x]) & (z!=x)):
#                     df_link_index[x].append(z)
#                     y_list.append(z)
#     counter=counter+1
#     if counter in [10000,20000,30000,40000, 50000]:
#         print (counter)
        
        
#     while len(id_y_list)!=0:
#         id_y=id_y_list.pop()
#         filt1=df_linked_user.get_group(id_y).userid_y.isin(df['userid_y'])
#         filt2=df_linked_user.get_group(id_y).userid_y==id_x
#         filt=filt1|filt2
#         new_df=df_linked_user.get_group(id_y)[~filt]
#         if len(new_df)!=0:
#             new_df['userid_x']=id_x
#             df=pd.concat([df, new_df])
#             id_y_list=id_y_list+list(new_df['userid_y'])     
#     return df
    

df_result = pd.merge(df_orders, df_linked_user,  how='left', left_on=['buyer_userid','seller_userid'], right_on = ['userid_x','userid_y'])



df_result['is_fraud']=(~df_result.userid_x.isna())
print(df_result.tail(10))
df_result=df_result[['orderid','is_fraud']]
df_result['is_fraud']=df_result['is_fraud'].astype(int)
df_result.to_csv('ungrad_r2.csv',index=False)
df_linked_user.to_csv('linked_user.csv',index=False)