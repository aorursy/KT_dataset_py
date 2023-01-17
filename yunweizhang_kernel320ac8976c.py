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
df_keyword_list=pd.read_csv("/kaggle/input/undrg-rd1-listings/Extra Material 2 - keyword list_with substring.csv")
df_keyword_spam=pd.read_csv("/kaggle/input/undrg-rd1-listings/Keyword_spam_question.csv")

print(df_keyword_list.head())
print(df_keyword_spam.head())
print(df_keyword_list.shape)
print(df_keyword_spam.shape)
#df_keyword_list['Keywords']=df_keyword_list['Keywords'].str.split(',')
#df_keyword_list.Keywords=df_keyword_list.Keywords.apply(pd.Series)
#df_keyword_list['Keywords']


df_keyword_split=df_keyword_list['Keywords'].str.split(', ', expand=True)
df_keyword_list=pd.concat([df_keyword_list, df_keyword_split], axis=1)
df_keyword_list.drop(columns =["Keywords"], inplace = True) 
df_keyword_list=pd.melt(df_keyword_list, id_vars=['Group'], value_name='Keyword')
df_keyword_list=df_keyword_list.dropna()
df_keyword_list

#df_keyword_list=df_keyword_list.loc[df_keyword_list.Keyword.notnull()]
df_keyword_list=df_keyword_list.sort_values(by='Group', ascending=True).reset_index()
df_keyword_dict=dict(list(df_keyword_list.groupby('Keyword')))
for key in df_keyword_dict:
    df_keyword_dict[key]=df_keyword_dict[key].nsmallest(1,'Group').iloc[0,1]
#print(df_keyword_dict)
#df_keyword_dict['notebook'].iloc[0,1]
def find_groups(x):
    found={}
    for keyword in df_keyword_dict.keys():
        if keyword in x.lower():
            temp=found.copy()
            if keyword not in temp and not any(keyword in s for s in temp):
                    for s in temp:
                        if s in keyword:
                            del found[s]
                    found[keyword]=df_keyword_dict[keyword]
    groups_found=sorted(list(found.values()))
    groups_found = list(dict.fromkeys(groups_found))
    return groups_found

   
df_keyword_spam['groups_found']=df_keyword_spam.name.apply(find_groups)
df_keyword_spam
df_keyword_spam.drop(columns=['name'],inplace=True)
df_keyword_spam.to_csv("undergrad_r1_submission_yw.csv", index=False)
df_keyword_spam