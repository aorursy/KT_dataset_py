# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/basket-optimisation/Market_Basket_Optimisation.csv',header=None)

print(data.shape)
data.head()
#one_hot_encoder transform data



data['ColumnA'] = data[data.columns[0:]].apply(

    lambda x: ','.join(x.dropna().astype(str)),

    axis=1

)

#data.head()



df = data.ColumnA

df = pd.DataFrame(data=df.values, columns=['ColumnA'])

print(df.shape)

print(df)



data_hot_encoded = df.drop('ColumnA',1).join(df.ColumnA.str.get_dummies(','))

data_hot_encoded.shape
data_hot_encoded.head()



itemsets = apriori(data_hot_encoded,use_colnames=True, min_support=0.05)

itemsets = itemsets.sort_values(by="support" , ascending=False) 



frequent_itemsets = apriori(data_hot_encoded, min_support=0.05, use_colnames=True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets


rules =  association_rules(itemsets, metric='lift', min_threshold=1)



rules = rules.sort_values(by="lift" , ascending=False) 

#rules.to_csv('./rules.csv')

print('-'*20, 'association rules', '-'*20)

print(rules)


