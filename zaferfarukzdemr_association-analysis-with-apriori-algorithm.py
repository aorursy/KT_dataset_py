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
data =pd.read_csv("../input/supermarket/GroceryStoreDataSet.csv",header=None,names=['products']) #creating dataframe
print(data)
df = data.copy()
df.head(7) #first seven row 
df.values #shows the values ​​of each row
df.shape #shape of dataframe
df.columns #cloumns names
data = list(df["products"].apply(lambda x:x.split(','))) #Through this cell products are seperated
data
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

#all its products are converted into a column

#data type is boolean
df
te_data
from mlxtend.frequent_patterns import apriori
df_new = apriori(df,min_support=0.2,use_colnames=True,verbose=1)
df_new
df_new.sort_values(by="support", ascending = False)

df_new['length'] = df_new['itemsets'].apply( lambda x:len(x))
df_new
from mlxtend.frequent_patterns import association_rules
association_rules(df_new, metric ="confidence", min_threshold=0.2)

#if confidence values>2

association_rules(df_new, metric="support", min_threshold=0.2)

##if support values>2
association_rules(df_new, metric ="lift", min_threshold=1.0)

#if lift values>2