import pandas as pd

import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv("../input/supermarket/GroceryStoreDataSet.csv",names=['products'],header=None)
df.head()
df.shape
data = list(df["products"].apply(lambda x:x.split(',')))
data

from mlxtend.preprocessing import TransactionEncoder
#Data must be structured as 1-0 or True-False

te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df
df1 = apriori(df,min_support=0.01,use_colnames=True)

df1
df1.sort_values(by="support",ascending=False)
association_rules(df1, metric = "confidence", min_threshold = 0.6).head(10)