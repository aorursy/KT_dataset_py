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
!pip install mlxtend
df = pd.read_csv("../input/supermarket/GroceryStoreDataSet.csv")




df.head()
df.info()
df.shape
df.columns.size
# unique column cıkartma operasyonu

unique_row_items = []

for index, row in df.iterrows():

    items_series = list(row.str.split(','))

    for item_serie in items_series:

        for item in item_serie:

            if item not in unique_row_items:

                unique_row_items.append(item)

    





unique_row_items
df_apriori = pd.DataFrame(columns=unique_row_items)

#df_apriori.at[0,'JAM'] = 1

#df_apriori.at[1,'JAM'] = 1

df_apriori
#eldeki data'ları eşleştirip onehotencoding'e dönüştür ve dataframe'e ekle

for index, row in df.iterrows():

    items = str(row[0]).split(',')

    #print(items)

    one_hot_encoding = np.zeros(len(unique_row_items),dtype=int)

    for it in items:

        for i,column in enumerate(df_apriori.columns):

            #print(i,column,it)

            if it == column:

                one_hot_encoding[i] = 1

    df_apriori.at[index] = one_hot_encoding

    #print(one_hot_encoding)

df_apriori
df_apriori.info()
df_apriori=df_apriori.astype('int')
from mlxtend.frequent_patterns import apriori, association_rules
freq_items = apriori(df_apriori, min_support = 0.2, use_colnames = True, verbose = 1)
freq_items
freq_items.head()
df_association_rules = association_rules(freq_items, metric = "confidence", min_threshold = 0.2)

df_association_rules
df_association_rules.sort_values("confidence",ascending=False)
df_association_rules
df_association_rules["antecedents"].apply(lambda x: str(x))
cols = ['antecedents','consequents']

df_association_rules[cols] = df_association_rules[cols].applymap(lambda x: tuple(x))

print (df_association_rules)
df_association_rules = (df_association_rules.explode('antecedents')

         .reset_index(drop=True)

         .explode('consequents')

         .reset_index(drop=True))
df_association_rules
df_association_rules["product_group"] = df_association_rules["antecedents"].apply(lambda x: str(x)) + "," + df_association_rules["consequents"].apply(lambda x: str(x))
df_association_rules
df1 = df_association_rules.loc[:,["product_group","confidence","lift"]].sort_values("confidence",ascending=False)
import seaborn as sns

sns.set(font_scale=0.4) 

sns.set(rc={'figure.figsize':(21.7,5.27)})

sns.barplot(x="product_group",y="confidence",data=df1);
import seaborn as sns

sns.set(font_scale=0.4) 

sns.set(rc={'figure.figsize':(21.7,5.27)})

sns.barplot(x="product_group",y="confidence",hue="lift",data=df1);
import seaborn as sns

sns.set(font_scale=0.4) 

sns.set(rc={'figure.figsize':(21.7,5.27)})

sns.barplot(x="product_group",y="confidence",hue="lift",data=df1);
df1.plot.bar()
sns.barplot(x="Pclass",y="survived_grouped_count",hue="AgeGroup",data=df[df.Survived == 1]);