import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
%matplotlib inline
#warnings
import warnings
warnings.filterwarnings('ignore')
#we need to install mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df=pd.read_csv('../input/BreadBasket_DMS.csv')
print(df.info())
df.head(10)
df['Year']=df['Date'].apply(lambda x:x.split("-")[0]) #year
df['Month']=df['Date'].apply(lambda x:x.split("-")[1]) #month
df['Day']=df['Date'].apply(lambda x:x.split("-")[2]) #day
df['Hour']=df['Time'].apply(lambda x:x.split(":")[0]) #Hour
df['Minute']=df['Time'].apply(lambda x:x.split(":")[1]) #minutes
df['Seconds']=df['Time'].apply(lambda x:x.split(":")[2]) #seconds

df.head(10)
sold = df['Item'].value_counts()
sold.head(10)

#visualization
import seaborn as sns
sns.set(style ='whitegrid')

ax = sns.barplot(x='Month', y='Transaction', data=df)
ax = sns.barplot(x='Year', y='Transaction', data=df)


ax = sns.barplot(x='Hour', y='Transaction', data=df)
items_sold = sold.head(20)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
items_sold.plot(kind='bar')
plt.title('Sold Items')
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori
transaction_list = []

# For loop to create a list of the unique transactions throughout the dataset:
for i in df['Transaction'].unique():
    tlist = list(set(df[df['Transaction']==i]['Item']))
    if len(tlist)>0:
        transaction_list.append(tlist)
print(len(transaction_list))
te = TransactionEncoder()
te_ary=te.fit(transaction_list).transform(transaction_list)
df2=pd.DataFrame(te_ary, columns = te.columns_)
# Apply apriori algorithm to know how items are bought together by fixing more than 1 as limit and sort the values by confidence 
items = apriori(df2, min_support=0.03, use_colnames=True)
rules = association_rules(items, metric='lift', min_threshold=1.0)
rules.sort_values('confidence', ascending=False)
