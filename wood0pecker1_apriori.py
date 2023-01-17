import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
dt=pd.read_excel('../input/Online Retail.xlsx')
dt.head()
dt.tail()
dt.Description.unique()
dt.Description=dt.Description.str.strip()
dt.Description[:5]
dt.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
dt['InvoiceNo'] = dt['InvoiceNo'].astype('str')
dt.dtypes
df=dt.copy()
df = df[~df['InvoiceNo'].str.contains('C')]
data=df[df.Country=='France'].groupby(['InvoiceNo', 'Description'])['Quantity'].sum()
data[:5]
data=data.unstack().reset_index().fillna('0')
data.set_index('InvoiceNo').head()
data.columns=data.columns.str.replace(' ','_')
for cl in data.columns:
    data[cl]=data[cl].astype(int)
def onehot(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
dataset=data.applymap(onehot)
dataset=dataset.set_index('InvoiceNo')
frequent_itemsets= apriori(dataset, min_support=0.1, use_colnames=True)
frequent_itemsets.head()
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
rules.sort_values('lift',ascending=False)