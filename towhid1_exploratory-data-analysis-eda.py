%matplotlib inline
import matplotlib.pyplot as plt #for data visualizing
import seaborn as sns 
color = sns.color_palette()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_csv("../input/echocardiogram.csv")
df.head()
print('Total cols : ',df.shape[1],' and total rows : ',df.shape[0])
print('Missing data sum :')
print(df.isnull().sum())

print('\nMissing data percentage (%):')
print(df.isnull().sum()/df.count()*100)
x=df.isnull().sum()
x=x.sort_values(ascending=False)

plt.figure(figsize=(16,6))
ax= sns.barplot(x.index, x.values, alpha=0.9,color=color[9])
locs, labels = plt.xticks()
plt.setp(labels, rotation=60)
plt.title("Missing value checking",fontsize=20)
plt.ylabel('Missing value sum', fontsize=16)
plt.xlabel('Col name', fontsize=16)
plt.show()
plt.figure(figsize=(10,4))
sns.distplot( df[df['alive'] == 0.0]['age'][~df['age'].isnull()] 
             , color="skyblue",hist=True, label="not alive",rug=True)
sns.distplot( df[df['alive'] == 1.0]['age'][~df['age'].isnull()]
             , color="red",hist=True, label="alive" ,rug=True)
plt.legend()
plt.show();
