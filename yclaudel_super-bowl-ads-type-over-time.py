# Import librairies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')
# Load the data without delimiter (line by line)

temp = pd.read_csv('../input/superbowlads/superbowl-ads.csv', delimiter='^',quotechar='"')

# Line 561 , the 'Year' is missing

temp.iloc[561]='2020,'+temp.iloc[561]

# Split the data

new = temp.iloc[:,0].str.split(',', n = 3, expand = True) 

new[3]=new[3].str.replace('"','')

new[1]=new[1].str.replace('"','')

new[2]=new[2].str.strip('"')

new2=new[2].str.split('""', n = 1, expand = True) 

new = pd.concat([new.iloc[:,[0,1]],new2,new.iloc[:,3]],axis=1).values

# Build the final dataframe

df=pd.DataFrame(new,columns=["Year","Type","Product","Title","Notes"])

df=df.loc[df['Type']!='Product type']

del temp , new , new2

df.head()

dictOfTypes = {'Car': 'Car',

 'Security': 'Security',

 'Copy': 'Copy',

 'Soft drink': 'Beverage',

 'Camera': 'Technology',

 'Computer': 'Technology',

 'Restaurant': 'Food',

 'Beer': 'Beverage',

 'Food': 'Food',

 'Credit card': 'Bank Insurance',

 'Soft Drink': 'Beverage',

 'Footwear': 'Clothing',

 'Clothing': 'Clothing',

 'Film': 'Film',

 'Mail': 'Technology',

 'Shoes': 'Clothing',

 'Sports': 'Sports',

 'Tire': 'Car',

 'Website': 'Technology',

 'Candy': 'Food',

 'TV': 'TV',

 'Gaming': 'Video games',

 'Potato chips': 'Food',

 'PSA': 'PSA',

 'Technology': 'Technology',

 'Manufacturing': 'Car',

 'Mobile phone': 'Phone',

 'TV show': 'TV',

 'Video Game': 'Video games',

 'Store': 'Store',

 'Truck': 'Truck',

 'Product type': 'Product type',

 'Car accessories': 'Car',

 'Adhesives': 'Adhesives',

 'Antifungal medication': 'Care',

 'Feminine care products': 'Care',

 'Insurance': 'Bank Insurance',

 'Lingerie': 'Clothing',

 'Phone': 'Phone',

 'Phone accessory': 'Phone',

 'Service': 'Service',

 'Skincare products': 'Care',

 'Software': 'Technology',

 'Game': 'Video games',

 'Toothpaste': 'Care',

 'Building materials': 'Building materials',

 'Cleaner': 'Cleaner',

 'Tax preparation': 'Bank Insurance',

 'TV series': 'TV',

 'Video games': 'Video games',

 'Web hosting': 'Technology',

 'Wireless': 'Technology',

 'Airlines': 'Travel',

 'Car / Film': 'Car',

 'Drink': 'Beverage',

 'Food / Drink': 'Food',

 'Headphones': 'Phone',

 'Investments': 'Bank Insurance',

 'Laundry detergent': 'Cleaner',

 'Loans': 'Bank Insurance',

 'Retail': 'Retail',

 'TV special': 'TV',

 'Travel': 'Travel',

 'Video game': 'Video games',

 'Alcohol': 'Beverage',

 'Alcohol / TV series': 'Beverage',

 'Beverage': 'Beverage',

 'Smart speaker': 'Phone',

 'NFL': 'NFL',

 'Video gaming': 'Video games'}





df['Type'] = df['Type'].map(dictOfTypes)

df.head(10)
df.info()
df_counts = pd.crosstab(df['Type'], df['Year'])

df_counts.head()
ads_by_year = df_counts.sum(axis=0)

ads_by_year.plot(figsize=(16,9),title="Number of ads over year")

plt.show()
df_counts['Total'] = df_counts.sum(axis=1)

df_counts.sort_values(by='Total', ascending=False,inplace=True)

top = df_counts.loc[df_counts['Total']>10].drop('Total', 1)

filter = list(top.index)

print(50*"#"+"\n# Most frequent type (descending order)\n"+50*"#")

print(filter)

top.head(10)
labels = filter

fig, ax = plt.subplots(figsize=(18,10), dpi= 100) 

ax.stackplot(top.columns, top.values, labels=labels)

ax.legend(loc='upper left')

plt.show()
df['Count']=df['Type']

df_group = df.groupby(['Year','Type'],as_index=False)['Count'].count()

df_group.head()



plot_data = df_group[df_group.Type.isin(['Film', 'Beverage', 'Car', 'Technology', 'Food', 'TV', 'Bank Insurance'])]



fig, ax = plt.subplots(figsize=(18,10), dpi= 100) 

ax = sns.pointplot(x='Year',y='Count',hue='Type',data=plot_data)

plt.show()

col = df_counts.columns

result=[]

for i in range(0,col.shape[0]):

    list_temp = list(df_counts.loc[df_counts[col[i]]>0,col[i]].sort_values(ascending=False)[:10].index)

    for j in range(len(list_temp),10):

        list_temp.append('-')

    result.append(list_temp)

result = np.vstack(result).transpose()

result = pd.DataFrame(data=result,columns=col)
result.iloc[:,:18].head(10)
result.iloc[:,18:].head(10)
from scipy.spatial.distance import pdist, squareform

from scipy.cluster.hierarchy import linkage, dendrogram

from matplotlib.colors import rgb2hex, colorConverter

from scipy.cluster.hierarchy import set_link_color_palette



df_counts.head()

clustdf= df_counts.transpose()

c_dist = pdist(clustdf) # computing the distance

c_link = linkage(clustdf,  metric='correlation', method='complete')# computing the linkage

fig, axes = plt.subplots(1, 1, figsize=(14, 14))

dendro  = dendrogram(c_link,labels=list(df_counts.columns),orientation='right',ax=axes,color_threshold=0.5)

plt.show()
