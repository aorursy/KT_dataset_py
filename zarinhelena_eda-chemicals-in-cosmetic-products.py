import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df_original = pd.read_csv('../input/chemicals-in-cosmetics/chemicals-in-cosmetics.csv')

df = df_original.drop_duplicates()

print('The original database shape:', df_original.shape)

print('Database without duplicates:', df.shape)
df.head()
df['ChemicalName'].value_counts().size
df['ChemicalCount'].describe()
df.loc[df.ChemicalCount==0].head()
# when the result is False, there are no NaN values

df.loc[df.ChemicalCount==0]['ChemicalDateRemoved'].isnull().max()
df_n0 = df.loc[(df.ChemicalCount>0) & (df['DiscontinuedDate'].isna())]
df_n0.loc[df.ChemicalCount==9]
df_n0.loc[df['CDPHId']==26]
data = df_n0.groupby(['ChemicalCount']).nunique()['CDPHId']



fig = plt.figure(figsize=(9,7))

ax = plt.subplot(111)

ax.bar(data.index, data.values, log=True, align='center', alpha=0.5, edgecolor='k')



ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')

ax.xaxis.set_ticks_position('bottom')

ax.set_xticks(np.arange(1,10))



for x,y in zip(data.index,data.values):

    plt.annotate(y, (x,y), textcoords="offset points", xytext=(0,4), ha='center') 



ax.set_title('Number of reported products containing chemicals', fontsize=15)

ax.title.set_position([.5, 1.05])

ax.set_xlabel('Number of chemicals', fontsize=12)

ax.set_ylabel('Number of products (log scale)', fontsize=12)



plt.show()
baby_prod = df_n0.loc[df_n0['PrimaryCategory']=='Baby Products']

baby_prod.head()
baby_prod_chem = baby_prod['ChemicalName'].value_counts()

print(baby_prod_chem)
long_text = baby_prod_chem.index[2]

print('Old chemical name: ', long_text)

print()

baby_prod_chem.rename({baby_prod_chem.index[2]: 'Retinol *'}, inplace=True)

print('New chemical name: ', baby_prod_chem.index[2])
fig = plt.figure(figsize=(10, 6))

ax = plt.subplot(111)

ax.barh(baby_prod_chem.index, baby_prod_chem.values, color='red', alpha=0.6)



ax.xaxis.grid(linestyle='--', linewidth=0.5)



for x,y in zip(baby_prod_chem.values,baby_prod_chem.index):

    ax.annotate(x, (x,y), textcoords="offset points", xytext=(4,0), va='center') 



ax.set_title('Chemicals in baby products', fontsize=15)

ax.title.set_position([0.5,1.02])

ax.set_xlabel('Number of baby products', fontsize=12)

ax.set_xticks(np.arange(0,18,5))

plt.text(-0.15,-0.2, "* "+long_text, size=12, transform=ax.transAxes)



plt.show()
reported_baby_prod = baby_prod[['ProductName', 'CompanyName', 'SubCategory']].sort_values('SubCategory')

reported_baby_prod.columns=['Baby product', 'Company', 'Type of product']

reported_baby_prod.style.hide_index()