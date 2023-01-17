# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Display full output rather than just the last line of output

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 500)
df = pd.read_csv('/kaggle/input/kernel51d9b7c83e/raw_data_no_dummies_imputed.csv')

df.head()
'''########### FIX TO 10% ########## -------- Identify columns with more than 30% missing values and remove them



col_to_delete = df.columns[df.isna().sum() >= 0.30*len(df)].tolist()



ncols_before = df.shape[1]



  ######

#Keep Hybrid columns (['Hybrid/Electric Components Miles/km', 'Hybrid/Electric Components Years', 'Hybrid/Electric Components Note', 'Hybrid'] )

#                 even if they have many missing values

hyb_cols = [col for col in df if 'ybri' in col]



for x in hyb_cols, 'EPA Classification':

    if x in col_to_delete:

        col_to_delete.remove(x)

  ######



df.drop(col_to_delete, axis=1, inplace=True)



print('Number of columns removed: '+ str(ncols_before - df.shape[1]))

print('Current number of columns: ' + str(df.shape[1]))

df.head()'''
#split to electric and petrol cars

df_hyb = df.loc[df['Electric']==1]

df = df.loc[df['Electric']!=1]



df_hyb.shape

df.shape
df.shape
df.describe()
skewness = df.skew().to_frame('skew')



print('Positivily Skewed >1')

skewness.loc[skewness['skew']>1].sort_values(by='skew', ascending=False)

print('Negatively Skewed <-1')

skewness.loc[skewness['skew']<-1].sort_values(by='skew')
#Number of Cars per year

df.groupby('Year')['Year'].count().sort_values(ascending=False)
# it is the same as previous corr plot

f, ax = plt.subplots(figsize=(50, 50))

corr = df.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(200, 10, as_cmap=True),

            square=True, ax=ax)

#get details for the most expensive car

df.loc[df.MSRP.idxmax()][[0,1]]



#get details for the least expensive car

df.loc[df.MSRP.idxmin()][[0,1]]
#number of cars per company and company name, sorted by compact



company_cat_count = df.groupby(['Company Name', 'EPA Classification'])['MSRP'].count().to_frame().unstack()

company_cat_count.sort_values(by=('MSRP', 'Compact'), ascending=False).head(20)
fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df["Company Name"].loc[(df["EPA Classification"]=='Compact', )], y=df["MSRP"].loc[df["EPA Classification"]=='Compact'] )
#Number of NaNs

df.isna().sum().sort_values(ascending=False)
#many outliers for price

df.boxplot('MSRP', figsize=(20,10))
df.MSRP[df.MSRP<100000].hist(bins=100) #NEG skewed
#number of cars per company and company name, sorted by compact after 2017



company_cat_count = df[df.Year>2017].groupby(['Company Name', 'EPA Classification'])['MSRP'].count().to_frame().unstack()

company_cat_count.sort_values(by=('MSRP', 'Compact'), ascending=False).head(20)
fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df["Company Name"].loc[df["MSRP"]<100000], y=df["MSRP"].loc[df["MSRP"]<100000] )
fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df["Company Name"].loc[df["EPA Classification"]=='Compact'], y=df["MSRP"].loc[df["EPA Classification"]=='Compact'] )
fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df["Company Name"].loc[df["EPA Classification"]=='Midsize'], y=df["MSRP"].loc[df["EPA Classification"]=='Midsize'] )
#Year of first Hybrid car

df_hyb['Year'].min()
#number of cars per company and company name, sorted by compact HYBRID

company_cat_count_hyb = df_hyb.groupby(['Company Name', 'EPA Classification'])['MSRP'].count().to_frame().unstack()

company_cat_count_hyb.sort_values(by=('MSRP', 'Compact'), ascending=False).head(20)
print(df_hyb["Company Name"].loc[df_hyb["EPA Classification"]=='Compact'].value_counts())



fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df_hyb["Company Name"].loc[df_hyb["EPA Classification"]=='Compact'], y=df_hyb["MSRP"].loc[df_hyb["EPA Classification"]=='Compact'] )
print(df_hyb["Company Name"].loc[df_hyb["EPA Classification"]=='Midsize'].value_counts())

fig = plt.figure(figsize=(40, 10))

sns.boxplot( x=df_hyb["Company Name"].loc[df_hyb["EPA Classification"]=='Midsize'], y=df_hyb["MSRP"].loc[df_hyb["EPA Classification"]=='Midsize'] )
mean_mpg_hybrid = df_hyb.groupby('Company Name')['EPA Fuel Economy Est - Hwy (MPG)'].mean().dropna().sort_values(ascending=False)



#Save the name of companies which create hyb



mean_mpg_hybrid_index = mean_mpg_hybrid.index

mean_mpg_hybrid
mean_mpg_nonhybrid = df.groupby('Company Name')['EPA Fuel Economy Est - Hwy (MPG)'].mean().dropna().sort_values(ascending=False)



#Save the companies which also create hyb



mean_mpg_nonhybrid.loc[mean_mpg_nonhybrid.index.isin(mean_mpg_hybrid_index)]
max_mpg_hybrid = df_hyb.groupby('Company Name')['EPA Fuel Economy Est - Hwy (MPG)'].max().dropna().sort_values(ascending=False)



#Save the name of companies which create hyb



max_mpg_hybrid_index = max_mpg_hybrid.index

max_mpg_hybrid
max_mpg_nonhybrid = df.groupby('Company Name')['EPA Fuel Economy Est - Hwy (MPG)'].max().dropna().sort_values(ascending=False)



#Save the companies which also create hyb



max_mpg_nonhybrid.loc[max_mpg_nonhybrid.index.isin(mean_mpg_hybrid_index)]
mean_price_hybrid_compact = df_hyb.loc[df_hyb["EPA Classification"]=='Compact'].groupby(['Company Name'])['MSRP'].mean().dropna().sort_values(ascending=False)





#keep the name of the companies which create hybrid compact

mean_price_hybrid_compact_index = mean_price_hybrid_compact.index



mean_price_hybrid_compact
mean_price_nonhybrid_compact = df.loc[df["EPA Classification"]=='Compact'].groupby(['Company Name'])['MSRP'].mean().dropna().sort_values(ascending=False)



#filter to keep only the companies which also create hybrid compact cars



mean_price_nonhybrid_compact[mean_price_nonhybrid_compact.index.isin(mean_price_hybrid_compact_index)]

mean_price_hybrid_compact = df_hyb.loc[df_hyb["EPA Classification"]=='Compact'].groupby(['Company Name'])['MSRP'].max().dropna().sort_values(ascending=False)





#keep the name of the companies which create hybrid compact

mean_price_hybrid_compact_index = mean_price_hybrid_compact.index



mean_price_hybrid_compact
mean_price_nonhybrid_compact = df.loc[df["EPA Classification"]=='Compact'].groupby(['Company Name'])['MSRP'].max().dropna().sort_values(ascending=False)



#filter to keep only the companies which also create hybrid compact cars



mean_price_nonhybrid_compact[mean_price_nonhybrid_compact.index.isin(mean_price_hybrid_compact_index)]


