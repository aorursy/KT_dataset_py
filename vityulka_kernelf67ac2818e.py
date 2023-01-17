import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#full data set

csv_set_full = pd.read_csv('../input/sales_data_sample.csv', encoding = 'unicode_escape')



#create data frame

df = pd.DataFrame(csv_set_full)



#initialize list of data frames

df_init = df[(df['PRODUCTLINE']=='None')]

df_c = [df_init]

df_c.clear()



#define list of years and products

prod_lst = ['Classic Cars','Ships','Planes']

year_lst = [2003,2004,2005]



#populate list of data frames for specific products, years and columns

for i in prod_lst:

    for j in year_lst:

        df_var = df[(df['PRODUCTLINE']==i) & (df['YEAR_ID']==j)]

        df_var = df_var.loc[:,['SALES']]

        df_c.append(df_var)
for i in df_c:

    i.plot.kde()

#initialize list of data frames

s_init = df_c[0]

s_c = [s_init]

s_c.clear()



for i in df_c:

    mu = i.mean(axis=0)



    sigma = i.std(axis=0)



    num = i.count()



    s_c.append(np.random.normal(mu, sigma, num))
sns.distplot(df_c[0])

    

sns.distplot(s_c[0])
sns.distplot(df_c[4])

    

sns.distplot(s_c[4])
sns.distplot(df_c[8])

    

sns.distplot(s_c[8])