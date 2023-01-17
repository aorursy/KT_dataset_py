import os

print(os.listdir("../input"))
%matplotlib inline

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 
os.chdir('../input')

data=pd.read_csv('2004-2019.tsv', sep='\t',parse_dates=[1,2],index_col=0)
data.shape
data.head()
data.columns
data.rename(columns={'DATA INICIAL':'date_first', 

                     'DATA FINAL':'date_last',

                     'REGIÃO':'macro_region',

                     'ESTADO':'state',

                     'PRODUTO':'product',

                     'NÚMERO DE POSTOS PESQUISADOS':'num_gas_station', 

                     'UNIDADE DE MEDIDA':'unit',

                     'PREÇO MÉDIO REVENDA':'mean_mkt_value',

                     'DESVIO PADRÃO REVENDA':'sd',

                     'PREÇO MÍNIMO REVENDA':'min_price',

                     'PREÇO MÁXIMO REVENDA':'max_price',

                     'MARGEM MÉDIA REVENDA':'mean_price_margin',

                     'COEF DE VARIAÇÃO REVENDA':'coef_var',

                     'PREÇO MÉDIO DISTRIBUIÇÃO':'mean_dist_price',

                     'DESVIO PADRÃO DISTRIBUIÇÃO':'dist_sd',

                     'PREÇO MÍNIMO DISTRIBUIÇÃO':'dist_min_price',

                     'PREÇO MÁXIMO DISTRIBUIÇÃO':'dist_max_price',

                     'COEF DE VARIAÇÃO DISTRIBUIÇÃO':'dist_coef_var',

                     'MÊS':'month',

                     'ANO':'year'}

            , inplace=True)
data.head()
for i in list(data.columns):

    print(i,len(set(data[i])))
data.isnull().any()
data.info()
for col in ['mean_price_margin','mean_dist_price','dist_sd','dist_min_price','dist_max_price','dist_coef_var']:

    data[col]=pd.to_numeric(data[col],errors='coerce')
data.info()
data.isnull().any()
data.isnull().sum()
data.isnull().any(axis=1).sum()
data.dropna(inplace=True)
data.shape
data.head()
data.describe()
set(zip(data['product'],data['unit']))
avg_price_by_prod_region = pd.pivot_table(data,index=['product','year'],columns='macro_region',values='mean_mkt_value',aggfunc=np.mean)

avg_price_by_prod_region.head()
for i in set(avg_price_by_prod_region.index.get_level_values(0)):

    sub_table = avg_price_by_prod_region[avg_price_by_prod_region.index.get_level_values(0)==i].reset_index(drop=False)

    sub_table =avg_price_by_prod_region[avg_price_by_prod_region.index.get_level_values(0)==i].reset_index(drop=False)

    sub_table.drop('product',axis=1,inplace=True)

    sub_table.set_index('year').plot()

    plt.title('Price change for {}'.format(i))

    plt.show()
avg_price_by_prod_region_states = pd.pivot_table(data,index=['macro_region','state','year'],columns='product',values='mean_mkt_value',aggfunc=np.mean)

avg_price_by_prod_region_states.head(10)
avg_price_by_prod_region_states.reset_index(drop=False,inplace=True)
price_increase = {}



for state in set(data['state']):

    for prod in set(data['product']):

        pct_change = []

        price_list = avg_price_by_prod_region_states[avg_price_by_prod_region_states['state']==state][prod]

        price_list = price_list[~np.isnan(price_list)]

        pct_change.append(price_list.pct_change().mean())

    price_increase[state] = np.mean(pct_change)
price_increase_df = pd.DataFrame.from_dict(price_increase,orient ='index')

price_increase_df.reset_index(drop=False,inplace=True)

price_increase_df.columns=['state','price_increase']
region_state = pd.DataFrame(list(set(zip(data['state'],data['macro_region']))),columns=['state','region'])
price_increase_result = pd.merge(price_increase_df,region_state,left_on='state',right_on='state')
price_increase_result.sort_values('price_increase',ascending=False).drop_duplicates(['region'])
avg_price_ovreall_by_prod_state = pd.pivot_table(data,index=['state'],columns='product',values='mean_mkt_value',aggfunc=np.mean)
avg_price_ovreall_by_prod_state.head()
avg_price_ovreall_by_prod_state.idxmax()
avg_price_ovreall_by_prod_state.idxmin()