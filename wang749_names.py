import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import bq_helper





plt.style.use('ggplot')

sns.set(style='ticks')
#https://www.kaggle.com/salil007/a-very-extensive-exploratory-analysis-usa-names

usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""

data = usa_names.query_to_pandas_safe(query)
names=data.copy()

names.head()
print('数据形状：',names.shape,'\n')

print('可用的年份长度及范围：共',names['year'].nunique(),'年，',names['year'].min(),'至',names['year'].max())

print('总共有多少申请者：',names['count'].count(),'\n')



print('男性申请者所占比例：','{:.2%}'.format(sum(names['count'][names['gender']=='M'])/sum(names['count'])))

print('女性申请者所占比例：','{:.2%}'.format(sum(names['count'][names['gender']=='F'])/sum(names['count'])),'\n')



print('非重复姓名共有多少个：',names['name'].nunique())



print('在此期间最受欢迎的女性名字：')

namesf=names[names['gender']=='F'].loc[:,['name','count']].groupby(by='name').sum()

print(namesf.sort_values('count',ascending=False).head())

print('在此期间最受欢迎的男性名字：')

namesm=names[names['gender']=='M'].loc[:,['name','count']].groupby(by='name').sum()

print(namesm.sort_values('count',ascending=False).head())
names_date=names[names['year']>=1998]

names_date=names_date.groupby('year').sum()

names_date['date']=names_date.index

fig,ax=plt.subplots(figsize=(15,5))

ax.bar(names_date['date'],names_date['count'])



ax.set_xticks(names_date['date'])

ax.set_ylabel('count')

plt.show()
def name_year_count(name):

    names_date20=names[names['year']>=1998]

    agg_name=names_date20[names_date20['name']==name].groupby('year',as_index=False).agg({'count':'sum'})

    if len(agg_name)==0:

        print('无匹配姓名：{0}'.format(name))

    else:

        #groupby后会删除掉count为0的年份，观察对象不完整

        #所以新建year_df匹配groupby之后的agg_name，这样年份就是完整的了，包括count为0的年份

        year_df=pd.DataFrame()

        year_df['year']=names_date20['year'].unique()

        agg_name=pd.merge(year_df,agg_name,on='year',how='left')

        agg_name.fillna(0,inplace=True)

        

        ax=agg_name.plot(x='year',y='count',kind='bar',color='g',alpha=0.8,figsize=(12,5))

        ax.set_ylabel('count',fontsize=12)

        ax.legend_.remove()

        plt.show()
name_year_count('wangxukun')
name_year_count('Daenerys')
name_year_count('Justin')
name_year_count('Tylor')
name_year_count('Arya')