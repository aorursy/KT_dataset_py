import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# 这个2b_data需要去修改一下MCM_XXX，然后获得输出

data=pd.read_csv('../input/2020mcm-c-2/2b_data_microwave.csv',sep=',')

data.info()
sales_number=data.groupby(['product_id'],as_index=False)['product_id'].agg({'cnt':'count'})

sales_number.to_csv('./sales_number_dry.csv',index=False)

top_XX=sales_number.nlargest(3,columns='cnt').tail(3)

top_XX
top_product=top_XX['product_id']
small_data={}

len_small_data=len(top_product)

len_small_data
for i in range(len_small_data):

    small_data[i]=data[data['product_id'].isin([top_product.values[i]])]

    # 把review_date转成专用的数据格式

    small_data[i]['review_date']=pd.to_datetime(small_data[i]['review_date'])

    # 设定评论日期为索引列

    small_data[i].set_index(['review_date'],inplace=True)

    small_data[i].info()

    # 摘取完成    
plt.figure(figsize=(10,6))

x0=small_data[0].resample('QS').count().index

y0=small_data[0].resample('QS').count()['final_review_score']

x1=small_data[1].resample('QS').count().index

y1=small_data[1].resample('QS').count()['final_review_score']

x2=small_data[2].resample('QS').count().index

y2=small_data[2].resample('QS').count()['final_review_score']

l0=plt.plot(x0,y0,'r')

l1=plt.plot(x1,y1,'b')

l2=plt.plot(x2,y2,'y')

# my_x_ticks=np.arange(1)

plt.xticks()

plt.show()
plt.figure(figsize=(10,6))

x0=small_data[0].resample('M').count().index

y0=small_data[0].resample('M').mean()['final_review_score']

x1=small_data[1].resample('M').count().index

y1=small_data[1].resample('M').mean()['final_review_score']

x2=small_data[2].resample('M').count().index

y2=small_data[2].resample('M').mean()['final_review_score']

l0=plt.plot(x0,y0,'r')

l1=plt.plot(x1,y1,'b')

l2=plt.plot(x2,y2,'y')

# my_x_ticks=np.arange(1)

plt.xticks()

plt.show()
plt.figure(figsize=(10,6))

x0=small_data[0].resample('QS').count().index

y0=small_data[0].resample('QS').mean()['final_review_score']

x1=small_data[1].resample('QS').count().index

y1=small_data[1].resample('QS').mean()['final_review_score']

x2=small_data[2].resample('QS').count().index

y2=small_data[2].resample('QS').mean()['final_review_score']

l0=plt.plot(x0,y0,'r')

l1=plt.plot(x1,y1,'b')

l2=plt.plot(x2,y2,'y')

# my_x_ticks=np.arange(1)

plt.xticks()

plt.show()
small_data[0].resample('M').mean().to_period('M')['final_review_score'].plot(color='r')

small_data[1].resample('M').mean().to_period('M')['final_review_score'].plot(color='b')

small_data[2].resample('M').mean().to_period('M')['final_review_score'].plot(color='y')
small_data[0].resample('QS').mean().to_period('M')['final_review_score'].plot()
def half_year(classify0):

    classify0.shape

    index=classify0.index

    index_series=pd.Series(index)

    new_index_len=classify0.shape[0]//2

    mean_score=[]

    mean_star=[]

    new_index=[]



    for i in range(new_index_len):

        mean_score.append((classify0.iat[2*i,0]+classify0.iat[2*i+1,0])/2)

        mean_star.append((classify0.iat[2*i,1]+classify0.iat[2*i+1,1])/2)

        new_index.append(index_series[2*i])

    half_year_data={

        'date':new_index,

        'mean_score':mean_score,

        'mean_star':mean_star

    }

    half_year_data=pd.DataFrame(half_year_data)

    half_year_data.set_index(['date'],inplace=True)

    return half_year_data



half_year_data0=half_year(small_data[0].resample('QS').mean().to_period('M'))

half_year_data1=half_year(small_data[1].resample('QS').mean().to_period('M'))

half_year_data2=half_year(small_data[2].resample('QS').mean().to_period('M'))



half_year_data0['mean_score'].plot(color='r')

half_year_data1['mean_score'].plot(color='b')

half_year_data2['mean_score'].plot(color='y')

plt.show()
plt.figure(figsize=(10,6))



x2=small_data[2].resample('QS').count().index

y2=small_data[2].resample('QS').count()['product_id']



l2=plt.plot(x2,y2,'y')

plt.show()
def get_season_data(year,index):

    season_data=small_data[index][str(year)+'-4'].append(small_data[index][str(year)+'-5']).append(small_data[index][str(year)+'-6'])

    return season_data
get_season_data(2014,0).to_csv('./top_1_2014.csv')

get_season_data(2014,1).to_csv('./top_2_2014.csv')

get_season_data(2014,2).to_csv('./top_3_2014.csv')