# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# -*- coding: UTF-8 -*-

#coding:utf-8

from matplotlib import pyplot as plt

from matplotlib import animation



plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['font.family']='sans-serif'

plt.rcParams['axes.unicode_minus']=False





takeAway = pd.read_csv('/kaggle/input/catering-takeaway-supply-list-of-beijing-area/TakeAway.csv')

takeAwayf=takeAway.copy()
print(takeAwayf.info())
print(takeAwayf.head(10))
takeAwayf['所在区县']=takeAwayf['所在区县'].apply(lambda x: '大兴区' if x=='经济技术开发区' else x)
#area of each region

area = {"东城区": 42, "西城区": 51, "朝阳区": 471,

        "丰台区": 304, "石景山区": 86,"海淀区":431,

        "顺义区":1021,"通州区":906,"大兴区":1036,

        "房山区":2019,"门头沟区":1451,"昌平区":1344,

        "平谷区":950,"密云区":2229,"怀柔区":2123,"延庆区":1994}
#create a data frame to analysis each region's takeaway restaurants number and density

takeAwayf["面积(平方千米)"] = takeAwayf['所在区县'].map(area)



areaCount=takeAwayf['所在区县'].value_counts()

areaCount=areaCount.to_frame()

areaCountDataFrame=areaCount.rename(columns={'所在区县':'count'})



areaDataFrame=pd.DataFrame.from_dict(area,orient='index')

areaDataFrame=areaDataFrame.rename(columns={0:'area'})



areainfo=pd.concat([areaCountDataFrame, areaDataFrame], axis=1,sort=True)



areainfo['density']=round(areainfo['count']/areainfo['area'],3)



areainfo=areainfo.sort_values(by='density', ascending=False)



print(areainfo)
#Proportion of Takeaway Foods by Category

Rtype=takeAwayf['分类'].str.split('+',expand=True)

Rtype=Rtype.rename(columns={0:"分类1",1:'分类2'})



takeAwayf=pd.concat([takeAwayf,Rtype],axis=1)







rType=takeAwayf['分类1'].value_counts().to_frame()



rType2=takeAwayf['分类2'].value_counts().to_frame()





for n in range(len(rType.index)):

        for m in range(len(rType2.index)):

                if rType.index[n]==rType2.index[m]:

                        rType.values[n][0]+=rType2.values[m][0]



rType=rType.rename(index=str,columns={'分类1':'总计'})



explode=(0.1,)+((0,)*(len(rType.index)-1))

#print(explode)

plt.pie(rType['总计'],labels=rType.index.tolist(),autopct='%1.1f%%',explode=explode)

plt.title("餐饮店分类分布占比情况")

plt.show()
#Proportion of take-out restaurant brands

brand=takeAwayf['品牌'].value_counts().to_frame()

brand=brand.rename(index=str,columns={'品牌':'总计'})



totalBrand=takeAwayf['品牌'].count()



precentage=[]

for i in range(len(brand.values)):

        precentage.append(brand.values[i][0]/totalBrand)



brand['占比']=precentage



brand['累计占比']=brand['占比'].cumsum()



topTenBrand=brand[:10]



vis = plt.bar(topTenBrand.index.tolist(), height=topTenBrand['总计'], width=0.5, label="餐厅数量", tick_label=topTenBrand.index.tolist())



for a, b in zip(topTenBrand.index.tolist(), topTenBrand['总计']):

    plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)



plt.title("外卖前十餐饮品牌")

plt.show()