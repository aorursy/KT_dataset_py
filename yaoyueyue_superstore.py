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
df = pd.read_csv("../input/superstore-data/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")
df.head()

#查看数据集大小，有多少行，多少列

df.shape
df.dtypes

####

#Row ID              int64                  #行编号

#Order ID           object                  #订单ID

#Order Date         object                #订单日期

#Ship Date          object                 #发货日期

#Ship Mode          object                #发货模式

#Customer ID        object                #客户ID

#Customer Name      object             #客户姓名

#Segment            object                 #客户类别

#City               object                     #客户所在城市

#State              object                    #客户城市所在州

#Country            object                   #客户所在国家

#Postal Code       float64                 #邮编

#Market             object                    #market所属区域

#Region             object                    #market所属洲

#Product ID         object                   #产品ID

#Category           object                   #产品类别

#Sub-Category       object                #产品自类别

#Product Name       object                #产品名称

#Sales             float64                     #销售额

#Quantity            int64                    #销售量

#Discount          float64                   #折扣

#Profit            float64                      #利润

#Shipping Cost     float64                 #发货成本

#Order Priority     object                   #订单优先级###
#数据格式不一致，拆分为两个数据集

data1=df.loc[0:20066,:]

data2=df.loc[20067:51289,:]

#数据格式转换，改为时间格式

data1.loc[:,'Order Date']=pd.to_datetime(df.loc[:,'Order Date'],format='%d/%m/%Y',errors='coerce')

data2.loc[:,'Order Date']=pd.to_datetime(df.loc[:,'Order Date'],format='%d-%m-%Y',errors='coerce')

#合并data1和data2

data=data1.append(data2)

data.head()
#按Order Date进行排序

data_1=data.sort_values(by='Order Date',ascending=True,na_position='first')

data_1.head()
#截取订单日期中的年月

from datetime import datetime #导入datetime模块

dt=data_1['Order Date'].astype(str)#将字段转化为字符格式

dt=dt.apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))

data_1['year']=dt.map(lambda x:x.year)#获取年份，并添加列

data_1['month']=dt.map(lambda x:x.month)#获取月份，并添加列month
data_1.head()
################选择子集-销售分析子集###############

sales=data_1[['Order Date','Sales','Profit','year','month']]

sales.head()
#查看缺失值

sales[sales.isnull().values==True]
#查看sales的基本描述信息，是否有异常值

sales.describe()
import matplotlib.pyplot as plt

%matplotlib inline
########################对整体的运营情况分析，了解运营状况###############################

#计算年度&月度销售额&利润

gb=sales.groupby(['year','month'])#按照年月分组

sales_year=gb.sum()#分组后求和，得到不同年份不同月份的销售额

sales_year
##################构建单独的销售表#################

year_2011=sales_year.loc[(2011,slice(None)),:].reset_index()

year_2012=sales_year.loc[(2012,slice(None)),:].reset_index()

year_2013=sales_year.loc[(2013,slice(None)),:].reset_index()

year_2014=sales_year.loc[(2014,slice(None)),:].reset_index()

#对上述分组的数据进行拆分，获取每年每月的销售额与利润表

year_2011
####构建销售表###

sales_data=pd.concat([year_2011['Sales'],year_2012['Sales'],

                      year_2013['Sales'],year_2014['Sales']],axis=1)

sales_data

#对行列重命名

sales_data.columns=['sales-2011','sales-2012','sales-2013','sales-2014']

sales_data.index=['Jau','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

sales_data
##计算2011年-2014年年度总销售额及增长率

sales_sum_data=sales_data.sum()

sales_sum_data

#sales-2011    2.259451e+06

#sales-2012    2.677439e+06

#sales-2013    3.405746e+06

#sales-2014    4.299866e+06

#dtype: float64



sales_sum_data.plot(kind='bar',colormap='RdYlGn_r',alpha=0.5)

plt.grid()
#计算每年增长率

rise_12=sales_sum_data[1]/sales_sum_data[0]-1

rise_13=sales_sum_data[2]/sales_sum_data[1]-1

rise_14=sales_sum_data[3]/sales_sum_data[2]-1

rise_rate=[0,rise_12,rise_13,rise_14]

#表格显示增长率

sales_sum=pd.DataFrame({'sales_sum_data':sales_sum_data})

sales_sum['rise_rate']=rise_rate

sales_sum
######################################################
#对每年每月销售额进行总体预览

sales_data.style.background_gradient(cmap='Reds',axis =0)
##面积堆积图，每年每月销售的对比

sales_data.plot.area(colormap='RdYlGn_r',stacked=False)
rise=pd.DataFrame()
#计算每月同比增长率

rise=pd.DataFrame()

rise['rise_2012']=(sales_data['sales-2012']-sales_data['sales-2011'])/sales_data['sales-2011']

rise['rise_2013']=(sales_data['sales-2013']-sales_data['sales-2012'])/sales_data['sales-2012']

rise['rise_2014']=(sales_data['sales-2014']-sales_data['sales-2013'])/sales_data['sales-2013']

rise



# 表格色阶显示

rise.style.background_gradient(cmap='Greens',axis =1,low=0,high=1)
##############新建利润表###############

profit_data=pd.concat([year_2011['Profit'],year_2012['Profit'],

                      year_2013['Profit'],year_2014['Profit']],axis=1)

profit_data

#对行列重命名

profit_data.columns=['Profit-2011','Profit-2012','Profit-2013','Profit-2014']

profit_data.index=['Jau','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

profit_data
##计算2011年-2014年年度总销售额及增长率

profit_sum_data=profit_data.sum()

profit_sum_data

#Profit-2011    248940.81154

#Profit-2012    307415.27910

#Profit-2013    406935.23018

#Profit-2014    504165.97046

#dtype: float64



profit_sum_data.plot(kind='bar',colormap='RdYlGn_r',alpha=0.5)

plt.grid()
#计算每年利润增长率

profit_rise_12=profit_sum_data[1]/profit_sum_data[0]-1

profit_rise_13=profit_sum_data[2]/profit_sum_data[1]-1

profit_rise_14=profit_sum_data[3]/profit_sum_data[2]-1

profit_rise_rate=[0,profit_rise_12,profit_rise_13,profit_rise_14]

###每年利润率

profit_rate_11=profit_sum_data[0]/sales_sum_data[0]

profit_rate_12=profit_sum_data[1]/sales_sum_data[1]

profit_rate_13=profit_sum_data[2]/sales_sum_data[2]

profit_rate_14=profit_sum_data[3]/sales_sum_data[3]

profit_rate=[profit_rate_11,profit_rate_12,profit_rate_13,profit_rate_11]

#表格显示增长率

profit_sum=pd.DataFrame({'profit_sum_data':profit_sum_data})

profit_sum['rise_rate']=profit_rise_rate

profit_sum['profit_rate']=profit_rate

profit_sum