# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/World_Bank_Data_India.csv')
data.head()
# Convert fractional indicators to actual numbers.
data['UEM_TOTL']= data['UEM_TOTL']*data['LF_TOTL']/100
data['LF_BASC_EDU']= data['LF_BASC_EDU']*data['LF_TOTL']/100
data['LF_INTM_EDU']= data['LF_INTM_EDU']*data['LF_TOTL']/100
data['LF_ADVN_EDU']= data['LF_ADVN_EDU']*data['LF_TOTL']/100
data['EMP_SRV']= data['EMP_SRV']*data['EMP_TOTL']/100
data['EMP_IND']= data['EMP_IND']*data['EMP_TOTL']/100
data['EMP_AGR']= data['EMP_AGR']*data['EMP_TOTL']/100
data['GDP_SRV']=100-data['GDP_AGR']-data['GDP_IND']
data.head()
fig, axes1 = plt.subplots(figsize=(10,8))

axes1.plot(data['Years'],data['POP_TOTL'],color='red',lw=4)
axes1.plot(data['Years'],data['POP_014'],color='green',lw=3)
axes1.plot(data['Years'],data['POP_1564'],color='blue',lw=3)
axes1.plot(data['Years'],data['POP_65'],color='black',lw=3)
axes1.set_title("Population of different ages for last 26 Years")
axes1.set_xlabel("Years")
axes1.set_ylabel("Population in Billions")
axes1.legend(loc=0)
fig, axes2 = plt.subplots(figsize=(10,8))

axes2.plot(data['Years'],data['POP_1564'],color='red',lw=4)
axes2.plot(data['Years'],data['LF_TOTL'],color='green',lw=3)
axes2.plot(data['Years'],data['EMP_TOTL'],color='blue',lw=2)
axes2.plot(data['Years'],data['UEM_TOTL'],color='black',lw=2)
axes2.fill_between(data['Years'], data['POP_1564'], data['LF_TOTL'], color="yellow", alpha=0.5)
axes2.set_title("Employment rate for Last 26 years vs Population rise")
axes2.set_xlabel("Years")
axes2.set_ylabel("Population in Billions")
axes2.legend(loc=0)
fig, axes3 = plt.subplots(figsize=(15,8))

axes3.plot(data['Years'],data['LF_BASC_EDU'],color='#008B8B',lw=3, marker='o', markersize=10)
axes3.plot(data['Years'],data['LF_INTM_EDU'],color='#0000ff',lw=3, marker='p', markersize=10)
axes3.plot(data['Years'],data['LF_ADVN_EDU'],color='#00EEEE',lw=3, marker='s', markersize=10)
axes3.plot(data['Years'],data['EMP_SRV'],color='#ff0000',lw=4)
axes3.plot(data['Years'],data['EMP_IND'],color='#660000',lw=4)
axes3.plot(data['Years'],data['EMP_AGR'],color='#cc0000',lw=4)
axes3.set_title("Rate of Education level of our Labur force and their sector wise Employment")
axes3.set_xlabel("Years")
axes3.set_ylabel("Population in 100 millions")
axes3.legend(loc=0)
fig, axes4 = plt.subplots(figsize=(15,8))

axes4.plot(data['Years'],data['GDP_SRV'],color='#ff0000',lw=4)
axes4.plot(data['Years'],data['GDP_IND'],color='#660000',lw=4)
axes4.plot(data['Years'],data['GDP_AGR'],color='#cc0000',lw=4)
axes4.set_title("Rate of GDP sector wise")
axes4.set_xlabel("Years")
axes4.set_ylabel("% of total GDP")
axes4.legend(loc=0)
fig, axes5 = plt.subplots(1,4,figsize=(20,8))

axes5[0].plot(data['Years'],data['GDP_PP_EMP'],color='green',lw=3)
axes5[0].set_title("GDP per person employed")
axes5[1].plot(data['Years'],data['INFL'],color='blue',lw=3)
axes5[1].set_title("Inflation, consumer prices")
axes5[2].plot(data['Years'],data['XPD_TOTL_EDU'],color='black',lw=3)
axes5[2].set_title("Government expenditure on education")
axes5[3].plot(data['Years'],data['FRTL'],color='Orange',lw=3)
axes5[3].set_title("Fertility rate, total (births per woman)")
fig, axes6 = plt.subplots(1,2,figsize=(20,10))

axes6[0].plot(data['Years'],data['BR'],color='green',lw=3)
axes6[0].set_title("Birth Rate per 1000 people")
axes6[0].plot(data['Years'],data['DR'],color='red',lw=3)
axes6[0].set_title("Death Rate")
axes6[0].set_xlabel("Years")
axes6[0].set_ylabel("Person per 1000")
axes6[0].legend(loc=0)
axes6[1].plot(data['Years'],data['D_COM'],color='#ff0000',lw=3, marker='o', markersize=10)
axes6[1].plot(data['Years'],data['D_INJ'],color='#660000',lw=3, marker='p', markersize=10)
axes6[1].plot(data['Years'],data['D_NC'],color='#cc0000',lw=3, marker='s', markersize=10)
axes6[1].set_title("Death Rate due to different Reasons")
axes6[1].set_xlabel("Years")
axes6[1].set_ylabel("% of Death Rate")
axes6[1].legend(loc=0)
