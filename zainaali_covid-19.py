# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

#for jupyter notebook we use this line

%matplotlib inline 

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#loading data

nCoronaData=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#Check the 10 samples for data

nCoronaData.head(10)
#rename columns names 

nCoronaData=nCoronaData.rename(columns={ "Country/Region": "Country", "Province/State": "Province"})
#delete SNo column

nCoronaData.drop(['SNo'],axis=1)
#Check the last 10 samples for data

#Last update 8-4-2020

nCoronaData.tail(10)
#check simple information like  columns names ,  columns datatypes and null values

nCoronaData.info()
#check summary of numerical data  such as count , mean , max , min  and standard deviation.

nCoronaData.describe()
nCoronaData["ObservationDate"].value_counts() [:10]
nCoronaData['Country'].value_counts().head(10)

nCoronaData['Province'].value_counts().head(10)
from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

nCoronaData.groupby('Country').sum()['Confirmed'].bar()

plt.title('No of confirmed cases in country ')

plt.tight_layout()
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')



nCoronaData[nCoronaData['Country']=='US'].groupby('Province').count()['Deaths'].plot()

plt.title('US_Provinces & No of Deaths in Province/State ')

plt.show()
figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')

nCoronaData[nCoronaData['Country']=='US'].groupby('Province').count()['Confirmed'].plot()

plt.title('No of cases in every State in US')

plt.show()