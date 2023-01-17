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
import seaborn as sns
from statistics import stdev
import matplotlib.pyplot as plt
import math
df=pd.read_csv("../input/iris/Iris.csv")
df
SL=4.7
SW=3.7
PL=2
PW=0.3
df['Species'].nunique()
df['Species'].value_counts()
#P(versicolor)
#P(setosa)
#P(virginica)

pver=50/150
pset=50/150
pvir=50/150
df_ver=df[df['Species']=='Iris-versicolor']
df_set=df[df['Species']=='Iris-setosa']
df_vir=df[df['Species']=='Iris-virginica']
plt.figure(figsize=(20,10))
sns.distplot(df_ver['SepalLengthCm'])
sns.distplot(df_ver['SepalWidthCm'])
sns.distplot(df_ver['PetalLengthCm'])
sns.distplot(df_ver['PetalWidthCm'])
plt.figure(figsize=(20,10))
sns.distplot(df_set['SepalLengthCm'])
sns.distplot(df_set['SepalWidthCm'])
sns.distplot(df_set['PetalLengthCm'])
sns.distplot(df_set['PetalWidthCm'])
plt.figure(figsize=(20,10))
sns.distplot(df_vir['SepalLengthCm'])
sns.distplot(df_vir['SepalWidthCm'])
sns.distplot(df_vir['PetalLengthCm'])
sns.distplot(df_vir['PetalWidthCm'])
sns.boxplot(df_set['PetalLengthCm'])
df_set[(df_set['PetalLengthCm']<1.2) | (df_set['PetalLengthCm']>1.7)] #outliers of setosa PL
sns.boxplot(df_set['PetalLengthCm'])
sns.distplot(df_set['PetalLengthCm'])
sns.boxplot(df_set['PetalWidthCm'])
sns.distplot(df_set['PetalWidthCm'])
df_set[df_set['PetalWidthCm']>0.4]
#function for finding normal distribution

def normal(x,y,z):
    return ((math.exp((-((x*x)+(y*y)-(2*x*y)))/2*z*z))/(z*1.414*3.14))
#P(versicolor|SL=4.7, SW=3.7,PL=2,PW=0.3)=P(SL=4.7|versicolor)P(SW=3.7|versicolor)P(PW=0.3|versicolor)P(PL=2|versicolor)P(versicolor)

#P(SL=4.7|versicolor)
mean_SL_ver=df_ver['SepalLengthCm'].mean()
sd_SL_ver=df_ver['SepalLengthCm'].std()
pslver=normal(SL,mean_SL_ver,sd_SL_ver)

#P(SW=3.7|versicolor)
mean_SW_ver=df_ver['SepalWidthCm'].mean()
sd_SW_ver=df_ver['SepalWidthCm'].std()
pswver=normal(SW,mean_SW_ver,sd_SW_ver)

#P(PL=2|versicolor)
mean_PL_ver=df_ver['PetalLengthCm'].mean()
sd_PL_ver=df_ver['PetalLengthCm'].std()
pplver=normal(PL,mean_PL_ver,sd_PL_ver)

#P(PW=0.3|versicolor)
mean_PW_ver=df_ver['PetalWidthCm'].mean()
sd_PW_ver=df_ver['PetalWidthCm'].std()
ppwver=normal(PW,mean_PW_ver,sd_PW_ver)

pversicolor=pslver*pswver*ppwver*pplver*pver
pversicolor
#P(setosa|SL=4.7, SW=3.7,PL=2,PW=0.3)=P(SL=4.7|setosa)P(SW=3.7|setosa)P(PW=0.3|setosa)P(PL=2|setosa)P(setosa)

#P(SL=4.7|setosa)
mean_SL_set=df_set['SepalLengthCm'].mean()
sd_SL_set=df_set['SepalLengthCm'].std()
pslset=normal(SL,mean_SL_set,sd_SL_set)

#P(SW=3.7|setosa)
mean_SW_set=df_set['SepalWidthCm'].mean()
sd_SW_set=df_set['SepalWidthCm'].std()
pswset=normal(SW,mean_SW_set,sd_SW_set)

#P(PW=0.3|setosa)P(PL=2|setosa)P(setosa)
mean_PW_set=df_set['PetalWidthCm'].mean()
sd_PW_set=df_set['PetalWidthCm'].std()
ppwset=normal(PW,mean_PW_set,sd_PW_set)

#P(PL=2|setosa)
mean_PL_set=df_set['PetalLengthCm'].mean()
sd_PL_set=df_set['PetalLengthCm'].std()
pplset=normal(PL,mean_PL_set,sd_PL_set)

psetosa=pslset*pswset*ppwset*pplset*pset
psetosa
#P(virginica|SL=4.7, SW=3.7,PL=2,PW=0.3)=P(SL=4.7|virginica)P(SW=3.7|virginica)P(PW=0.3|virginica)P(PL=2|virginica)P(virginica)

#P(SL=4.7|virginica)
mean_SL_vir=df_vir['SepalLengthCm'].mean()
sd_SL_vir=df_vir['SepalLengthCm'].std()
pslvir=normal(SL,mean_SL_vir,sd_SL_vir)

#P(SW=3.7|virginica)
mean_SW_vir=df_vir['SepalWidthCm'].mean()
sd_SW_vir=df_vir['SepalWidthCm'].std()
pswvir=normal(SW,mean_SW_vir,sd_SW_vir)

#P(PW=0.3|virginica)
mean_PW_vir=df_vir['PetalWidthCm'].mean()
sd_PW_vir=df_vir['PetalWidthCm'].std()
ppwvir=normal(PW,mean_PW_vir,sd_PW_vir)

#P(PL=2|virginica)
mean_PL_vir=df_vir['PetalLengthCm'].mean()
sd_PL_vir=df_vir['PetalLengthCm'].std()
pplvir=normal(PL,mean_PL_vir,sd_PL_vir)

pvirginica=pslvir*pswvir*ppwvir*pplvir*pvir
pvirginica
def maxi(a, b, c): 
  
    if (a >= b) and (a >= c): 
        largest = a 
  
    elif (b >= a) and (b >= c): 
        largest = b 
    else: 
        largest = c 
          
    return largest 
maxi(pversicolor,psetosa,pvirginica)
