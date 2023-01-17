# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

bikes=pd.read_csv("../input/train.csv",index_col='datetime',parse_dates=True)
bikes['temp_F']=bikes.temp*1.8+32
bikes.rename(columns={'count':'total'},inplace=1)
bikes['hour']=bikes.index.hour

bikes['temp_cut']=pd.cut(bikes.temp_F,[40,60,80,100])
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize']=(14,12)

plt.rcParams['font.size']=14

sns.set_style("whitegrid")

g=sns.swarmplot(x='hour',y='total',hue='temp_cut',data=bikes)

g.set(xlabel="Hour within a Day", ylabel="Number of bike Rental")