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
import pandas as pd

import seaborn as sns

import warnings



warnings.filterwarnings("ignore")



#read data as pandas data frame



data=pd.read_csv("/kaggle/input/calcofi/bottle.csv")
data.head()
import numpy as np

from scipy import stats
median_depthm = np.median(data.Depthm)

print('median depthm:',median_depthm)



mean_depthm = np.mean(data.Depthm)

print('mean depthm:',mean_depthm)



mode_depthm = stats.mode(data.Depthm)

print('mode depthm:',mode_depthm)
median_salnty = np.median(data.Salnty)

print('median salnty:',median_salnty)



mean_salnty = np.mean(data.Salnty)

print('mean salnty:',mean_salnty)



mode_salnty = stats.mode(data.Salnty)

print('mode salnty:',mode_salnty)
#Depthm Range

print("Depthm Range: ", (np.max(data.Depthm)-np.min(data.Depthm)))
#Salnty Range 

print("Salnty Range: ", (np.max(data.Salnty)-np.min(data.Salnty)))
#Depthm Variance

print("Depthm Variance: ", (np.var(data.Depthm)))
#Salnty Variance

print("Salnty Variance: ", (np.var(data.Salnty)))
#Depthm Standard Deviation

print("Depthm Std: ", (np.std(data.Depthm)))
#Salnty Standard Deviation

print("Salnty Std: ", (np.std(data.Salnty)))
desc = data.Depthm.describe() 

print(desc)



Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR



print("Anything outside this range is an outlier: (","Lover Quartile:", lower_bound ,"| Upper Quartile:", upper_bound,")")



print("Outliers: ",data[(data.Depthm < lower_bound) | (data.Depthm > upper_bound)].Depthm.values)
desc = data.Salnty.describe() 

print(desc)



Q1 = desc[4]

Q3 = desc[6]

IQR = Q3-Q1

lower_bound = Q1 - 1.5*IQR

upper_bound = Q3 + 1.5*IQR



print("Anything outside this range is an outlier: (","Lover Quartile:", lower_bound ,"| Upper Quartile:", upper_bound,")")





print("Outliers: ",data[(data.Salnty < lower_bound) | (data.Salnty > upper_bound)].Salnty.values)
import pandas as pd

import seaborn as sns

import warnings



import matplotlib.pyplot as plt

plt.style.use("ggplot")

f,ax=plt.subplots(figsize = (28,28))





# corr() is actually pearson correlation

sns.heatmap(data.corr(),annot= True, linewidths=0.5, fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()

p1 = data.loc[:,["Depthm","Salnty"]].corr(method= "pearson")



print('Pearson Correlation: ')

print(p1)

sns.jointplot(data.Depthm,data.Salnty,kind="regg")

plt.show()
ranked_data = data.rank()

spearman_corr = ranked_data.loc[:,["Depthm","Salnty"]].corr(method= "pearson")

print("Spearman's correlation: ")

print(spearman_corr)