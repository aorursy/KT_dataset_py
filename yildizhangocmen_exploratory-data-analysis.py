# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset= pd.read_csv("../input/mobile_phone_price.csv")
columns_name = dataset.columns
print(columns_name)
print(len(columns_name))
data_type = dataset.dtypes
print(data_type)
data_null = dataset.empty
if data_null == False :
    print("Veri seti içerisinde boş değer yoktur.") 
else :
    print ("Veri seti içerisinde boş değer vardır.")

    
    
dataset.info()
dataset_shape = dataset.shape
print(dataset_shape)
dataset_istatistik = dataset.describe()
print (dataset_istatistik)

for i in dataset.columns :
    
    print(dataset_istatistik[i][["mean","std"]])
    print("___________________________________")
import matplotlib.pyplot as plt

import seaborn as sns 

f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(dataset.corr(),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
dataset[['pc','fc']].corr()
dataset[['four_g','three_g']].corr()         