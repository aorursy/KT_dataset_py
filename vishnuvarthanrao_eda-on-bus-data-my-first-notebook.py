# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/bus-dataset/bus.csv')
df.dropna(inplace=True)
df.head(5)

types = df['Type'].unique()
types

types_dict = dict()
for item in types:
    types_dict[item]= (df['Type'].str.lower() == item.lower()).sum()
types_dict.pop('Single deck')
labels,num_data= list(types_dict.keys()),  list(types_dict.values())
x = np.arange(len(labels))
plt.bar(x,num_data)
plt.xticks(x,labels,rotation=90,fontSize=18)
plt.title("Type Distribution",fontSize=20)
plt.show()
temp = min(types_dict.values()) 
res = [key for key in types_dict if types_dict[key] == temp] 
res
countries = df['Country'].unique()
countries