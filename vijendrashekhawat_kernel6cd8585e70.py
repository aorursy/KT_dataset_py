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
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/memory-test-on-drugged-islanders-data/Islander_data.csv")
from sklearn.model_selection import train_test_split
data.head(4)
import matplotlib.pyplot as plt
data.describe()
plt.scatter(data[data.columns['age']],data[data.columns['Diff']])
x = data['age']
y = data['Diff']
z = data['Dosage']
plt.scatter(x,y,c='red')
plt.xlabel("age")
plt.ylabel("difference")
plt.title("age vs diff",c="green")
plt.scatter(z,y)
plt.scatter(x,data['Mem_Score_Before'])
plt.scatter(x,y/(max(y)-min(y)),c='red')
hs = data['Happy_Sad_group']
hs[:4]
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
hsp = lb.fit_transform(hs)
plt.scatter(y,hsp)
plt.scatter(x,hsp)
