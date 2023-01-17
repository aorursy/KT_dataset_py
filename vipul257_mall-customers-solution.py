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
import pandas as pd
df_mall = pd.read_csv('../input/mall-customers/Mall_Customers.csv')
df_mall.head()
df = df_mall[['Annual Income (k$)','Spending Score (1-100)']]
df.head()
from sklearn.cluster import KMeans
loss = []
for i in range(1,10):
    km = KMeans(n_clusters=i)
    km.fit(df)

    loss.append(km.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1,10),loss)
plt.show()
df['Spending Score (1-100)'].min()
import numpy as np
cent = {1: [np.random.randint(15,135),np.random.randint(1,99)],
        2: [np.random.randint(15,135),np.random.randint(1,99)],
        3: [np.random.randint(15,135),np.random.randint(1,99)],
        4: [np.random.randint(15,135),np.random.randint(1,99)],
        5: [np.random.randint(15,135),np.random.randint(1,99)]}
cent
df.columns
def assign():
    for i in range(1,6):
        df[str(i)] = ((df['Annual Income (k$)'] - cent[i][0])**2 + (df['Spending Score (1-100)'] - cent[i][1])**2)**0.5  
    df['nearest'] = df.loc[:,'1':'5'].idxmin(axis=1)
def update():
    for i in range(1,6):
        cent[i][0] = df[df['nearest'] == str(i)]['Annual Income (k$)'].mean()
        cent[i][1] = df[df['nearest'] == str(i)]['Spending Score (1-100)'].mean()
assign()
for i in range(0,500):
    update()
    assign()
df['c'] = df['nearest'].map({'1':'r','2':'g','3':'b','4':'brown','5':'yellow'})
plt.scatter(df['Annual Income (k$)'] , df['Spending Score (1-100)'] , c=df['c'])
plt.show()
df.head(15)
