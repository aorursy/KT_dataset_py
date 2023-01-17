# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.describe()
df.isna().sum()
plt.style.use('ggplot')
p = df.hist(figsize=(10,10))
target = df['DEATH_EVENT']
df = df.drop('DEATH_EVENT', axis=1)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.99)

pca.fit(scaled_data)
reduced = pca.transform(scaled_data)
plt.figure(figsize=(10,6))

fig, ax =plt.subplots()
x = np.arange(1,12, step =1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(x, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of components')
plt.ylabel('Cumulative variance (%)')
plt.title('Number of components vs variance explained')

plt.axhline(y=0.95, color='r', linestyle ='-')
plt.text(1, 0.85, '95% cutoff')

ax.grid(axis='y')
plt.show()
