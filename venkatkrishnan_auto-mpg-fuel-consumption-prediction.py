# Basic library imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
auto_data = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv', na_values='?')
auto_data.info()
print("--"*40)
auto_data.head().append(auto_data.tail())
auto_data.isnull().sum()

# this can be visualized using heatmap
sns.set(style='whitegrid', palette='muted', color_codes=True)

plt.figure(figsize=(10,6))
sns.heatmap(auto_data.isnull(), cbar=False)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10,7))
sns.boxplot(auto_data['horsepower'])
plt.show()
auto_data['horsepower'] = auto_data['horsepower'].fillna(auto_data['horsepower'].median())
# Checking the categorical attributes values
print(auto_data['origin'].value_counts())
print("--"*40)

cylinder_dist = auto_data['cylinders'].value_counts() / len(auto_data)
print(cylinder_dist)
