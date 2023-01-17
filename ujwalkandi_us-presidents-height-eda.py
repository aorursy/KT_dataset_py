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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/us-president-height-dataset/president_heights.csv")
df.head()
df.columns
plt.figure(figsize=(12,8))
sns.distplot( a=df["height(cm)"], hist=True, kde=False, rug=False)
plt.title("Height of US Presidents")
plt.xlabel("height(cm)")
plt.ylabel("Number of Presidents")
plt.show()
plt.figure(figsize=(12,8))
sns.regplot(x=df["order"], y=df["height(cm)"], marker="o", scatter_kws={"color":"blue","alpha":0.5,"s":100})
plt.show()
plt.figure(figsize=(12,8))
plt.plot('order', 'height(cm)', data=df, linestyle='-', marker='o')
plt.title("Height of US Presidents")
plt.xlabel("Presidents order")
plt.ylabel("height(cm)")
plt.show()
df.max()
df.min()