# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/fifa19/data.csv')
data.head(10)
data.info()
data.corr()
f,ax = plt.subplots(figsize=(30,30))
sns.heatmap(data.corr(), annot=True, linewidths=5, fmt='.1f',ax=ax)
plt.show()

data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Age.plot(kind = 'line', color = 'g',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.SprintSpeed.plot(color = 'r',label = 'SprintSpeed',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Age',y = 'SprintSpeed',alpha = 0.5,color = 'r')
plt.xlabel('Age')
plt.ylabel('SprintSpeed')
plt.title('Age SprintSpeed Scatter Plot')
plt.show()
data.SprintSpeed.plot(kind='hist',bins = 50 ,figsize = (12,12))
plt.show()
x = data['Age']>41
data[x]
data[np.logical_and(data['Age']>41, data['SprintSpeed']<40)]