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
# importing required libraries
import seaborn as sns
sns.set()
sns.set(style="darkgrid")

import numpy as np
import pandas as pd

# importing matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize']=(10,10)
# read the dataset
df = pd.read_csv('/kaggle/input/data-analysis-and-visualization-of-indian-railways/Indian_railway1.csv')
# drop the null values
df = df.dropna(how="any")

# view the top results
df.head()
df.info()
sns.distplot(df['Distance'])
# scatter plot
sns.relplot(x="Station Name", y="Train Name", data=df[:200], kind="scatter");
sns.relplot(x="Distance", y="Train Name", data=df[:100], kind="scatter");
sns.relplot(x="Arrival time", y="Departure Time", data=df[:100], kind="scatter");
sns.relplot(x="Distance", y="Train Name", hue="Station Name",data=df[:100]);
x = df['Train Name']
y = df['Destination Station']

plt.plot(x, y)

plt.show()
df.apply(pd.value_counts).plot(kind='bar', subplots=True)
sns.relplot(x="Source Station", y="Destination Station", data=df[:100], kind="scatter");
sns.relplot(x="Source Station", y="Train Name", data=df[:100], kind="scatter");
sns.relplot(x="Train Name", y="Destination Station", data=df[:100], kind="scatter");
sns.relplot(x="Arrival time", y="Departure Time", data=df[:100], kind="scatter");
df.columns