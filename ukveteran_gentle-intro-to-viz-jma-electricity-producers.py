import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cost-function-for-electricity-producers/Electricity.csv')

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df.info()
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(8)
df.dtypes.value_counts()
corrs = df.corr()

corrs
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize = (20, 8))

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
sns.distplot(df["cost"])
sns.scatterplot(x='cost',y='q',data=df)
sns.countplot(df["q"])
ax = sns.swarmplot(x="cost", y="q", data=df)
print ("Skew is:", df.cost.skew())

plt.hist(df.cost, color='pink')

plt.show()
target = np.log(df.cost)

print ("Skew is:", target.skew())

plt.hist(target, color='green')

plt.show()