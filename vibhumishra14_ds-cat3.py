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
# Group number= 2

# Group members

# Vibhu Mishra 18SCSE1010052

# DUSHYANT CHAUDHARY 18SCSE1010047

# SHIVAM SETHI 18SCSE1010043

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

df=pd.read_csv('/kaggle/input/student-performance-data-set/student-por.csv')

df
# 2. Describe the dataset. 

df.describe(include = 'all')
# 3. Find mean,median and mode of columns.

df.mean()
# 3. Find mean,median and mode of columns. 

df.median()
# 3. Find mean,median and mode of columns. 

df.mode ()
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.

# Numeric Dataset and these features are normal

numeric = list(df._get_numeric_data().columns)

numeric
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.

# Ordinal columns

# columns which are not normal are ordinal

categorical = list(set(df.columns) - set(df._get_numeric_data().columns))

categorical
# 5. Draw different types of graphs to represent information hidden in the dataset.

plt.rcParams['figure.figsize']=(10,6)
# 5. Draw different types of graphs to represent information hidden in the dataset.

df.plot.hist()
# 5. Draw different types of graphs to represent information hidden in the dataset.

df.plot.bar()
# 5. Draw different types of graphs to represent information hidden in the dataset.

plt.plot(df.age,label='age')

plt.plot(df.sex,label='sex')

plt.plot(df.Medu,label='Medu')

plt.legend()
# 5. Draw different types of graphs to represent information hidden in the dataset.

plt.hist(df.age)

plt.xlabel('age')

plt.ylabel('total number of student')
# 5. Draw different types of graphs to represent information hidden in the dataset.

female=df.loc[df['sex']=='F'].count()[0]

male=df.loc[df['sex']=='M'].count()[0]

labels=['Female','Male']

plt.pie([female,male],labels=labels,autopct='%.2f%%')
# 6. Find columns which are correlated.

# 7. Find columns which are not correlated.

df.corr()
# 8. Compare different columns of dataset

comparison_column = np.where(df["G1"] == df["G2"], True, False)

df["equal"] = comparison_column

df