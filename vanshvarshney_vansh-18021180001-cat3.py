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
 # Q1. Read the dataset.

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

df=pd.read_csv('/kaggle/input/loan-data-set/loan_data_set.csv')

df
# Q2. Describe the dataset. 

df.describe(include = 'all')
# Q3(1). Find mean,median and mode of columns.

df.mean()
# Q3(2) Find mean,median and mode of columns. 

df.median()
# Q3(3). Find mean,median and mode of columns. 

df.mode()
# Q4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.

# Numeric Dataset and these features are normal

numeric = list(df._get_numeric_data().columns)

numeric
# Q4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.

# Ordinal columns

# columns which are not normal are ordinal

categorical = list(set(df.columns) - set(df._get_numeric_data().columns))

categorical
# Q5(2). Draw different types of graphs to represent information hidden in the dataset.

df.plot.hist()
# 5(3). Draw different types of graphs to represent information hidden in the dataset.



plt.plot(df.ApplicantIncome,label='ApplicantIncome')

plt.plot(df.CoapplicantIncome,label='CoapplicantIncome')

plt.plot(df.LoanAmount,label='Loan_Amount_Term')

plt.legend()
# Q6. Find columns which are correlated.

# Q7. Find columns which are not correlated.

df.corr()
# Q8(1). Compare different columns of dataset

comparison_column = np.where(df["ApplicantIncome"] == df["CoapplicantIncome"], True, False)

df["equal"] = comparison_column

df
# Q8(2). Compare different columns of dataset

comparison_column = np.where(df["LoanAmount"] == df["Loan_Amount_Term"], True, False)

df["equal"] = comparison_column

df