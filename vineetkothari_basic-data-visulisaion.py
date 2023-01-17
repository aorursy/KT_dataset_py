# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read a csv file pandas

df = pd.read_csv('../input/production.csv')

df.head()
# Number of variables with missing values

variables_missing_value = df.isnull().sum()

variables_missing_value
# Impute missing value of Loan_Amount_Term with median

df['Quantity 2011-12'].fillna(df['Quantity 2011-12'].median(), inplace=True)

variables_missing_value = df.isnull().sum()

variables_missing_value
# Impute missing value of LoanAmount with -9999

df['Value 2011-12'].fillna(df['Value 2011-12'].median(), inplace=True)

df['Value 2011-12'].isnull()
Quantiy_have_missing_value = df['Quantity 2011-12'].isnull().sum() > 0

Quantiy_have_missing_value
# Plot histogram for variable Quantitiy 2012-13

df['Quantity 2010-11'].head().hist()
# Treat Outliers of Quantity and Value

# Perform log transformation of Quantity to make it closer to normal

df['Quantity 10-11'] = np.log(df['Quantity 2010-11'])

df['Quantity 10-11'].head(20).hist(bins=20)
# Look at the summary of numerical variables for train data set

df.describe()
#quantity mean

quantity=[df['Quantity 2010-11'].mean(),

          df['Quantity 2011-12'].mean(),

          df['Quantity 2013-14'].mean(),

          df['Quantity 2014-15(P)'].mean()

          ]

#value mean

value=[df['Value 2010-11'].mean(),

       df['Value 2011-12'].mean(),

       df['Value 2013-14'].mean(),

       df['Value 2014-15(P)'].mean()

      ]

states=df['States']
#removing missing values

df.dropna().head()
# Print the unique values and their frequency of variable Quantity 2010-11

df1=df['Quantity 2010-11'].value_counts()

df1.head()
# Convert all non-numeric values to number

for var in df.columns:

    le = preprocessing.LabelEncoder()

    df[var]=le.fit_transform(df[var].astype('str'))

df[var].head()
# Plot a box plot for variable LoanAmount by variable Gender of training data set

df.boxplot(column='Quantity 2012-13')

#Understanding distribution of categorical variables

#values in absolute numbers

value_numbers = df['Quantity 2012-13'].value_counts()

value_numbers.head(10)
# Two-way comparison: Credit values and quantity

pd.crosstab(df ["Value 2012-13"], df ["Quantity 2012-13"], margins=True).head()
#finding labels for x and y

quantity_cols = [col for col in df.columns if 'Quantity' in col]

value_cols = [col for col in df.columns if 'Value' in col]

width=10

plt.bar(value,quantity, width, color="blue")

plt.ylabel('quantity')

plt.xlabel('value')

plt.xticks(value)

plt.show()
#lines_bars_and_markers example

N = len(value)



colors = np.random.rand(N)

area = np.pi * (30 * np.random.rand(N))**2  # 0 to 15 point radiuses

plt.ylabel('quantity')

plt.xlabel('value')

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(quantity,value)

plt.scatter(quantity,value, s=area, c=colors, alpha=0.5)

plt.show()