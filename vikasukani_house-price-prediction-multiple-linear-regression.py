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
# Import visualization packages

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



import warnings

warnings.filterwarnings('ignore')


#  Load data set 

df  = pd.read_csv('/kaggle/input/minihomeprices.csv')





# show first five rows

df.head()
# show information about data set



df.info()
# Descriptiion of our data set



df.describe().style.background_gradient(cmap='CMRmap')

#  to know how many null values



df.isna().sum()
# fill null values with median value



df['bedrooms'] = df['bedrooms'].fillna( df['bedrooms'].mean() )



#  here we can use inplace=True as well.  both are valid for update data frame



df.head()
# barploat show



plt.figure(figsize=(10, 7))

plt.title("Bedroom wise price increase.")



sns.barplot('bedrooms', 'price', data=df)

plt.xlabel('Bedrooms', )

plt.ylabel('Price')

plt.show()
# Scatter plot

plt.figure(figsize=(10, 5))



sns.scatterplot('bedrooms', 'price',data=df)

plt.title("Price vs Bedroom Scatter plot")



plt.xlabel("House Bedrooms")

plt.ylabel('House Price')

plt.show()
plt.figure(figsize=(10, 7))



sns.lmplot(x="bedrooms", y="price", data=df);

plt.title("Price and bedroom wise line plot")

plt.show()
# import model 

from sklearn.linear_model import LinearRegression
mdl = LinearRegression()
#  Set dependent and independent variables



X = df.drop(['price'], axis=1)

y = df['price']
# Change bedrooms data type flaot to int



df['bedrooms'] = df['bedrooms'].astype('int64')



df.info()
#  shows the variables

print(X)

print("-" * 25)

print(y)
#  Fitting Model



mdl.fit( X, y  )
# Now custimize prediction testing



mdl.predict([[ 4000, 2, 50 ]])

# show house price here
mdl.coef_
mdl.intercept_
#  know score 

score = mdl.score( X, y )



print(score * 100)