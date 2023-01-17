# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np   

from sklearn.linear_model import LinearRegression

import pandas as pd    

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns

from sklearn.model_selection import train_test_split  #(Sklearn package's randomized data splitting function)
df = pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")  

df.head()
df.shape
df = df.drop('car name', axis=1)



# Also replacing the categorical var with actual values



df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

df.head()
df = pd.get_dummies(df, columns=['origin'])

df.head()
# quick summary of data columns



df.describe()
# We can see horsepower is missing, cause it does not seem to be reqcognized as a numerical column!

# lets check the types of data



df.dtypes
# horsepower is showing as object type but as we see the data, it's a numeric value

# so it is possible that horsepower is missing some data in it

# lets check it by using 'isdigit()'. If the string is made of digits, it will store True else False

 

missing_value = pd.DataFrame(df.horsepower.str.isdigit())  



#print missing_value = False!



df[missing_value['horsepower'] == False]   # prints only those rows where hosepower is false
# Missing values have a'?''

# Replace missing values with NaN



df = df.replace('?', np.nan)

df[missing_value['horsepower'] == False] 
df.median()
median_fill = lambda x: x.fillna(x.median())

df = df.apply(median_fill,axis=0)



# converting the hp column from object / string type to float



df['horsepower'] = df['horsepower'].astype('float64')  

df_plot = df.iloc[:, 0:7]

sns.pairplot(df_plot, diag_kind='kde')   



# kde -> to plot density curve instead of histogram on the diag
# lets build our linear model



# independant variables

X = df.drop(['mpg','origin_europe'], axis=1)



# the dependent variable

y = df[['mpg']]



# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
regression_model = LinearRegression()

regression_model.fit(X_train, y_train)



# Here are the coefficients for each variable and the intercept



for idx, col_name in enumerate(X_train.columns):

    print(f"The coefficient for {col_name} is {regression_model.coef_[0][idx]}")
intercept = regression_model.intercept_[0]

print(f"The intercept for our model is {regression_model.intercept_}")
in_sampleScore = regression_model.score(X_train, y_train)

print(f'In-Sample score = {in_sampleScore}')



out_sampleScore = regression_model.score(X_test, y_test)

print(f'Out-Sample Score = {out_sampleScore}')
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



poly = PolynomialFeatures(degree=2, interaction_only=True)

X_train2 = poly.fit_transform(X_train)

X_test2 = poly.fit_transform(X_test)



poly_regr = linear_model.LinearRegression()



poly_regr.fit(X_train2, y_train)



y_pred = poly_regr.predict(X_test2)



#print(y_pred)



#In sample (training) R^2 will always improve with the number of variables!



print(poly_regr.score(X_train2, y_train))
# number of extra variables used in Polynomial Regression



print(X_train.shape)

print(X_train2.shape)