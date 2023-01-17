#defining the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
#Both mean squared error (MSE) and mean absolute error (MAE) are used in predictive modeling. ... Therefore, MAE is more robust to outliers since it does not make use of square. On the other hand, MSE is more useful if we are concerned about large errors whose consequences are much bigger than equivalent smaller ones.
#read the data file and take the look of the data
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
print('the shape of data is: ', df.shape)
#check if there are any null values in the data
df.isnull().sum()
#dropping null values
df.dropna(inplace=True)
#converting categories to discrete variables
df['Platform'] = df['Platform'].astype('category')
df['Platform'] = df['Platform'].cat.codes

df['Genre'] = df['Genre'].astype('category')
df['Genre'] = df['Genre'].cat.codes

df['Publisher'] = df['Publisher'].astype('category')
df['Publisher'] = df['Publisher'].cat.codes
#to check the distribution of numeric data
plt.figure(figsize=(15,20))
plt.subplot(4,3,1)
sns.distplot(df['Global_Sales'])
sns.distplot(df['JP_Sales'])
plt.subplot(4,3,2)
sns.distplot(df['EU_Sales'])
plt.subplot(4,3,3)
sns.distplot(df['NA_Sales'])
sns.distplot(df['Other_Sales'])
#to check the outliers of numeric data
plt.figure(figsize=(15,20))
plt.subplot(4,1,1)
sns.boxplot(df['Global_Sales'])
sns.boxplot(df['JP_Sales'])
plt.subplot(4,1,2)
sns.boxplot(df['EU_Sales'])
plt.subplot(4,1,3)
sns.boxplot(df['NA_Sales'])
plt.subplot(4,1,4)
sns.boxplot(df['Other_Sales'])
#to check the strength of relationship among variables
plt.figure(figsize=(15,10)) #manage the size of the plot
sns.heatmap(df.corr(),annot=True, square = True) 
plt.show()
df.columns
#to check the distribution of numeric data
plt.figure(figsize=(15,20))
plt.subplot(4,3,1)
sns.scatterplot(df['Global_Sales'], df['JP_Sales'])
plt.subplot(4,3,2)
sns.scatterplot(df['Global_Sales'],df['EU_Sales'])
plt.subplot(4,3,3)
sns.scatterplot(df['Global_Sales'],df['NA_Sales'])
plt.subplot(4,3,4)
sns.scatterplot(df['Global_Sales'], df['Other_Sales'])
plt.figure(figsize=(15,20))
plt.subplot(4,1,1)
sns.barplot(x = 'Platform', y ='Global_Sales', data = df)
plt.subplot(4,1,2)
sns.barplot(x = 'Genre', y ='Global_Sales', data = df)
plt.subplot(4,1,3)
sns.barplot(x = 'Year', y ='Global_Sales', data = df)
plt.xticks(rotation=90, ha='right')
df_new = df[:][0:100]
plt.figure(figsize=(25,8))
sns.barplot(x = df_new['Name'], y = df_new['Global_Sales'])
plt.xticks(rotation=90, ha='right');
#dropping the name column since it if of no use in sales analysis
df = df.drop('Name', axis =1)
plt.figure(figsize=(25,8))
sns.barplot(x = df_new['Name'], y = df_new['JP_Sales'])
plt.xticks(rotation=90, ha='right');
plt.figure(figsize=(25,8))
sns.barplot(x = df_new['Name'], y = df_new['EU_Sales'])
plt.xticks(rotation=90, ha='right');
plt.figure(figsize=(25,8))
sns.barplot(x = df_new['Name'], y = df_new['NA_Sales'])
plt.xticks(rotation=90, ha='right');
plt.figure(figsize=(25,8))
sns.barplot(x = df_new['Name'], y = df_new['Other_Sales'])
plt.xticks(rotation=90, ha='right');
df =df.drop('Year', axis =1)
X= df.copy()
X = X.drop('Global_Sales', axis =1)
y = df['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =100)
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred =model_linear.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('\033[32mMean Absolute Error: ', mae)
model_DT = DecisionTreeRegressor()
model_DT.fit(X_train, y_train)
y_pred =model_DT.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('\033[32mMean Absolute Error: ', mae)
X_train.columns
X_train = X_train.drop(columns=['Platform', 'Genre', 'Publisher'], axis =1)
X_test = X_test.drop(columns=['Platform', 'Genre', 'Publisher'], axis =1)
model_lin = LinearRegression()
model_lin.fit(X_train, y_train)
y_pred =model_lin.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('\033[32mMean Absolute Error: ', mae)
model_tree = DecisionTreeRegressor()
model_tree.fit(X_train, y_train)
y_pred =model_tree.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('\033[32mMean Absolute Error: ', mae)