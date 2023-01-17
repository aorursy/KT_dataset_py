# Basic libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Plot related libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Linear Regression Model
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils import shuffle
TRAIN_FILE = "/kaggle/input/E-Commerce_Participants_Data/Train.csv"
TEST_FILE = "/kaggle/input/E-Commerce_Participants_Data/Test.csv"

# Using pandas read_csv method to import data
train_ecomm_df = pd.read_csv(TRAIN_FILE, header=0)
test_ecomm_df = pd.read_csv(TEST_FILE, header=0)
train_ecomm_df.info()
print("=="*30)
train_ecomm_df.head()
test_ecomm_df.info()
print("=="*30)
test_ecomm_df.head()
train_ecomm_df.describe().T
train_ecomm_df.columns
sns.set(style='whitegrid', palette='muted')
fig, ax = plt.subplots(1,2, figsize=(12,6))

sns.distplot(train_ecomm_df['Selling_Price'], kde=True, ax=ax[0])
sns.scatterplot(x='Item_Rating', y='Selling_Price', data=train_ecomm_df, marker='o', color='r', ax=ax[1])

plt.tight_layout()
plt.show()
# Transform the target variable
y_target = np.log1p(train_ecomm_df['Selling_Price'])
fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(train_ecomm_df['Selling_Price'], kde=True, ax=axes[0])
sns.distplot(y_target, kde=True, ax=axes[1])
axes[0].set_title("Skewed Y-Values")
axes[1].set_title("Normalized Y-Values")
plt.show()
# Merge train and test data
tempset = pd.concat([train_ecomm_df, test_ecomm_df], keys=[0,1])

# Impute the 'unknown' values with Mode
tempset['Subcategory_1'] = tempset['Subcategory_2'].replace('unknown', np.nan).bfill().ffill()
tempset['Subcategory_2'] = tempset['Subcategory_2'].replace('unknown', np.nan).bfill().ffill()

tempset['Subcategory_1'] = tempset['Subcategory_1'].fillna(tempset['Subcategory_1'].mode()[0])
tempset['Subcategory_2'] = tempset['Subcategory_2'].fillna(tempset['Subcategory_2'].mode()[0])
tempset.drop(['Date', 'Product'], axis=1, inplace=True)
# Getting the categorical columns
cat_data = tempset.select_dtypes(include=['object'])

# One-hot encoding
X_encode = pd.get_dummies(tempset, columns=cat_data.columns)

# Getting back the Tran and Test data
X_train, X_enc_test = X_encode.xs(0), X_encode.xs(1)
# Prepare X and y for fitting the model
y = X_train['Selling_Price'].values
X = X_train.drop('Selling_Price', axis=1).values

X_test = X_enc_test.drop('Selling_Price', axis=1)
ridge_cv = RidgeCV(normalize=True,cv=10,gcv_mode='svd',scoring='neg_mean_squared_error')

#Initializing Linear Regression algorithm with Ridge regularizer(K-fold with 10 folds)
ridge_reg = TransformedTargetRegressor(regressor= ridge_cv,
                                      func=np.log1p,
                                      inverse_func=np.expm1)
ridge_reg.fit(X, y)

# Predict the test data
predictions = ridge_reg.predict(X_test)
final_df = pd.DataFrame({'Selling_Price': predictions})

final_df['Selling_Price'] = final_df.apply(lambda x: round(x, 2))
final_df = pd.concat([test_ecomm_df, final_df['Selling_Price']], axis=1)
final_df.head(20)
