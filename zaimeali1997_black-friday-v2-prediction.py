import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/black-friday/train.csv')

data.head()
print("Number of Rows: ", data.shape[0])
data.describe()
data.info()
print("Missing Values in Each Column:")

print(data.isna().sum())
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.boxplot(data.Product_Category_2, ax=axes[0])

data.Product_Category_2.plot(kind='box', ax=axes[1])

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)

sns.boxplot(data.Product_Category_3, ax=axes[0])

data.Product_Category_3.plot(kind='box', ax=axes[1])

plt.show()
data.fillna(method='ffill', inplace=True)

data.isna().sum()
data.fillna(method='bfill', inplace=True)

data.isna().sum()
print("Now lets see how's our data looking: ")

data.head(15)
data.info()
data.nunique()
col = list(data.columns)

print("Unique Values in each column:\n")

for c in col:

    print(c, ": ", data[c].unique())

    print()
data['Gender'] = data.Gender.map({

    'M' : 0,

    'F' : 1

})

data.head()
from sklearn.preprocessing import LabelEncoder

lE = LabelEncoder()

data.City_Category = lE.fit_transform(data.City_Category)

data.head()
data.loc[data['Stay_In_Current_City_Years'] == '4+','Stay_In_Current_City_Years'] = '4'

data.Stay_In_Current_City_Years = data.Stay_In_Current_City_Years.astype('int64')

data.info()
data.groupby(['Age', 'Marital_Status', 'Gender']).count()
print("Let see which user pays the maximum price for a product and for which Product:")

data.loc[data.Purchase.idxmax()]
data.head()
data.info()
data.Age = lE.fit_transform(data.Age)

data.head()
data.head(15)
data.Age.unique()
cols = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

corr_result = data[cols].corr()

corr_result
sns.heatmap(corr_result, annot=True)

plt.show()
var1 = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

var2 = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status', 'Purchase']

var3 = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

var4 = ['Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Purchase']

var5 = ['Age', 'Occupation', 'City_Category', 'Purchase']

var6 = ['City_Category', 'Purchase']

var7 = ['Age', 'Purchase']
df1 = data[var1]

df2 = data[var2]

df3 = data[var3]

df4 = data[var4]

df5 = data[var5]

df6 = data[var6]

df7 = data[var7]
X1 = df1.iloc[:, :-1]

y1 = df1.iloc[:, -1]



X2 = df2.iloc[:, :-1]

y2 = df2.iloc[:, -1]



X3 = df3.iloc[:, :-1]

y3 = df3.iloc[:, -1]



X4 = df4.iloc[:, :-1]

y4 = df4.iloc[:, -1]



X5 = df5.iloc[:, :-1]

y5 = df5.iloc[:, -1]



X6 = df6.iloc[:, :-1]

y6 = df6.iloc[:, -1]



X7 = df7.iloc[:, :-1]

y7 = df7.iloc[:, -1]
from sklearn.model_selection import train_test_split

train_X1, test_X1, train_y1, test_y1 = train_test_split(X1, y1, shuffle=False, test_size=0.2)

train_X2, test_X2, train_y2, test_y2 = train_test_split(X2, y2, shuffle=False, test_size=0.2)

train_X3, test_X3, train_y3, test_y3 = train_test_split(X3, y3, shuffle=False, test_size=0.2)

train_X4, test_X4, train_y4, test_y4 = train_test_split(X4, y4, shuffle=False, test_size=0.2)

train_X5, test_X5, train_y5, test_y5 = train_test_split(X5, y5, shuffle=False, test_size=0.2)

train_X6, test_X6, train_y6, test_y6 = train_test_split(X6, y6, shuffle=False, test_size=0.2)

train_X7, test_X7, train_y7, test_y7 = train_test_split(X7, y7, shuffle=False, test_size=0.2)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

dtr = DecisionTreeRegressor()

rfr = RandomForestRegressor()
def custom_model(train_X, test_X, train_y, test_y, model, model_name):

    model.fit(train_X, train_y)

    pred_y = model.predict(test_X)

    print("Accuracy of "+ model_name+" is: ", model.score(test_X, test_y))

    print()
custom_model(train_X1, test_X1, train_y1, test_y1, dtr, "Decision Tree")

custom_model(train_X2, test_X2, train_y2, test_y2, dtr, "Decision Tree")

custom_model(train_X3, test_X3, train_y3, test_y3, dtr, "Decision Tree")

custom_model(train_X4, test_X4, train_y4, test_y4, dtr, "Decision Tree")

custom_model(train_X5, test_X5, train_y5, test_y5, dtr, "Decision Tree")

custom_model(train_X6, test_X6, train_y6, test_y6, dtr, "Decision Tree")

custom_model(train_X7, test_X7, train_y7, test_y7, dtr, "Decision Tree")
custom_model(train_X1, test_X1, train_y1, test_y1, rfr, "Random Forest")
custom_model(train_X2, test_X2, train_y2, test_y2, rfr, "Random Forest")

custom_model(train_X3, test_X3, train_y3, test_y3, rfr, "Random Forest")
custom_model(train_X4, test_X4, train_y4, test_y4, rfr, "Random Forest")

custom_model(train_X5, test_X5, train_y5, test_y5, rfr, "Random Forest")
custom_model(train_X6, test_X6, train_y6, test_y6, rfr, "Random Forest")

custom_model(train_X7, test_X7, train_y7, test_y7, rfr, "Random Forest")