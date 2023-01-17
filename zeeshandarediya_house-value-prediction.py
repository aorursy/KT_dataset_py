import pandas as pd
import numpy as np
data = pd.read_csv('../input/kc-house-data/kc_house_data.csv',engine='python')

data
# Here we have year of built as well as year of renovation. So I am considering if the house is renovated then it will be the effective year.
# Also there are various values of sqft. So I am combining all of them into just one value.

p1 = data['yr_built'].to_numpy()
p2 = data['yr_renovated'].to_numpy()
for i in range(0,21613):
    if(p2[i]>p1[i]):
        p1[i]=p2[i]
p1
data['yr_effective'] = p1
data
data['sqft_total'] = data['sqft_living'] + data['sqft_lot'] + data['sqft_above'] + data['sqft_basement'] + data['sqft_living15'] + data['sqft_lot15']
data['bathrooms'] = data['bathrooms'].astype(int)
data
#Lets split out only required data
dataSub = data[['price','bedrooms','bathrooms','floors','waterfront','view','sqft_total','yr_effective','lat','long']]
dataSub.head(5)
dataSub.describe()
dataSub.isnull().sum()
dataSub.dtypes
from sklearn.model_selection import train_test_split
# We will use DecisionTree here rather than Linear Regression
from sklearn.tree import DecisionTreeRegressor
x = dataSub.iloc[:,1:10]
x
import seaborn as sns

correlation = dataSub.corr()
sns.heatmap (correlation,annot=True)
y = dataSub['price']
y
#lets split into train and test now.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=30)
reg = DecisionTreeRegressor()
#Lets fit the model now
reg.fit(x_train,y_train)
pred = reg.predict(x_test)
pred
y_test
# Above results are almost similar , example:
# pred value = 379950 , actual = 384950
# pred value = 290000 , actual = 302400
# pred value = 550000 , actual = 557000
#Lets plot and see

ax1 = sns.distplot(y_test, hist=False, color='r', label='Actual')
sns.distplot(pred, hist=False, color='b', label='pred', ax=ax1)
#lets manually enter the values of entry #2 and check
x = [['3','2','2','0','0','21711','1991','47.7210','-122.319']]
reg.predict(x)
#and the actual value in table was 538000
# I feel I have done a good work on this data
# Feel free to give your reviews on it
# Please let me know if I had done any mistake or how I could make it better
# I am attaching the data along with the post 
# Do try this dataset, this is interesting 
