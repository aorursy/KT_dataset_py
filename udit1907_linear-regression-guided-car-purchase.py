# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



#core imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#load dataset and assign it to a variable

vehicles=pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
#use the 'head' method to show the first five rows of the table as well as their names. 

vehicles.head() 
vehicles.info()
vehicles.isnull().sum()
vehicles.describe()
sns.barplot(x='Owner',y='Selling_Price',data=vehicles,palette='spring')
sns.barplot(x='Transmission',y='Selling_Price',data=vehicles,palette='spring')
sns.barplot(x='Fuel_Type',y='Selling_Price',data=vehicles,palette='spring')
sns.barplot(x='Seller_Type',y='Selling_Price',data=vehicles,palette='spring')
plt.figure(figsize=(10,10))

sns.lmplot(x='Kms_Driven',y='Selling_Price',data=vehicles)
plt.figure(figsize=(10,10))

sns.lmplot(x='Present_Price',y='Selling_Price',data=vehicles)
#creating a new column 'Vehicle_Age' and storing the age of the vehicles to establish a direct relationship between the age and selling price

vehicles['Vehicle_Age']=2020- vehicles['Year']



#check out the newly added column

vehicles.head(10)
plt.figure(figsize=(10,10))

sns.regplot(x='Vehicle_Age',y='Selling_Price',data=vehicles)
#using Pandas' built in function 'get_dummies()' to swiftly map the categorical values to integers like (0/1/2/3....)

vehicles=pd.get_dummies(vehicles,columns=['Fuel_Type','Transmission','Seller_Type'],drop_first=True)



#dropping the Year column since it becomes redundant and irrelevant after Vehicle_Age column

vehicles.drop(columns=['Year'],inplace=True)



#check out the dataset with new changes

vehicles.head()
sns.pairplot(vehicles)
correlations = vehicles.corr()



indx=correlations.index

plt.figure(figsize=(26,22))

sns.heatmap(vehicles[indx].corr(),annot=True,cmap="YlGnBu")

# We're splitting up our data set into groups called 'train' and 'test'

from sklearn.model_selection import train_test_split



np.random.seed(0)

vehicles_train,vehicles_test = train_test_split(vehicles, test_size=0.3, random_state=100)



# We'll perform feature scaling to ensure normalization of the data within a particular range.

#Sometimes, it also helps in speeding up the calculations in an algorithm.

from sklearn.preprocessing import StandardScaler



scaler= StandardScaler()



#features we need to scale are assigned as a list.

var=['Selling_Price','Present_Price','Kms_Driven','Vehicle_Age']



#scaling the training data(fitting the parameters and transforming the values)

vehicles_train[var]=scaler.fit_transform(vehicles_train[var])



#transforming the test data.We avoid fitting the values to prevent data leakage!

vehicles_test[var]=scaler.transform(vehicles_test[var])



#We will toss out the Car_Name column from training and test data because it only has text info that the linear regression model can't use!



X_test=vehicles_test.drop(columns=['Car_Name','Selling_Price'],axis=1)

y_test=vehicles_test['Selling_Price']



X_train=vehicles_train.drop(columns=['Car_Name','Selling_Price'],axis=1)

y_train=vehicles_train['Selling_Price']
from sklearn.linear_model import LinearRegression



lm=LinearRegression()



lm.fit(X_train,y_train)
# print the intercept of best-fit line

print(lm.intercept_)
# temp here stores the numerical columns from the vehicles dataset that influence the prediction

temp=vehicles.drop(columns=['Car_Name','Selling_Price'])



coeff_df = pd.DataFrame(lm.coef_,temp.columns,columns=['Coefficient'])

coeff_df 
predictions=lm.predict(X_test)





fig = plt.figure()

# Plot-label

fig.suptitle('y_test vs predictions')



#X-label

plt.xlabel('y_test')



# Y-label

plt.ylabel('predcitions')

plt.scatter(y_test,predictions)
fig=plt.figure(figsize=(8,8))

  

sns.distplot((y_test-predictions),bins=20)



#Plot Label

fig.suptitle('Residual Analysis', fontsize = 20)           
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
R2 = metrics.r2_score(y_test,predictions)

R2