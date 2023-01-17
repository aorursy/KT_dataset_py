# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#TÜRKÇE KAYNAK



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # Matematiksel hesaplamalar için

import pandas as pd 

import matplotlib.pyplot as plt # Görselleştirme

import seaborn as sns # Görselleştirme

from collections import Counter

import plotly.plotly as py

import plotly.graph_objs as go

data=pd.read_csv("../input/BlackFriday.csv")

print(data.shape)

print("Müşteri Sayısı ",data['User_ID'].nunique())

print("Ürün Sayısı ",data['Product_ID'].nunique())

print("Alışveriş Adedi ",len(data))

#create copy of dataset

data_orig=data.copy()
#Gender Attribute

group_1 = data.groupby(['Gender'])

print (group_1[['Purchase']].count()/len(data)*100, end="\n\n\n\n")

# Pie chart for gender distribution

plt.subplot(2,2,1)

gender_count = [data_orig.Gender[data_orig['Gender']=='F'].count(),

                data_orig.Gender[data_orig['Gender']=='M'].count()]

gender_lab = data_orig.Gender.unique()

expl = (0.1,0)

plt.pie(gender_count, labels=gender_lab, explode=expl, shadow=True , autopct='%1.1f%%');
#Age Attribute

#How many age categories we have?

print("Age in Dataset:\n")

print("There are {} different values\n".format(len(data.Age.unique())))

print(data.Age.unique())

group_2 = data_orig.groupby(["Age"])

print (group_2[['Purchase']].count()/len(data_orig)*100, end="\n\n\n\n")

# Bar chart for Age

plt.subplot(2,2,2)

ordr =data_orig.groupby(["Age"]).count().sort_values(by='Purchase',ascending=False).index

sns.countplot(data_orig['Age'], label=True, order=ordr)
#City_Category distribution

group_3 = data_orig.groupby(["City_Category"])

print (group_3[['Purchase']].count()/len(data_orig)*100, end="\n\n\n\n")



#Occupation

#       How many occupation categories we have?

print("Occupation in Dataset:\n")

print("There are {} different values\n".format(len(data.Occupation.unique())))

print(data.Occupation.unique())

group_4 = data_orig.groupby(["Occupation"])

print (group_4[['Purchase']].count()/len(data_orig)*100, end="\n\n\n\n")

#       Bar chart for Occupation

plt.subplot(2,2,4)

ordr1 =data_orig.groupby(["Occupation"]).count().sort_values(by='Purchase',ascending=False).index

sns.countplot(y=data_orig['Occupation'], label=True, order=ordr1)

#Stay_In_Current_City_Years Attributes

group_5 = data_orig.groupby(["Stay_In_Current_City_Years"])

print (group_5[['Purchase']].count()/len(data_orig)*100, end="\n\n\n\n")

plt.figure(figsize=(8,6))

ordr2 =data_orig.groupby(["Stay_In_Current_City_Years"]).count().sort_values(by='Purchase',ascending=False).index

sns.countplot(data_orig['Stay_In_Current_City_Years'], label=True, order=ordr2)

plt.show()

#Gender * MaritalStatus Attributes   / Creating new column in the dataset 

data_orig['Gender_MaritalStatus'] = data_orig.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)

print(data_orig.Gender_MaritalStatus.unique())

group_5 = data_orig.groupby(["Gender_MaritalStatus"])

plt.figure(figsize=(15,10))

plt.subplot(1,2,1)

count1 = group_5[['Purchase']].count().values.tolist()

lab1 = data_orig.groupby(["Gender_MaritalStatus"]).count().index.values

expl2 = (0,0,0.1,0.1)

plt.pie(count1, labels=lab1,explode=expl2, shadow=True, autopct='%1.1f%%')

#Age*Gender*Marital_Status

plt.subplot(1,2,2)

sns.countplot(data_orig['Age'],hue=data_orig['Gender_MaritalStatus'])

plt.show()
#Missing values 

print("Are there any missing values on the original dataset ? {}".format(data.isnull().any().any()))

missings=data.isnull().sum()

print(data.isnull().sum()) #check missing values before removing  rows that missing value

data["Product_Category_2"] = data["Product_Category_2"].fillna(0)

data["Product_Category_3"] = data["Product_Category_3"].fillna(0)



print(data.isnull().sum())#check missing values after removing  rows that missing value
print(data.dtypes) # before converting int to object



data["User_ID"]=data["User_ID"].astype(object) #converting int to object

data["Occupation"]=data["Occupation"].astype(object)#converting int to object

data["Marital_Status"]=data["Marital_Status"].astype(object) #converting int to object

data["Product_Category_1"]=data["Product_Category_1"].astype(object) #converting int to object



print(data.dtypes) #after converting int to object
# Count per user

user_count = pd.DataFrame(data.groupby('User_ID')

                            ['Product_ID', 'Product_Category_1'].nunique())

user_count = user_count.sort_values('Product_ID', ascending=False)





# Sum per user

user_amount = pd.DataFrame(data.groupby('User_ID')

                            ['Purchase'].sum())

user_amount = user_amount.sort_values('Purchase', ascending=False)



df1 = pd.merge(user_amount, user_count, left_index=True, right_index=True)

df1.head()
#Datapreperation 

#drop User_ID and Product_ID columns

data = data.drop(columns="User_ID")

data = data.drop(columns="Product_ID")



#Changing to Int values

# Gender Attribute  / M=1 and F=0

def map_gender(gender):

    if gender == 'M':

        return 1

    else:

        return 0

data['Gender'] = data['Gender'].apply(map_gender)





# Age Attribute 0-17=0, 18-25=1, 26-35=2, 36-45=3, 46-50'=4, 51-55=5, others=6

def map_age(age):

    if age == '0-17':

        return 0

    elif age == '18-25':

        return 1

    elif age == '26-35':

        return 2

    elif age == '36-45':

        return 3

    elif age == '46-50':

        return 4

    elif age == '51-55':

        return 5

    else:

        return 6

data['Age'] = data['Age'].apply(map_age)





# City_Category Attribute/ A=2, B=1, C=0

def map_city_categories(city_category):

    if city_category == 'A':

        return 2

    elif city_category == 'B':

        return 1

    else:

        return 0

data['City_Category'] = data['City_Category'].apply(map_city_categories)





#Stay_In_Current_City_Years Attribute / 4+= 4, 3=3,...

def map_stay(stay):

        if stay == '4+':

            return 4

        else:

            return int(stay)

data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].apply(map_stay) 



X = data.drop(["Purchase"], axis=1)

from sklearn.preprocessing import LabelEncoder#Now let's import encoder from sklearn library

LE = LabelEncoder()

X = X.apply(LE.fit_transform)#Here we applied encoder onto data

X.Gender = pd.to_numeric(X.Gender)

X.Age = pd.to_numeric(X.Age)

X.Occupation = pd.to_numeric(X.Occupation)

X.City_Category = pd.to_numeric(X.City_Category)

X.Stay_In_Current_City_Years = pd.to_numeric(X.Stay_In_Current_City_Years)

X.Marital_Status = pd.to_numeric(X.Marital_Status)

X.Product_Category_1 = pd.to_numeric(X.Product_Category_1)

X.Product_Category_2 = pd.to_numeric(X.Product_Category_2)

X.Product_Category_3 = pd.to_numeric(X.Product_Category_3)
Y = data["Purchase"]

from sklearn.preprocessing import StandardScaler

SS = StandardScaler()

Xs = SS.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xs,Y,test_size=0.33)



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor





lr = LinearRegression()

dtr = DecisionTreeRegressor()

rfr = RandomForestRegressor()

gbr = GradientBoostingRegressor()



fit1 = lr.fit(X_train,y_train)#Here we fit training data to linear regressor

fit2 = dtr.fit(X_train,y_train)#Here we fit training data to Decision Tree Regressor

fit3 = rfr.fit(X_train,y_train)#Here we fit training data to Random Forest Regressor

fit4 = gbr.fit(X_train,y_train)#Here we fit training data to Gradient Boosting Regressor



print("Accuracy Score of Linear regression on test set",fit1.score(X_test,y_test)*100)

print("Accuracy Score of Decision Tree on test set",fit2.score(X_test,y_test)*100)

print("Accuracy Score of Random Forests on test set",fit3.score(X_test,y_test)*100)

print("Accuracy Score of Gradient Boosting on testset",fit4.score(X_test,y_test)*100)



#RMSE

#1-LinearRegression RMSE

tahmin_LR= lr.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse_LR=np.sqrt(mean_squared_error(y_test, tahmin_LR))

print("Linear Regression RMSE")

print(rmse_LR)



#2-Decision Tree Regression RMSE

tahmin_DTR= dtr.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse_DTR=np.sqrt(mean_squared_error(y_test, tahmin_DTR))

print("Decision Tree Regression RMSE")

print(rmse_DTR)



#3-Random Forest Regression RMSE

tahmin_RF= rfr.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse_RF=np.sqrt(mean_squared_error(y_test, tahmin_RF))

print("Random Forest Regression RMSE")

print(rmse_RF)



#4-Gradient Boosting Regression RMSE

tahmin_GB= gbr.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse_GB=np.sqrt(mean_squared_error(y_test, tahmin_GB))

print("Gradient Boosting Regression RMSE")

print(rmse_GB)