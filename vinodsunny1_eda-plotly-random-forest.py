from IPython.display import Image

Image('../input/firstimage/1.jpg')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
car = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

car.columns = [i.lower() for i in car.columns]

print("Number of rows : ",car.shape[0])

print("Number of columns : ",car.shape[1])





# drop duplicates rows if present

car.drop_duplicates(keep = 'first',inplace = True)



# resetting index 

car.reset_index(inplace = True,drop = True)
print("Numerical feature : ")

print(car.select_dtypes(include = "number").columns)

print("\n Categorical feature : ")

print(car.select_dtypes(exclude = "number").columns)
# info of the dataset ,

car.info()
# description of data 

# luckily we don't have any missing value's in our dataset . 

car.describe()
car[car.present_price == car.present_price.max()][['car_name','present_price']]
px.pie(data_frame = car ,names = car.car_name.value_counts().head(10).index,values = car.car_name.value_counts().values[0:10],

      title = "Top 10 vehicle company is more for sale",hole = 0.7

      )
plt.rcParams['figure.figsize'] = (15,6)

sns.countplot(car['year'])

plt.title("year v/s vehicle's availabel for second's ")
car[["selling_price","present_price"]].plot(kind = 'line')

plt.title("Comparsion of  Present price v/s Selling price ")

plt.ylabel("Price")
px.pie(data_frame = car ,names = car.fuel_type.value_counts().index,values = car.fuel_type.value_counts().values,

      title = "vehicle fuel type " ,hole = 0.7

      )
px.pie(data_frame = car ,names = car.transmission.value_counts().index,values = car.transmission.value_counts().values,

      title = "vehicle transmission type " ,hole = 0.7

      )
px.scatter(data_frame = car,y = car['selling_price'],x =  car['kms_driven'],title = 'sp v/s kms_driven')

# higher the kms driven lesser the selling price of vehicle would be . 
# Most of the vehicle owner are trying to sell their vehicle with the help of  dealer  

plt.rcParams['figure.figsize'] = (8,4)

sns.countplot(car['seller_type'])
# let's create one new feature .

car['cur_year'] = 2020

car.head(3)
car['year'].max()
# let's create one more new feature called . 

car['years_used'] = car['cur_year'] - car['year']



# drop cur_year and year feature from the dataset .

car.drop(['cur_year','year'],axis = 1,inplace = True)

car.head(3)
px.scatter(data_frame = car,x = car['years_used'],y =  car['selling_price'],title = 'years_used v/s sp',color = 'years_used')

# higher the years_used lesser the selling price of vehicle would be . 
px.bar(x = car['years_used'].value_counts().index , y =car['years_used'].value_counts().values,

       color = car['years_used'].value_counts().index,title = 'years used',labels = {"x":'years used',"y":"count"})
new_df = pd.pivot_table(car, index = ['years_used'],values = 'kms_driven')

px.pie(names = new_df.index ,values = new_df.values,title = 'years used v/s kms_driven' )

# Higher the years used higher the kms_driven will be .
# we can see the linear relationship between sp v/s pp .

sns.scatterplot(x = car['selling_price'],y =  car['present_price'])

plt.title('selling_price v/s present_price')
sns.boxplot(car['present_price'])
q1 = car['present_price'].quantile(0.25)

q3 = car['present_price'].quantile(0.75)

iqr = q3 - q1

lb = q1 - iqr*1.5

ub = q3 + iqr*1.5

lb,ub



# -11 is meaningless, price can't be negative .



# 22.8 only few  of the vehicle present price is greater than 20 lakhs  .
print("Vehicle  present prices which are acting has outlier :")

car[car['present_price'] >= 22.8 ][['car_name','present_price','selling_price']]
sns.pairplot(car)
plt.figure(figsize=(15,8))

sns.heatmap(car.corr(),annot = True)
# let's drop unwanted features 

car.drop(['car_name','owner'],axis = 1,inplace = True)

car.head(2)
# convert categorical variable into numerical variable . 

car = pd.get_dummies(car,drop_first = True)

car.head(2)
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split,RandomizedSearchCV
x = car.drop('selling_price',axis = 1)

y = car['selling_price']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 33)

print("Training data : ",xtrain.shape, ytrain.shape)

print("Testing data  : ",xtest.shape, ytest.shape)
rf = RandomForestRegressor()
param_grid = {

     

    'n_estimators' : [50,100,150,250,300,350],

     'max_depth'   : [i for i in range(7,25)] , 

     'min_samples_split' : [i for i  in range(5,25)],

     'min_samples_leaf' : [2,3,4,7,10]

}
# using 10 fold

rf = RandomizedSearchCV(estimator = rf,param_distributions = param_grid,n_jobs = 1,cv = 10)
rf.fit(xtrain,ytrain)
print("best parameters : ")

rf.best_params_
# make prediction 

pred = rf.predict(xtest)
print("MSE value is : ",mean_squared_error(ytest,pred))

print("r2  value is : ",r2_score(ytest,pred))

r2 = r2_score(ytest,pred)

n = len(xtest)

k = xtest.shape[1]

adj_r2_score = 1 - (((1- r2)*(n-1)) / (n - k - 1))

print("adj_r2_score  value is : ",adj_r2_score)
plt.scatter(ytest,pred)

plt.xlabel('actual')

plt.ylabel('predict')
# Although this is simple analysis ,the dataset is too small so we cannot explore further .
