### Import some usefull libraries==>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# Read and Understand the data
car_data=pd.read_csv("../input/predict-car-selling-price/car data.csv")
car_data.head()
# let's see the shape of data
car_data.shape     
car_data.info()
car_data.Owner=car_data.Owner.astype("object") #change the data type of Owner feature into category type
car_data.dtypes
car_data.head()
# check unique value in my categorial features
for x in car_data.iloc[:,1:]:
    if car_data[x].dtype =="O":
        print(car_data[x].unique())
    else:
        pass
car_data.isnull().sum()
# summary of data
car_data.describe(include="all")
# check if there are any outliers
car_data.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
sns.barplot('Seller_Type','Selling_Price',data=car_data)
sns.barplot('Transmission','Selling_Price',data=car_data,palette='spring')
sns.barplot('Fuel_Type','Selling_Price',data=car_data)
plt.scatter(car_data.Selling_Price,car_data.Present_Price)
plt.xlabel("Selling_price")
plt.ylabel("Present_price")
plt.title("sellling vs present ")
plt.scatter(car_data.Selling_Price,car_data.Kms_Driven)
plt.xlabel("Selling_price")
plt.ylabel("Kms_Driven")
plt.title("sellling vs Kms_Driven ")
car_data.head(2)
car_data["No_of_years"]=2020-car_data.Year
car_data.head()
car_data.drop("Year",axis=1,inplace=True) 
car_data.drop("Car_Name",axis=1,inplace=True) 
car_data.head(3)
#change into integer data type because i want to create dummy varibles
car_data.Owner=car_data.Owner.astype("int64")
car_data.dtypes
plt.figure(figsize=(10,5))
sns.barplot(car_data["No_of_years"],car_data["Selling_Price"])
# create dummies 
car_data_new=pd.get_dummies(car_data,drop_first=True)
car_data_new.head()
cor=car_data_new.corr()
# check corelation with better Visualization (heatmap) 
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True,cmap="RdYlGn") 
x=car_data_new.iloc[:,1:]
y=car_data_new.Selling_Price
x.head()
y.head()
#Feature importances 
from sklearn.ensemble import ExtraTreesRegressor
feature_model=ExtraTreesRegressor()
feature_model.fit(x,y)
feature_model.feature_importances_
feature_df=pd.DataFrame()
feature_df["features"]=x.columns
feature_df["importancy"]=feature_model.feature_importances_.round(2)
feature_df
 # five important features
important_five=feature_df.nlargest(5,columns="importancy")
important_five
x.head()
X=x[important_five.features] # select only important features
X.head()
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
x_train.shape
x_test.shape
from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(random_state=0)
rf_model.fit(X,y)
rf_model
#import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
# No. of trees in random forest
n_estimators=[100,200,300,400,500,600,700,800]

# No. of features to consider at every split
max_features=["auto","sqrt"]

# max no. of levels in tree
max_depth=[5,10,15,20,25,30]

# min no. of sample required to split a node
min_samples_split=[2,5,10,15,100,150]

# min no. of sample required at each leaf node
min_samples_leaf=[1,2,5,10,20,30]

parametrs={'n_estimators':n_estimators,'max_features':max_features,'max_depth':
          max_depth,'min_samples_split':min_samples_split,'min_samples_leaf':
          min_samples_leaf}
parametrs
random_srch=RandomizedSearchCV(estimator=rf_model,param_distributions=parametrs)
random_srch.fit(x_train,y_train)
random_srch.best_estimator_
# check out best parameters out of total given parameters
random_srch.best_params_
# now,predict with rf using RandomizedsearchCV
y_predict=random_srch.predict(x_test)
y_predict
fig = plt.figure()
sns.distplot(y_test-y_predict) 
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
fig = plt.figure()
plt.scatter(y_test,y_predict)
fig.suptitle('y_test vs y_predict', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=15)                          # X-label
plt.ylabel('y_predict', fontsize=13)  
# for checking model accuracy
from sklearn import metrics
score=metrics.r2_score(y_test,y_predict)
score
