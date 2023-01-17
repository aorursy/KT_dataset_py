import pandas as pd

import matplotlib.pyplot as plt



path="/kaggle/input/covid-india/covid_india.csv"

data1 = pd.read_csv(path)

y = data1.total_cases

x = data1.date



plt.plot(x, y, color='r') 

plt.xlabel('date')

plt.ylabel('total cases')

plt.title('compare')

plt.legend() 

plt.show()
df_final = pd.read_csv(path,na_values=['null'],parse_dates=True,infer_datetime_format=True)

df_final.drop(['total_cases','total_deaths','new_deaths','total_cases_per_million','total_tests','new_cases_per_million','total_deaths_per_million','new_deaths_per_million','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cvd_death_rate','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_100k','new_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','tests_units','stringency_index','population_density','median_age'], axis = 1,inplace=True)

df_final.to_csv('covid_india2.csv',index = False)

df_final.tail()
df_final.describe() ##df_final.shape
###Check for null data

df_final.isnull().values.any()
df_final['date']=pd.to_datetime(df_final['date'])

# test['Date']=test['Date'].dt.strftime("%Y%m%d")

df_final['date']=df_final['date'].dt.strftime("%Y%m%d").astype(int)

df_final.index.name = 'Id'

df_final.head()
y = df_final.new_cases

x = df_final.date



plt.plot(x, y, color='r') 

plt.xlabel('date')

plt.ylabel('new cases')

plt.title('compare')

plt.legend() 

plt.show()
x=df_final.iloc[:,[0,1]].values

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

x[:,0]=labelencoder.fit_transform(x[:,0])

y=df_final.iloc[:,[2]].values

y[107]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.15)

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn import linear_model

from sklearn.metrics import r2_score

rfr=RandomForestRegressor()

rfr.fit(x_train,y_train)

y_predicted2=rfr.predict(x_test)





r2_score(y_test,y_predicted2)
plt.plot(y_test, label='True')

plt.plot(y_predicted2, label='RandomForest')

plt.title("RF_Prediction")

plt.xlabel('Observation')

plt.ylabel('Cases')

plt.legend()

plt.show()

#create new dataframe with only the column Close

import math

data1 = df_final.filter(['new_cases'])

#into numpy array

print(len(y_train))

dataset = data1.values

train_data_len=math.ceil(len(dataset)-23)

train_data_len
valid=df_final[train_data_len:]

train=df_final[:train_data_len]

valid['Prediction']=y_predicted2
plt.figure(figsize=(16,8))

plt.title('Random Forest')

plt.xlabel('Date',fontsize=18)

plt.ylabel('Cases',fontsize=18)

plt.plot(train['new_cases'])

plt.plot(valid[['new_cases','Prediction']])

plt.legend(['Train','Real Value','Prediction'],loc='upper left')

plt.show()
from sklearn.tree import DecisionTreeRegressor 

from sklearn import tree

clf = DecisionTreeRegressor()

clf=RandomForestRegressor()

clf.fit(x_train,y_train)

y_predicted_clf=clf.predict(x_test)



valid=df_final[train_data_len:]

train=df_final[:train_data_len]

valid['Prediction']=y_predicted_clf



r2_score(y_test,y_predicted_clf)
plt.plot(y_test, label='True')

plt.plot(y_predicted2, label='Decision Tree')

plt.title("DT_Prediction")

plt.xlabel('Observation')

plt.ylabel('Cases')

plt.legend()

plt.show()

plt.figure(figsize=(16,8))

plt.title('Decision Tree')

plt.xlabel('Date',fontsize=18)

plt.ylabel('Cases',fontsize=18)

plt.plot(train['new_cases'])

plt.plot(valid[['new_cases','Prediction']])

plt.legend(['Train','Real Value','Prediction'],loc='upper left')

plt.show()