#importing pandas:
import pandas as pd

# import libraries for feature engineering and preprocessing:
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder

# finally, importing algorithms to perform regression with:
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

# importing metrics to see how well the model predicts:
from sklearn.metrics import *
data = pd.read_csv("../input/fuel-consumption-co2/FuelConsumptionCo2.csv")
data.head(10)
y = data['CO2EMISSIONS']

data.drop(['CO2EMISSIONS'], axis=1, inplace=True)
data.isnull().sum()
c = (data.dtypes == 'object')

catlist = list(c[c].index)
catlist.append('MODELYEAR')
catlist.append('CYLINDERS')
data[catlist].nunique()
data.drop(['MODEL', 'MODELYEAR'], inplace=True, axis=1)
catlist.remove('MODELYEAR')
catlist.remove('MODEL')

encdata = data.copy()

cbe = CatBoostEncoder()

cbe.fit(data[catlist], y)

encdata[catlist] = cbe.transform(data[catlist])
encdata[catlist].head()
scale = StandardScaler()
scaleddata = scale.fit_transform(encdata)
train, test, ytrain, ytest = train_test_split(scaleddata, y, train_size=0.8, test_size=0.2)
RanMod = RandomForestRegressor(n_estimators=300)

RanMod.fit(train, ytrain)

RPreds = RanMod.predict(test)

print("The R2 Score is:",r2_score(ytest, RPreds), ", while the Mean Absolute Error is:",mean_absolute_error(ytest, RPreds))
TreeMod = DecisionTreeRegressor()

TreeMod.fit(train, ytrain)

TPreds = TreeMod.predict(test)

print("The R2 Score is:",r2_score(ytest, TPreds), ", while the Mean Absolute Error is:",mean_absolute_error(ytest, TPreds))
xgb = XGBRegressor(n_estimators=200)

xgb.fit(train, ytrain)

xPreds = xgb.predict(test)

print("The R2 Score is:",r2_score(ytest, xPreds), ", while the Mean Absolute Error is:",mean_absolute_error(ytest, xPreds))