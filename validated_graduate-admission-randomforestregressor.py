# import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# set random seed
np.random.seed(100)
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.columns
df.head()
df.plot(subplots=True,kind="box",figsize=(15,15))
X = df.drop(['Chance of Admit '], axis=1)
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# create a RandomForestRegressor
reg = RandomForestRegressor(n_estimators=50, max_depth = 5)
reg.fit(X_train,y_train)

print(reg.feature_importances_)
print('train accuracy ' + str(reg.score(X_train,y_train)))
print('test accuracy ' + str(reg.score(X_test,y_test)))
