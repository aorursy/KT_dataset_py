import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_excel("Data_Train.xlsx")
test_df = pd.read_excel("Test_set.xlsx")
train_df.head()
test_df.head()
big_df = train_df.append(test_df,sort=False)
big_df.tail()
big_df.dtypes
big_df['Date']  = big_df['Date_of_Journey'].str.split('/').str[0]
big_df['Month'] = big_df['Date_of_Journey'].str.split('/').str[1]
big_df['Year']  = big_df['Date_of_Journey'].str.split('/').str[2]
big_df.head()
big_df.dtypes
big_df['Date']  = big_df['Date'].astype(int)
big_df['Month'] = big_df['Month'].astype(int)
big_df['Year']  = big_df['Year'].astype(int)
big_df.dtypes
big_df = big_df.drop('Date_of_Journey',axis=1)
big_df.head()
big_df['Arrival_Time'] = big_df['Arrival_Time'].str.split(' ').str[0]
big_df.head()
big_df.isnull().sum()
big_df[big_df['Total_Stops'].isnull()]
big_df['Total_Stops'] = big_df['Total_Stops'].fillna('1 stop')
big_df['Total_Stops'] = big_df['Total_Stops'].replace('non-stop', '0 stop')
big_df.head()
big_df['stop'] = big_df['Total_Stops'].str.split(' ').str[0]
big_df.head()
big_df['stop'] = big_df['stop'].astype(int)
big_df = big_df.drop('Total_Stops',axis=1)
big_df.head()
big_df.dtypes
big_df['Arrival_Hour']     = big_df['Arrival_Time'].str.split(':').str[0]
big_df['Arrival_Minute']   = big_df['Arrival_Time'].str.split(':').str[1]
big_df['Departure_Hour']   = big_df['Dep_Time'].str.split(':').str[0]
big_df['Departure_Minute'] = big_df['Dep_Time'].str.split(':').str[1]
big_df['Arrival_Hour'] = big_df['Arrival_Hour'].astype(int)
big_df['Arrival_Minute'] = big_df['Arrival_Minute'].astype(int)
big_df['Departure_Hour'] = big_df['Departure_Hour'].astype(int)
big_df['Departure_Hour'] = big_df['Departure_Hour'].astype(int)
big_df = big_df.drop(['Arrival_Time','Dep_Time'],axis=1)
big_df.head()
big_df.Route.unique()
big_df['Route_1'] = big_df['Route'].str.split('→ ').str[0]
big_df['Route_2'] = big_df['Route'].str.split('→ ').str[1]
big_df['Route_3'] = big_df['Route'].str.split('→ ').str[2]
big_df['Route_4'] = big_df['Route'].str.split('→ ').str[3]
big_df['Route_5'] = big_df['Route'].str.split('→ ').str[4]
big_df['Route_6'] = big_df['Route'].str.split('→ ').str[5]
big_df.head()
big_df['Route_1'].fillna('None',inplace=True)
big_df['Route_2'].fillna('None',inplace=True)
big_df['Route_3'].fillna('None',inplace=True)
big_df['Route_4'].fillna('None',inplace=True)
big_df['Route_5'].fillna('None',inplace=True)
big_df['Route_6'].fillna('None',inplace=True)
big_df.head()
big_df = big_df.drop(['Duration','Route'],axis=1)
big_df.head()
big_df[big_df['Price'].isnull()]
big_df['Price'].fillna(big_df['Price'].mean(),inplace=True)
big_df.isnull().sum()
big_df.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
big_df['Airline'] = encoder.fit_transform(big_df['Airline'])
big_df['Source'] = encoder.fit_transform(big_df['Source'])
big_df['Destination'] = encoder.fit_transform(big_df['Destination'])
big_df['Additional_Info'] = encoder.fit_transform(big_df['Additional_Info'])
big_df['Route_1'] = encoder.fit_transform(big_df['Route_1'])
big_df['Route_2'] = encoder.fit_transform(big_df['Route_2'])
big_df['Route_3'] = encoder.fit_transform(big_df['Route_3'])
big_df['Route_4'] = encoder.fit_transform(big_df['Route_4'])
big_df['Route_5'] = encoder.fit_transform(big_df['Route_5'])
big_df['Route_6'] = encoder.fit_transform(big_df['Route_6'])
big_df.head()
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
df_train = big_df[:10683]
df_test = big_df[10683:]
X = df_train.drop('Price',axis=1)
y = df_train['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
model = SelectFromModel(Lasso(alpha=0.005,random_state=0))
model.fit(X_train,y_train)
model.get_support()
selected_features = X_train.columns[(model.get_support())]
selected_features
X_train = X_train.drop('Year',axis=1)
X_test=X_test.drop(['Year'],axis=1)
from sklearn.model_selection import RandomizedSearchCV
# number of trees
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
# number of fetaures to consider at every split
max_features = ['auto','sqrt']
# max level in tree
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
# min sample required for split
min_samples_split = [2,5,10,15,100]
# min samples at each leaf node
min_samples_leaf = [1,2,5,10]
# create a random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)
# use the random search to find best hyper parameters
# first create a base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# search of parameters
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=50,cv=5,verbose=2,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
y_pred = rf_random.predict(X_test)
sns.distplot(y_test-y_pred)
plt.scatter(y_test,y_pred,c='black')
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score of Our Model is : ",metrics.r2_score(y_test, y_pred))