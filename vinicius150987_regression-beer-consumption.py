import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

sns.set(style='white', palette='deep')

warnings.filterwarnings('ignore')

%matplotlib inline



def autolabel_without_pct(rects,ax): #autolabel

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy = (rect.get_x() + rect.get_width()/2, height),

                    xytext= (0,3),

                    textcoords="offset points",

                    ha='center', va='bottom', fontsize=10)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing dataset

df = pd.read_csv('/kaggle/input/beer-consumption-sao-paulo/Consumo_cerveja.csv')

df.head()
#Analysing dataset with tradicinal way

#Statistical

statistical = df.describe()

statistical
#Using info

df.info()
#Looking for null values

null_values = (df.isnull().sum()/len(df))*100

null_values = pd.DataFrame(null_values,columns=['% of Null Values'])

null_values
df_feature = df.copy()

df_feature = df_feature.dropna()
df_feature.columns
#Tranforming string in float columns

df_feature.columns

for i in df_feature[['Temperatura Media (C)', 'Temperatura Minima (C)',

       'Temperatura Maxima (C)', 'Precipitacao (mm)']]:

    df_feature[i] = df_feature[i].str.replace(',','.')



for i in df_feature[['Temperatura Media (C)', 'Temperatura Minima (C)',

       'Temperatura Maxima (C)', 'Precipitacao (mm)']]:

    df_feature[i] = df_feature[i].astype(np.float64)
#Creating 'Dia da Semana' column

import calendar



df_feature['Data'] = df_feature['Data'].apply(pd.to_datetime)

df_feature['Dia da Semana'] = df_feature['Data'].apply(lambda x: x.weekday())



dias = {}

for i,v in enumerate(list(calendar.day_name)):

    dias[i]=v

    

dias_nomes = np.array([])

for i in df_feature['Dia da Semana']:

    for j in range(0,len(dias)):

        if i == list(dias.keys())[j]:

            dias_nomes = np.append(dias_nomes,dias[j])



df_feature['Dia da Semana'] = dias_nomes 

df.columns
#Which day of the week is consumed more alcohol?

pivot_table_day = df_feature.pivot_table('Consumo de cerveja (litros)','Dia da Semana').sort_values(by='Consumo de cerveja (litros)',

                                          ascending=False)



pivot_table_day.index[0]

pivot_table_day.values[1][0]

ind = np.arange(len(pivot_table_day.index))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

for i in np.arange(len(ind)):

    rects = ax.bar(pivot_table_day.index[i],np.around(pivot_table_day.values[i][0],2), edgecolor='b')

    autolabel_without_pct(rects,ax)

ax.set_title('Consumption per day of week')

ax.set_xlabel('Day of week')

ax.set_xticks(ind)

ax.set_ylabel('Consumption')

ax.grid(b=True, which='major', linestyle='--')

plt.tight_layout()
df.columns
#Which range of temperature is consumed more alcohol?

df_feature.columns

bins = np.arange(12,35,5)

bins_temp = bins.astype(str)

labels = []

count1=0

count2=1

while count2 != len(bins_temp):

    labels.append('['+bins_temp[count1]+' - '+bins_temp[count2]+']')

    count1+=1

    count2+=1



ind = np.arange(len(labels))



temp = pd.cut(df_feature['Temperatura Media (C)'], bins)



pivot_table_temp = df_feature.pivot_table('Consumo de cerveja (litros)',temp)



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

pivot_table_temp.plot(ax=ax, marker='o', markeredgecolor='b')

ax.set_xticks(ind)

ax.set_xticklabels(labels)

ax.set_ylabel('Consumption')

ax.set_title('Consumption per range of temperature')

ax.grid(b=True, which='major', linestyle='-')

plt.tight_layout()
#Define X and y

df_feature.columns

X = df_feature.drop(['Consumo de cerveja (litros)', 'Data'], axis=1)

y = df_feature['Consumo de cerveja (litros)']
#Get Dummies

X = pd.get_dummies(X)
#Avoiding Dummies Trap

X.columns

X = X.drop(['Dia da Semana_Monday'], axis= 1)
#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
## Multiple Linear Regression Regression

from sklearn.linear_model import LinearRegression

lr_regressor = LinearRegression()

lr_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = lr_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])
## Polynomial Regressor

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

X_poly = poly_reg.fit_transform(X_train)

lr_poly_regressor = LinearRegression()

lr_poly_regressor.fit(X_poly, y_train)



# Predicting Test Set

y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Suport Vector Regression 

'Necessary Standard Scaler '

from sklearn.svm import SVR

svr_regressor = SVR(kernel = 'rbf')

svr_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = svr_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

dt_regressor = DecisionTreeRegressor(random_state=0)

dt_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = dt_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)

rf_regressor.fit(X_train,y_train)



# Predicting Test Set

y_pred = rf_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
## Ada Boosting

from sklearn.ensemble import AdaBoostRegressor

ad_regressor = AdaBoostRegressor()

ad_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = ad_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

gb_regressor = GradientBoostingRegressor()

gb_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = gb_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Xg Boosting

from xgboost import XGBRegressor

xgb_regressor = XGBRegressor()

xgb_regressor.fit(X_train, y_train)



# Predicting Test Set

y_pred = xgb_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
##Ensemble Voting regressor

from sklearn.ensemble import VotingRegressor

voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),

                                                  ('lr_poly', lr_poly_regressor),

                                                  ('svr', svr_regressor),

                                                  ('dt', dt_regressor),

                                                  ('rf', rf_regressor),

                                                  ('ad', ad_regressor),

                                                  ('gr', gb_regressor),

                                                  ('xg', xgb_regressor)])



for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,

            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))



# Predicting Test Set

y_pred = voting_regressor.predict(X_test)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)  
#The Best Classifier

print('The best regressor is:')

print('{}'.format(results.sort_values(by='R2 Score',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

def display_scores (scores):

    print('Scores:', scores)

    print('Mean:', scores.mean())

    print('Standard:', scores.std())



lin_scores = cross_val_score(estimator=lr_regressor, X=X_train, y=y_train, 

                             scoring= 'neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
#Applying PCA (If Necessary)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_ 
## Multiple Linear Regression Regression (PCA)

from sklearn.linear_model import LinearRegression

lr_regressor_pca = LinearRegression()

lr_regressor_pca.fit(X_train_pca, y_train)



# Predicting Test Set

y_pred = lr_regressor_pca.predict(X_test_pca)

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, y_pred)

mse = metrics.mean_squared_error(y_test, y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

r2 = metrics.r2_score(y_test, y_pred)



model_results = pd.DataFrame([['Multiple Linear Regression (PCA)', mae, mse, rmse, r2]],

               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score'])



results = results.append(model_results, ignore_index = True)
results.sort_values(by='R2 Score', ascending=False)