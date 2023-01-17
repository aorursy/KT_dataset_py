import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier

from sklearn.svm import SVR

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score
wine_data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine_data.describe()
wine_data.head(10)
Missing = pd.concat([wine_data.isnull().sum()], axis=1, keys=['data'])

Missing[Missing.sum(axis=1) > 0]
corr = wine_data.corr()

plt.figure(figsize=(14,8))

plt.title('Overall Correlation', fontsize=14)

sns.heatmap(corr,annot=True,cmap='coolwarm_r',linewidths=0.2,annot_kws={'size':11})

plt.show()
wine_data.dropna(axis=0, subset=['quality'], inplace=True)

target = wine_data.quality

wine_data.drop(['quality'], axis=1, inplace=True)
print(target)
# Break off validation set from training data

wdX_train, wdX_valid, target_y_train, target_y_valid = train_test_split(wine_data, target, 

                                                                train_size=0.75, test_size=0.25,

                                                                random_state=0)
# Define models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=150, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=1000, random_state=0)

modelXGB = XGBRegressor(max_depth=3,learning_rate=0.01,n_estimators=2000,random_state=0, silent=True)



model = [model_1, model_2, model_3, model_4, model_5, modelXGB]
# Function for comparing different models

def score_model(model, X_t=wdX_train, X_v=wdX_valid, y_v=target_y_train, y_t=target_y_valid):

    model.fit(X_t, y_v)

    preds = model.predict(X_v)

    return mean_absolute_error(y_t, preds)



for i in range(0, len(model)):

    mae = score_model(model[i])

    print("Model %d MAE: %2f" % (i+1, mae))
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('model', model_5)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(wdX_train, target_y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(wdX_valid)



# Evaluate the model

score = mean_absolute_error(target_y_valid, preds)

print('MAE:', score)