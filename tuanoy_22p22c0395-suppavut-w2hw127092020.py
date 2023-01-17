# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plant1Generation = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv", parse_dates=["DATE_TIME"])
plant1Generation.head()
# plant1GenerationUpdated = plant1Generation.pivot(index="DATE_TIME", columns="SOURCE_KEY", values=["DAILY_YIELD", "DC_POWER", "AC_POWER","TOTAL_YIELD"])
# plant1GenerationUpdated.head()
plant1Generation["SOURCE_KEY"].unique()
plant1Generation.describe()
plant1Weather = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv", parse_dates=["DATE_TIME"])
plant1Weather = plant1Weather.drop(["PLANT_ID", "SOURCE_KEY"], axis=1)
plant1Weather.head(20)
plant1Weather.describe()
plant1WeatherSelect = plant1Generation[plant1Generation["SOURCE_KEY"]=="1BY6WEcLGh8j5v7"]
plant1WeatherUpdate = pd.DataFrame()
previousYield = 0
for index, row in plant1WeatherSelect.iterrows():
    tempValue = row["DAILY_YIELD"] - previousYield
    if(tempValue < 0):
        row["YIELD"] = 0
    else:
        row["YIELD"] = tempValue
    previousYield = row["DAILY_YIELD"]
    plant1WeatherUpdate = plant1WeatherUpdate.append(row)
plant1WeatherUpdate.head(5)
plant1WeatherReady = plant1WeatherUpdate.drop(["DAILY_YIELD", "PLANT_ID", "SOURCE_KEY", "TOTAL_YIELD"], axis=1)
plant1WeatherReady
mergeData = pd.merge(plant1Weather, plant1WeatherReady, on="DATE_TIME")
mergeData = mergeData.drop(columns="DATE_TIME")
mergeData.tail(60)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def resultOfModel(y_test, y_pred):
    # Reference each value : https://datarockie.com/2019/03/30/top-ten-machine-learning-metrics/
    print('\033[1m{:10s}\033[0m'.format('Confusion Matrix'))
    print('TN', 'FP', 'FN', 'TP')
    print(confusion_matrix(y_test, y_pred).ravel())

    print("\n")
    print('\033[1m{:10s}\033[0m'.format('Classification Report'))
    print(classification_report(y_test,y_pred))

    print("\n")
    print('\033[1m{:10s}\033[0m'.format('Matrix'))
    model_results = pd.DataFrame([['Model result (n=100)', accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)]],
                   columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    print(model_results)

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

targetColumn = 'YIELD'

X = mergeData.drop(columns=targetColumn).to_numpy()
y = mergeData[targetColumn].to_numpy()

### Normally use KFold to train test split
# kf = KFold(n_splits=10, random_state=None, shuffle=False)
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# 0. Create variable
fold = []
step = 10 
stop = len(mergeData)

# 1. Create index
for i in range(0,10):
    fold.append(np.arange(start=i, stop=stop, step=step))
    
# 2. User index to create train test data
# for i in range(len(fold)):
for i in range(1):
    # Train index derive from other but i 
    # Test index derive from i
    train_index = []
    test_index = []

    for j in range(len(fold)):
        if(i != j):
            train_index.extend(fold[j])
        
    test_index = fold[i]
    
    # 3. Assign train test value
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ### Train model here
    
    # 4.1  Linear Regression
    regr = LinearRegression()
    
    # 4.2  Ridge Regression
    # regr = Ridge(alpha = 0.5)

    # 4.3  SVR Kernel-linear
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    # y_pred = regr.predict(X_test)

    result = regr.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (result*100.0))
   
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = [0]
colors = ['r']
for target, color in zip(targets,colors):
    indicesToKeep = mergeData[targetColumn] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
