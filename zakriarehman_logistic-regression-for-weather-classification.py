# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/weather.csv')
#Checking Dataset before applying Encoding

df.head()
#To Convert Categorical vaariables into Numerical Variables using Pandas

df=pd.get_dummies(df,drop_first=True)
#Data after Encoding

df.head()
#Checking dataset for Datatypes and other information 

df.info()
#Checking missing values

df.isnull().sum()
df_1=df[['Temperature_c','Humidity','Wind_Speed_kmh','Wind_Bearing_degrees','Visibility_km','Pressure_millibars']]

df_1
df.describe()
def outliers(dataset,column_name):

    IQR=dataset[column_name].quantile(0.75)-dataset[column_name].quantile(0.25)

    

    upper_boundary=dataset[column_name].quantile(0.75)+(IQR*1.5)

    lower_boundary=dataset[column_name].quantile(0.25)+(IQR*1.5)

    

    return upper_boundary, lower_boundary
print('Upper and Lower Boundaries of Temperature_c ',outliers(df,'Temperature_c'))

print('Upper and Lower Boundaries of Humidity ',outliers(df,'Humidity'))

print('Upper and Lower Boundaries of Wind_Speed_kmh ',outliers(df,'Wind_Speed_kmh'))

print('Upper and Lower Boundaries of Wind_Bearing_degrees ',outliers(df,'Wind_Bearing_degrees'))

print('Upper and Lower Boundaries of Visibility_km ',outliers(df,'Visibility_km'))

print('Upper and Lower Boundaries of Pressure_millibars ',outliers(df,'Pressure_millibars'))
fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(8,8))

plt.tight_layout()

axes[0,0].set_title('Outliers For Temperature_c')

axes[0,1].set_title('Outliers For Humidity')

axes[1,0].set_title('Outliers For Wind_Speed_kmh')

axes[1,1].set_title('Outliers For Wind_Bearing_degrees')

axes[2,0].set_title('Outliers For Visibility_km')

axes[2,1].set_title('Outliers For Pressure_millibars')



axes[0,0].boxplot(df['Temperature_c'])

axes[0,1].boxplot(df['Humidity'])

axes[1,0].boxplot(df['Wind_Speed_kmh'])

axes[1,1].boxplot(df['Wind_Bearing_degrees'])

axes[2,0].boxplot(df['Visibility_km'])

axes[2,1].boxplot(df['Pressure_millibars'])



fig.savefig('Six_Subplots')
X=df.iloc[:,[0,1,2,3,4,5,7,8]]

y=df['Rain']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler

scaler =StandardScaler().fit(X_train)

scaler.transform(X_train)
grid={"C":np.logspace(1,10,10), "penalty":["l1","l2"]}# l1 lasso l2 ridge

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(X_train,y_train)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_)
regressor=LogisticRegression(C=10000,penalty= 'l2')

regressor.fit(X_train,y_train)
intercept = regressor.intercept_

coefficients = regressor.coef_
coef_list = list(coefficients[0,:])

coef_df = pd.DataFrame({'Feature': list(X_train.columns),'Coefficient': coef_list})

print(coef_df)
predicted_prob = regressor.predict_proba(X_test)[:,1]

y_pred=regressor.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = pd.DataFrame(confusion_matrix(y_test, y_pred))

cm['Total'] = np.sum(cm, axis=1)

cm = cm.append(np.sum(cm, axis=0), ignore_index=True)

cm.columns = ['Predicted No', 'Predicted Yes', 'Total']

cm = cm.set_index([['Actual No', 'Actual Yes', 'Total']])

print(cm)
from sklearn.metrics import classification_report

clf_report=classification_report(y_test,y_pred)

print(clf_report)