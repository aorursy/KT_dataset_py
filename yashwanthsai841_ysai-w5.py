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
import pandas as pd



train_path ='/kaggle/input/covid19-global-forecasting-week-5/train.csv'

test_path = '/kaggle/input/covid19-global-forecasting-week-5/test.csv'



X_full = pd.read_csv(train_path)

X_test = pd.read_csv(test_path)



X_full = X_full.drop(['County','Province_State'],axis=1)

X_test = X_test.drop(['County','Province_State'],axis=1)

X_test = X_test.rename(columns={'ForecastId':'Id'})



y=X_full.TargetValue



#print("Before Encoding")

#print(X_full.describe())

#print(X_full.head())

#print(X_test.describe())

#print(X_test.head())



from sklearn.preprocessing import LabelEncoder



def encode(data,columns):

    enc = LabelEncoder()

    for col in columns:

        data[col] = enc.fit_transform(data[col])

    return data



columns = ['Country_Region','Target','Date']

X_full = encode(X_full,columns)

X_test = encode(X_full,columns)



#print("After Encoding")

#print(X_full.describe())

#print(X_full.head())

#print(X_test.describe())

#print(X_test.head())



features = ['Country_Region', 'Population', 'Weight', 'Date', 'Target']



X = X_full[features]



from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=1)



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_val)



from sklearn.metrics import mean_absolute_error as mse

from sklearn.metrics import r2_score as r2



print("Mean Absolute Error:",mse(y_val,preds))

print("R2 Score:",r2(y_val,preds))



model.fit(X,y)

final_pred = model.predict(X_test[features])



print(final_pred)
import matplotlib.pyplot as plt

plt.plot(X.Country_Region,y,color='red')

plt.show()

plt.plot(X.Country_Region,final_pred,color='blue')

plt.show()