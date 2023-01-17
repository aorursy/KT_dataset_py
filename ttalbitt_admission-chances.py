import pandas as pd
import numpy as np

file = open('../input/graduate-admissions/Admission_Predict_Ver1.1.csv', 'r')
df = pd.read_csv(file)

y = df['Chance of Admit ']
x = df.drop(['Serial No.', 'Chance of Admit '], axis=1)

x_train = x.iloc[:400, :]
x_test = x.iloc[400:, :]

y_train = y.iloc[:400]
y_test = y.iloc[400:]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


reg = LinearRegression().fit(x_train.to_numpy(), y_train.to_numpy())

print(np.sqrt(mean_squared_error(y_test, reg.predict(x_test))))
