import pandas as pd

import numpy as np

from scipy import stats
data = pd.read_csv("../input/Summary of Weather.csv")



data
x=data['MinTemp']

y=data['MaxTemp']
stats.pearsonr(y,x)
x=np.array(x)

x=x.reshape(-1,1)



y=np.array(y)

y=y.reshape(-1,1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.0005)



model = LinearRegression()

model.fit(x_train,y_train)

a = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(a,y_test)
for i in range(len(a)):

    print(a[i],y_test[i])
import matplotlib.pyplot as plt



fig,ax = plt.subplots()



plt.scatter(x_test,y_test)

plt.scatter(a,y_test)



plt.show()