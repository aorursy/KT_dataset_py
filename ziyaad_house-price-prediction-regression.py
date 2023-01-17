import pandas as pd

import numpy as np

import scipy.stats as s
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
data.describe()
data.columns
for i in data.columns[3:21]:

    print(i,s.pearsonr(data['price'], data[i]))
y=data['price']

x=data[['bedrooms','bathrooms','sqft_living','waterfront','view','grade','sqft_above','sqft_basement','lat','sqft_living15']]
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,r2_score
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.001)
model = LinearRegression()

model.fit(x_train,y_train)

a = model.predict(x_test)



print(mean_absolute_error(y_test,a))

print(r2_score(y_test,a))
import matplotlib.pyplot as plt



fig,ax = plt.subplots()



ax.scatter(a,y_test)



plt.show()