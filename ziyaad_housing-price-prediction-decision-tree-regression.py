import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error



from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/melb_data.csv")



data.head()
data.columns
data.isnull().sum()

data = data.dropna(axis=0,how='any')

data.isnull().sum()
data.describe()
y = data['Price']     #Prediction Target
x = data[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']]
x.describe()
x.head()
model = DecisionTreeRegressor()



model.fit(x,y)
print("Making predictions")

print(x)



print("Predictions : ")

a = model.predict(x)

a
data.head()
mean_absolute_error(y,a)
train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)
model = DecisionTreeRegressor()



model.fit(train_x,train_y)
a = model.predict(val_x)

print(a)
print(mean_absolute_error(val_y,a))
def get_mae(leaf_nodes,train_x,val_x,train_y,val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=leaf_nodes,random_state=0)

    model.fit(train_x,train_y)

    a = model.predict(val_x)

    mae = mean_absolute_error(val_y,a)

    return(mae)
for leaf_nodes in np.arange(5,5000,500):

    mae = get_mae(leaf_nodes,train_x,val_x,train_y,val_y)

    print(leaf_nodes,"  >>>>>>>>>>  ",mae)
data.head()
x = data[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']]



y=data['Price']





from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=1)

model.fit(train_x,train_y)

a=model.predict(val_x)

mean_absolute_error(val_y,a)