import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/iris/Iris.csv')
df.head()
X = df.iloc[:, 3].values
Y = df.iloc[:, 4].values
plt.scatter(X,Y, color = 'red')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
# Building the model
m = 0
c = 0

count = 10000 # The number of iterations to perform gradient descent
alpha = 0.0001   # The learning Rate

n = (len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(count): 
    Yhat = m*X + c  # The current predicted value of Y
    m = m - (alpha/n)*sum(X*(Yhat-Y))
    c = c - (alpha/n)*sum(Yhat-Y)
    
print (m, c)
# Making predictions
Y_hat = m*X + c

plt.scatter(X,Y, color = 'red')
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.plot([min(X), max(X)], [min(Y_hat), max(Y_hat)], color='blue') #regression line
plt.show()
import math
def RSE(y_true, y_pred):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_pred)
    RSS = np.sum(np.square(y_true - y_pred))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


rse= RSE(df['PetalWidthCm'],Y_hat)
print(rse)
Ymean = np.mean(Y)
rsquare = 1- sum((Y-Y_hat)*(Y-Y_hat))/sum((Y-Ymean)*(Y-Ymean))
rsquare
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = np.array(df['PetalLengthCm']).reshape(-1,1)
y = np.array(df['PetalWidthCm']).reshape(-1,1)

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(x)

print(model.coef_)
print(model.intercept_)
rse = RSE(y,y_predict)
print(rse)
Length = [1.7]
ResWidth = model.predict([Length])
print(ResWidth)
y_mean = np.mean(y)
Rsquare = 1- sum((y-y_predict)*(y-y_predict))/sum((y-y_mean)*(y-y_mean))
Rsquare
print(Rsquare)