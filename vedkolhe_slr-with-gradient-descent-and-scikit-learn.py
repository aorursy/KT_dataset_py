import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/attendence-dataset/2)AttendanceMarksSA-200919-184800.csv')
df.head()
X= df['MSE']
Y=df['ESE']
sns.scatterplot(X,Y)
gama0=0
gama1=0
alpha=0.01
count =10000
n=float(len(X))
for i in range(count): 
    Ybar = gama1*X + gama0    
    gama1 = gama1 - (alpha/n)*sum(X*(Ybar-Y))
    gama0 = gama0 - (alpha/n)*sum(Ybar-Y)
    
print(gama0,gama1)
Ybar = gama1*X + gama0
 
plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')  # regression line
plt.show()
import math
def RSE(y_true, y_predicted):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


rse= RSE(df['ESE'],Ybar)
print(rse)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = np.array(df['MSE']).reshape(-1,1)
y = np.array(df['ESE']).reshape(-1,1)
 

lr = LinearRegression()
lr.fit(x,y)


print(lr.coef_)
print(lr.intercept_)

a = lr.predict(x)
rse = RSE(y,a)

print(rse)