import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math
df = pd.read_csv('../input/student-performance-record/AttendanceMarksSA.csv')
df.head()
X=df['MSE']
Y=df['ESE']
sns.scatterplot(X,Y)
beta0=0
beta1=0
alpha=0.01
count=10000
n=float(len(X))
for i in range(count):
  Ybar = beta1*X + beta0
  beta1 = beta1 - (alpha/n)*sum(X*(Ybar-Y))
  beta0 = beta0 - (alpha/n)*sum(Ybar-Y)

print(beta0,beta1)
Ybar = beta1*X + beta0

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
x=np.array(df['MSE']).reshape(-1,1)
y=np.array(df['ESE']).reshape(-1,1)

lr=LinearRegression()
lr.fit(x,y)

print(lr.coef_)
print(lr.intercept_)

yp=lr.predict(x)
rse=RSE(y,yp)

print(rse)