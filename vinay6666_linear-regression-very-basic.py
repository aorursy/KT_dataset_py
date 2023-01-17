from sklearn import linear_model
reg=linear_model.LinearRegression()
x=[[0],[1],[2]]
y=[2,5,6]
import matplotlib.pyplot as plt

%matplotlib inline  
plt.scatter(x,y,color='black')
reg.fit(x,y)
reg.coef_
reg.intercept_
plt.scatter(x,y,color='black')

#abline_values = [slope * i + intercept for i in x]

abline_values = [reg.coef_ * i + reg.intercept_ for i in x]

plt.plot(x, abline_values, 'b')