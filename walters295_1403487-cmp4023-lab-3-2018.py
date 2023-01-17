import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 7 + rng.randn(50)
pd.DataFrame({
    'input':x,
    'output':y
})
data = pd.DataFrame({
    'input':x,
    'output':y
})
data
pdTemp = pd.DataFrame
np.corrcoef(x, y)
data.corr()
plt.scatter(x, y)
plt.show()
#Making a copy of data and assigning it to 'dataTemp'
dataTemp = data.copy()
dataTemp

#separate our data into dependent (Y) and independent(X) variables
X_data = dataTemp['input']
Y_data = dataTemp['output']

#Using the train_test_split method to perform a 70/30 test split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)

# Create an instance of linear regression
regres = linear_model.LinearRegression()

#pdTemp(X_test)
pd.DataFrame(X_test)
#part b of Question four(4) - Cross validation train test
X_train = pdTemp(X_train)
model = regres.fit(X_train,y_train)
predictValues = regres.predict(pdTemp(X_test))
#View values in 70/30 cross validation test
print(predictValues)
print(model)
plt.scatter(X_test, predictValues,  color='black')
plt.show()
#Print accuracy score
print('The Score:', model.score(pdTemp(X_test),pdTemp(y_test)))
#Question Number Five(5) - Provide a plot illustrating the Residuals

import seaborn as sns
sns.set(style="darkgrid")
sns.residplot(regres.predict(X_train), regres.predict(X_train)-y_train,lowess=True, color="b")
sns.residplot(regres.predict(pdTemp(X_test)),regres.predict(pdTemp(X_test))-y_test,lowess=True, color="g")
plt.title('Residual Plot using Training (blue) and test (green) data ')
plt.ylabel('Residuals')
#Question Six(6) - Determine the Coefficient of Determination (R^2 ) of your model. Explain what this means

print('The Score:', model.score(pdTemp(X_test),pdTemp(y_test)))
regres.intercept_
regres.coef_
Y = 2.07789848 * 5.1432 + -7.18188796597139
print(Y)
#Calculate and print the result of Mean Absolute Error
print("The Mean Absolute Error: %.2f" % metrics.mean_absolute_error(pdTemp(y_test),predictValues))

#Calculate and print result of Mean Squared Error
print("The Mean Squared Error: %.2f" % metrics.mean_squared_error(pdTemp(y_test),predictValues))

#Calculate and print the result of Root Mean Squared Error
print("The Root Mean Squared Error: %.2f" % np.sqrt(metrics.mean_squared_error(pdTemp(y_test),predictValues)))
