#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the dataset

dataset=pd.read_csv("../input/data.csv")

x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,8].values
#This correlation coefficicent matrix follows the 

#example from the seaborn example:

#http://seaborn.pydata.org/examples/many_pairwise_correlations.html

import seaborn as sns

corr = dataset.corr('kendall')

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(8, 6))

cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,square=True, 

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax);
dataset.apply(lambda x: sum(x.isnull()),axis=0)
dataset.describe()
dataset['rings'].hist(bins=5)
dataset.boxplot(column='rings')
dataset.boxplot(column='rings', by = 'Sex')


plt.scatter(x[:,7],y,color='blue')
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()

x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])

x = onehotencoder.fit_transform(x).toarray()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Fitting the multiple linear regression training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)
#Predicting the test set results

y_pred=regressor.predict(x_test)
#Estimating rmse value

from sklearn import metrics

metrics.mean_absolute_error(y_test,y_pred)
metrics.mean_squared_error(y_test,y_pred)
np.sqrt(mse)
#Building the optimal model using backward elimination

import statsmodels.formula.api as sm

x=np.append(arr=np.ones((4177,1)).astype(int),values=x,axis=1)#just adding one column of all values '1'

x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10]]

regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()
x_opt=x[:,[0,1,2,3,5,6,7,8,9,10]]

regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()
#Replacing x with x_opt and calculating result

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)



from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)





y_pred=regressor.predict(x_test)





regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()

regressor_OLS.summary()

#Creating index array



array=[]

for i in range(0,836):

    array.append(i)

    



#Visualising the test set results

plt.scatter(array[:10],y_test[:10],color='red')

plt.scatter(array[:10],y_pred[:10],color='blue')

plt.show()

plt.plot(x_train[:,8],regressor.predict(x_train),color='blue')

plt.title('salary vs experience(test)')

plt.xlabel('years of experience')

plt.ylabel('salary')

plt.show()