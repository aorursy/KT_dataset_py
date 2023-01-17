#Import Libraries

import pandas as pd

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error 

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import median_absolute_error

import matplotlib.pyplot as plt

#----------------------------------------------------

#reading data

data = pd.read_csv('../input/house-delhi/datasets_846956_1445244_Delhi.csv')

data.head()

data.describe()



#X Data

X = data.drop(['Price'], axis=1, inplace=False)

print('X Data is \n' , X.head())

print('X shape is ' , X.shape)



#y Data

y = data['Price']

print('y Data is \n' , y.head())

print('y shape is ' , y.shape)



#load regression data



'''

X ,y = make_regression(n_samples=100, n_features=100, n_informative=10,

                       n_targets=1, bias=0.0, effective_rank=None,

                       tail_strength=0.5, noise=0.0, shuffle=True, coef=False,

                       random_state=None)

'''



X ,y = make_regression(n_samples=10000, n_features=500,shuffle=True)



#X Data

print('X Data is \n' , X[:10])

print('X shape is ' , X.shape)



#y Data

print('y Data is \n' , y[:10])

print('y shape is ' , y.shape)



#----------------------------------------------------

#Splitting data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)



#----------------------------------------------------

#Applying Linear Regression Model 



LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

LinearRegressionModel.fit(X_train, y_train)



#Calculating Details

print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))

print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)

print('----------------------------------------------------')



#Calculating Prediction

y_pred = LinearRegressionModel.predict(X_test)

print('Predicted Value for Linear Regression is : ' , y_pred[:10])



#----------------------------------------------------

#Calculating Mean Absolute Error

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

print('Mean Absolute Error Value is : ', MAEValue)



#----------------------------------------------------

#Calculating Mean Squared Error

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

print('Mean Squared Error Value is : ', MSEValue)



#----------------------------------------------------

#Calculating Median Absolute Error

MdSEValue = median_absolute_error(y_test, y_pred)

print('Median Absolute Error Value is : ', MdSEValue )



#----------------------------------------------------

# Adjusting Style . .  :  

plt.style.use('seaborn-notebook')



#----------------------------------------------------

# Adjusting Figure Size . .  :  

plt.figure(figsize = (7, 4))

plt.title('Graph')

plt.xlabel('X Data')

plt.ylabel('y Data')

#----------------------------------------------------

#Defining Data

XGraph = X_train[:,0]

yGraph = y_train



#----------------------------------------------------

# Drawing scatter graph No 1 . .  :  

plt.scatter(XGraph,yGraph,s=30,alpha=1,color= 'r') # can be : b ,g ,y ,c ,m ,k ,w 



plt.show()



# Drawing bar graph 

plt.bar(XGraph,yGraph,width=0.5,color='g',alpha=0.5)



plt.show()



# Drawing HexBin graph 

plt.hexbin(XGraph, yGraph, gridsize=5, cmap ='Blues')# it can be : Reds,Greens

plt.colorbar()

plt.show()

# Drawing plot graph No 1 . .  :   

plt.plot(XGraph,yGraph,linewidth=1,color= 'b', # can be : b ,g ,y ,c ,m ,k ,w 

         alpha=1,linestyle = 'solid') # can be  : dotted, dashed , dashdot



plt.show()

#Defining Data

XGraph = X_train[:,0]

yGraph = y_train



#----------------------------------------------------

# Drawing Hist2D graph 

plt.hist2d(XGraph, yGraph, bins = 20, cmap ='Blues')# it can be : Reds,Greens

plt.colorbar()

plt.show()