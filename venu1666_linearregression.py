import pandas as pd
train_data_path='../input/train.csv'     #setting path for training dataset in input directory
train_data=pd.read_csv(train_data_path)      #Reading the csv file of the training set
train_data.columns=['x','y']             #Rename the train_data columns name as x and y
from sklearn import linear_model       #importing linear_model for regression
Lreg=linear_model.LinearRegression()   # Taking the Linear regression model
Lreg.fit(train_data[['x']],train_data[['y']])      #Training the data using hypothesis function
test_path='../input/test.csv'         #setting the path to test the hypothesis function
test_read=pd.read_csv(test_path)      #reading test_data csv file
test_read.columns=['x','y']           #renaming the column of test_data
p=predict_test_data=Lreg.predict(test_read[['x']])   #predicting the output for new data
r=predict_train_data=Lreg.predict(train_data[['x']])  #predicting the output for training data
                                                   #print the values if u want
from sklearn.metrics import mean_squared_error           # importing mse to find the cost error
a=mean_squared_error(Lreg.predict(test_read[['x']]),test_read[['y']]) #cost function for test_data
b=mean_squared_error(Lreg.predict(train_data[['x']]),train_data[['y']])           #cost function for train_data
print(a)  #train data cost 
print(b)  #test data cost
import matplotlib.pyplot as plt
plt.scatter(train_data.x,train_data.y,color='r')   # train_data
plt.plot(train_data.x,r)
plt.scatter(test_read.x,test_read.y,color='g')     #test_data
plt.plot(test_read.x,p)
