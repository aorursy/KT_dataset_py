import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/SLR_Data.csv') #Reading the dataset and converting it into dataframe.
X = dataset.iloc[:, :-1]#accessing all the first columns of the dataset.
y = dataset.iloc[:, -1] #accessing the last column of the dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
from sklearn.linear_model import LinearRegression #This class contains the pre-built model.
regressor = LinearRegression() #reating an object of a class.
regressor.fit(X_train, y_train) #passing the training data into the model.
y_predict = regressor.predict(X_test) #for testing we only pass the features and not the ouput. 
plt.scatter(X_test, y_predict, color='red') #plotting the predicted points
plt.scatter(X_test, y_test,color='blue') #plotting the test data points
plt.title("SAT Score vs GPA Score (testing data vs predicted data)")
plt.xlabel("SAT Score")
plt.ylabel("GPA Score")
plt.show() # The red spots are the predicted spots of the GPA on the basis of SAT Score. The Accuracy can be seen.
print(y_test)
print(y_predict) #Check the similarity between the original and predicted data.
#Visualizing training data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue') 
plt.title("SAT Score vs GPA Score (Training Data)")
plt.xlabel("SAT Score")
plt.ylabel("GPA Score")
plt.show() #plotting the best fit line on the training data
#visualizing Testing data
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue') 
plt.title("SAT Score vs GPA Score (Testing Data)")
plt.xlabel("SAT Score")
plt.ylabel("GPA Score")
plt.show() #plotting the best fit line on the testing data