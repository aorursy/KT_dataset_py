
##A SUPERVISED LEARNING PREDICTION USING REGRESSION
import sklearn
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 
import pylab
import pandas 
#from sklearn.impute import SimpleImputer as SI
#imputer = SI(strategy="most_frequent")

Melbourne_data = pd.read_csv(r"../input/melbourne-housing-snapshot/melb_data.csv")
Melbourne_data.head()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()
melb = Melbourne_data["Regionname"]
Reg_label = LE.fit_transform(melb)
Reg_mapping = {index:label for index, label in enumerate(LE.classes_)}
Reg_mapping
Melbourne_data["Regionname"]= Reg_label
melb_2 = Melbourne_data['Type']
#Council_label = LE.fit_transform(melb_2)
#me= np.unique(melb_2)
Type_label = LE.fit_transform(melb_2)
type_mapping ={index:label for index, label in enumerate(LE.classes_)}
type_mapping
Melbourne_data['Type']=Type_label 
melb_3 = Melbourne_data['SellerG']
Seller_label = LE.fit_transform(melb_3)
seller_mapping = {index:label for index, label in enumerate(LE.classes_)}
Melbourne_data['sellerG'] = Seller_label
#Melbourne_data.info()
#b = np.unique(melb_3)





Melbourne_new = Melbourne_data.drop(['Suburb', 'Address', 'Method', 'SellerG','Date','CouncilArea'], axis=1)

Melbourne_1 = Melbourne_new.copy()

from sklearn.impute import SimpleImputer as SI
imputer = SI(missing_values=np.nan, strategy = 'median')
Melbourne_1[['Car', 'BuildingArea', 'YearBuilt']] = imputer.fit_transform(Melbourne_1[['Car', 'BuildingArea', 'YearBuilt']])
Melbourne_1.info()

#to check correlation between columns
k=Melbourne_1.corr()
k['Price'].sort_values(ascending=False)


#spliting of data
from sklearn.model_selection import train_test_split
Y_price = Melbourne_1.Price
#drop my target
Mbourne2= Melbourne_1.drop(['Price'], axis=1)
#drop all non-int
X_others=Melbourne_1.select_dtypes(exclude=['object'])
#divide into training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X_others,Y_price, train_size = 0.8, test_size =0.2, random_state=0)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
RFR= RandomForestRegressor()
#funxtion for comparing different approaches 
def score_dataset(X_train, X_valid, Y_train, Y_valid ):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_valid)
    #THE NUMBER OF PREDICTIONS I WANT TO MAKE, NOW 10
    some_valid = list(Y_valid.iloc[:])
    some_predict = list(Y_predict[:])
    c = pd.DataFrame({"ACTUAL_VALUE":some_valid, "PREDICTIONS":some_predict}) 
    print(c)
    

# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

y = ____

# Check your answer
step_1.check()
# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()
# Create the list of features below
feature_names = ___

# Select data corresponding to features in feature_names
X = ____

# Check your answer
step_2.check()
# step_2.hint()
# step_2.solution()
# Review data
# print description or statistics from X
#print(_)

# print the top few lines
#print(_)
# from _ import _
#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = ____

# Fit the model
____

# Check your answer
step_3.check()
# step_3.hint()
# step_3.solution()
predictions = ____
print(predictions)

# Check your answer
step_4.check()
# step_4.hint()
# step_4.solution()
# You can write code in this cell
