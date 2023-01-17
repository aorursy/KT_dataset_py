import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
seed = 0
np.random.seed(seed)
from sklearn.datasets import load_boston
# Load the Boston Housing dataset from sklearn
boston = load_boston()
bos = pd.DataFrame(boston.data)
# give our dataframe the appropriate feature names
bos.columns = boston.feature_names
# Add the target variable to the dataframe
bos['Price'] = boston.target
# For student reference, the descriptions of the features in the Boston housing data set
# are listed below
boston.DESCR
bos.head()
# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)







