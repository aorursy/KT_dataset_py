# 1.0 Call libraries

%reset -f

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

#%matplotlib qt5
# 1.0.1 For measuring time elapsed

from time import time
# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct
# 1.3 Model building

#    Helpful Hint: Install h2o as: conda install -c h2oai h2o=3.22.1.2

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# 1.4 for ROC graphs & metrics

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics
# 1.5 Misc

import gc
# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300
# 2.0 Read data

# Note: Test data is already provided in a seperate file

#os.chdir("E:\\Data\\work\\Analytics\\FORE\\Datasets\\fashion_mnist")

#mn = pd.read_csv("fashion-mnist_train.csv.zip")

#mn_test = pd.read_csv("fashion-mnist_test.csv.zip")



mn = pd.read_csv("../input/fashion-mnist_train.csv")

mn_test = pd.read_csv("../input/fashion-mnist_test.csv")
# 2.1 Explore data

mn.head(3)

mn.info()                       # # NULLS?? 

mn.isnull().sum()               # Any NULLS - None

# 2.1.1 Examine distribution of continuous variables

mn.describe()                   # Data distributed from 0 - 255



mn.shape                        # (60000,785)

mn.columns.values               # Target is Ist column: 'label'

mn.dtypes.value_counts()        # all 785 columns are type int64  4

# 2.2 Summary of target feature

mn['label'].value_counts()     # Uniformly Distributed across classes [0-9]: 6000

# 3.0 quickly view a data row in the form of image

#    Get the first row excluding first column

#    First column contains class labels and 

#    other columns contain pixel-intensity values

abc = mn.values[1, 1:]

abc.shape    # (784,)

abc = abc.reshape(28,28)   # Reshape to 28 X 28

# 4.0 And plot it

plt.imshow(abc)

plt.show()

# 4.0 Separation into target/predictors

# 4.1 Training Data

y = mn.iloc[:,0]

X = mn.iloc[:,1:]

X.shape              # 60000 X 784

# 4.2 Test Data

y_test = mn_test.iloc[:,0]

X_test = mn_test.iloc[:,1:]

X_test.shape              # 10000 X 784

# 5.0 Which columns are numerical and which categorical?

num_columns = X.select_dtypes(include = ['float64','int64']).columns

num_columns

# 6. Start creating transformation objects

# 6.1 tuple for numeric columns

num = ("numtrans", ss() , num_columns)

# 6.2 Instantiate column transformer object

colTrans = ct([num])

# 6.3 Fit and transform

X_trans = colTrans.fit_transform(X)

X_trans.shape              # 60000 X 784

type(X_trans)

# 7.0 Preparing to model data with deeplearning

#      H2o requires composite data with both predictors

#      and target



X_train = pd.DataFrame(data=X_trans, columns=num_columns)

type(X_train) #pandas.core.frame.DataFrame



y = pd.DataFrame(data=y)

X = np.hstack((X_train,y))

type(X) #numpy.ndarray

# 8 Delete not needed variables and release memory

del(X_trans)

del(X_train)

del(abc)

gc.collect()

# 9.1 Start h2o

h2o.init()

# 9.2 Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    # 785

df.shape           # 60000 X 785

#df.columns

# 10. Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

# 10.1 X_column names

X_columns = df.columns[0:784]        # Only column names. No data

X_columns       # C1 to C784

# 10.2 y_column names

y_columns = df.columns[784]

y_columns       #C785

df['C785'].head()      # Just to be sure

# 11.1 For classification, target column must be factor

#      Required by h2o

df['C785'] = df['C785'].asfactor()

# 12. Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html



dl_model = H2ODeepLearningEstimator(epochs=1000,

                                    distribution = 'multinomial',                 # Response has categories [0-9]

                                    missing_values_handling = "MeanImputation", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [100,100],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')

# 12.1 Train our model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60
# 12.2 Get model summary

print(dl_model)

# 13. Time to make predictions on actual 'test' data

#     Create a composite X_test data before transformation to

#     H2o dataframe.

y_test = (y_test.values).reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 10000,1

# 14.0 Get shape of Test dataset pre-stacking

X_test.shape     # 10000,784

# 14.1 Column-wise stack now

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 10000,785

# 14.2 Transform X_test to h2o dataframe

X_test = h2o.H2OFrame(X_test)

X_test['C785'] = X_test['C785'].asfactor() #y_test was stacked as 'C785'

# 15. Make prediction on X_test

result = dl_model.predict(X_test[: , 0:784])

result.shape       # 10000 X 11

result.as_data_frame().head()   # Class-wise predictions

# 16.1 Ground truth

#      Convert H2O frame back to pandas dataframe

xe = X_test['C785'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.columns      #['C785', 'result']

xe.head()       #See the results of predicted Vs actual labels

# 16.2 So compare ground truth with predicted

out = (xe['result'] == xe['C785'])

np.sum(out)/out.size    # Accuracy ~ 77.15%

# 17.1 Also create confusion matrix using pandas dataframe

f  = confusion_matrix( xe['C785'], xe['result'] )

f.shape     # 10x10 matrix

#17.2 The Matrix of confusion

f

#17.3 How were the classes distributed

xe['C785'].value_counts()   # Classes are equitably distributed

#17.4 How are the predictions distributed

xe['result'].value_counts() # skewed towards classes 4,8,9, 1

# 18.1 calculate the prediction propbabilities for all thresholds of the classification

pred_probability = result["p1"].as_data_frame()    #  Get probability values and convert to pandas dataframe

pred_probability
#  18.2 Which columns are of higher importance as per the trained and tested model

var_df = pd.DataFrame(dl_model.varimp(),

             columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])

var_df.head(10)
