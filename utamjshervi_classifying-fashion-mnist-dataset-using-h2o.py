# 1.0 Call libraries

%reset -f

import pandas as pd

import numpy as np

import os

# 1.0.1 For measuring time elapsed

from time import time
#   import imblearn;  imblearn.__version__

from imblearn.over_sampling import SMOTE, ADASYN
# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct
# 1.3 Data imputation

from sklearn.impute import SimpleImputer
# 1.4 Model building

#     Install h2o as: conda install -c h2oai h2o=3.22.1.2

import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator
# 1.6 Change ipython options to display all data columns

pd.options.display.max_columns = 300
# 2.0 Read data

import os

print(os.listdir("../input"))

os.chdir("../input")

ed = pd.read_csv("fashion-mnist_train.csv")
# 2.1 Explore data

ed.head(3)

ed.info()                       # # NULLS in total_toilets, establishment_year
# 3.0 Separation into target/predictors

y = ed.iloc[:,0]

X = ed.iloc[:,1:]

X.shape  # 60000  X 784

# 4 Create a StandardScaler instance

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

se = StandardScaler()

# 4.2 fit() and transform() in one step

X = se.fit_transform(X)

# 4.3

X.shape               # 60000 X 784 (an ndarray)
# 6. Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y,

                                                    test_size = 0.3)

# 7

X_train.shape    # 43314 X 135  if no kmeans: (18000, 784)

X_test.shape     # 18564 X 135; if no kmeans: (18000, 784)

y_train.shape

y_test.shape
# 8  Process X_train data with SMOTE

sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_sample(X_train, y_train)

type(X_res)       # No longer pandas dataframe

                  #  but we will convert to H2o dataframe
# 9 Check

X_res.shape                    # 25480 X 19

np.sum(y_res)/len(y_res)       # 0.5 ,earlier ratio was 0.047

#  10    H2o requires composite data with both predictors

#      and target

y_res = y_res.reshape(len(y_res),1)

y_res

X = np.hstack((X_res,y_res))

X.shape            # 42800 X 785

# 11 Start h2o

h2o.init()
# 12 Transform data to h2o dataframe

df = h2o.H2OFrame(X)

len(df.columns)    # 785

df.shape           # 42800 X 785

df.columns
# 13. Get list of predictor column names and target column names

#     Column names are given by H2O when we converted array to

#     H2o dataframe

X_columns = df.columns[0:784]        # Only column names. No data

X_columns       # C1 to C785

y_columns = df.columns[784]

y_columns       # C785
# 14 For classification, target column must be factor

#      Required by h2o

df['C785'] = df['C785'].asfactor()
# 15. Build a deeplearning model on balanced data

#     http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html

dl_model = H2ODeepLearningEstimator(epochs=500,

                                    distribution = 'bernoulli',                 # Response has two levels

                                    missing_values_handling = "MeanImputation", # Not needed by us

                                    variable_importances=True,

                                    nfolds = 2,                           # CV folds

                                    fold_assignment = "Stratified",       # Each fold must be sampled carefully

                                    keep_cross_validation_predictions = True,  # For analysis

                                    balance_classes=False,                # SMOTE is not provided by h2o

                                    standardize = True,                   # z-score standardization

                                    activation = 'RectifierWithDropout',  # Default dropout is 0.5

                                    hidden = [32,32],                  # ## more hidden layers -> more complex interactions

                                    stopping_metric = 'logloss',

                                    loss = 'CrossEntropy')

# 16.1 Train our model

start = time()

dl_model.train(X_columns,

               y_columns,

               training_frame = df)





end = time()

(end - start)/60

# 17. Time to make predictions on actual unbalanced 'test' data

#     Create a composite X_test data before transformation to

#     H2o dataframe.

y_test = (y_test.values).reshape(len(y_test), 1)     # Needed to hstack

y_test.shape     # 18000,1

X_test.shape     # 18000,784

# 17.1 Column-wise stack now

X_test = np.hstack((X_test,y_test))         # cbind data

X_test.shape     # 18000,785

# 17,2 Transform X_test to h2o dataframe

X_test = h2o.H2OFrame(X_test)

X_test['C785'] = X_test['C785'].asfactor()
# 18. Make prediction on X_test

result = dl_model.predict(X_test[: , 0:784])

result.shape       # 18000 X 11

result.as_data_frame().head()   # Class-wise predictions

# 18.1 Ground truth

#      Convert H2O frame back to pandas dataframe

xe = X_test['C785'].as_data_frame()

xe['result'] = result[0].as_data_frame()

xe.head()

xe.columns
# 19. So compare ground truth with predicted

out = (xe['result'] == xe['C785'])

np.sum(out)/out.size # 82% of accuracy  