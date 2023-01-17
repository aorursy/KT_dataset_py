"""

Last amended: 31st March, 2019

Myfolder: https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

Ref Kaggle:

   https://www.kaggle.com/c/statoil-iceberg-classifier-challenge



Good examples:

    https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide



Ref: PCA vs TruncatedSVD

   https://stats.stackexchange.com/questions/239481/difference-between-scikit-learn-implementations-of-pca-and-truncatedsvd

Evaluation: For each id in the test set, you

            must predict the probability that

            the image contains an iceberg or

            a ship (a number between 0 and 1).



Objectuves:

            1. Learn working of lightgbm

            2. lightgbm

            3. Singular Value Decomposition

            4. Cross-validation in python

            5. Learning curves

            6. Feature importance

            7. Bayesian optimization

            8. Bayesian optimization using skoptimize

               (For Bayesian optimization using hyperopt-sklearn

see folder 16.hyperopt. This method does not use

Gaussian Processes to search for next hyperparameter

point.)





Further study examples:

    https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide



"""
########################## A.Libraries ##################

## 1.0 Call needed libraries



#%reset -f              # Clear all variables

import gc

gc.collect()           # Garbage collection



# 1.1 Load pandas, numpy and matplotlib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#%matplotlib qt5    #getting error, need to check



# Troubleshooting and bug fixing

#import os;

#print(os.environ.get('QT_API'))





# Image manipulation

from skimage.io import imshow, imsave



# 1.2 Image normalizing and compression

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD



# 1.3 Libraries for splitting, cross-validation & measuring performance

from sklearn.model_selection import train_test_split

# 1.3.1 Return stratified folds. The folds are made by

#        preserving the percentage of samples for each class.

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import auc, roc_curve



# 1.4 ML - we will classify using lightgbm

#          with stratified cross validation

import lightgbm as lgb



# 1.5 OS related

import os, time



# 1.6 Bayes Optimization -- One method

#  Install as:

#       pip install bayesian-optimization

from bayes_opt import BayesianOptimization



# 1.7 Bayes optimization--IInd method

# SKOPT is a parameter-optimisation framewor

#  Install skopt as:

#       conda install -c conda-forge scikit-optimize

from skopt import BayesSearchCV

# 1.8 Set option to dislay many rows

pd.set_option('display.max_columns', 100)
## 2.0 Read files



train = pd.read_json("../input/train.json")
################### B. Understand data ##################

################  No data processing here #############





### Define some needed functions



# 3.0 Examine any dataset

#     ExamineData.__doc__  => Gives help

def ExamineData(x):

    """Prints various data charteristics, given x

    """

    print("Data shape:", x.shape)

    print("\nColumns:", x.columns)

    print("\nData types\n", x.dtypes)

    print("\nDescribe data\n", x.describe())

    print("\nData\n", x.head(2))

    print ("\nSize of data:", np.sum(x.memory_usage()))    # Get size of dataframes

    print("\nAre there any NULLS\n", np.sum(x.isnull()))

# 3.1 Let us understand train data

ExamineData(train)

# 3.2 Look at the first data-point

# 3.2.1

train.columns

train['band_1'].head(3)         # Each point in the Series is a list

train['band_1'][0]              # First data point

# 3.2.2

len(train['band_1'][0] )        # 5625 = 75 X 75 pixels

train['band_1'][0][:4]          # Look at first four pixel-values

### Transform Series into an nd-array

### And learning from this array



# 3.2.3

b= train['band_1'].values          # Get complete column (of data-points)

b                                #  is transformed to array of points

                                #    And each data-point in this array

                                #     is a list.



type(b)    # numpy.ndarray

# 3.2.4

b.shape    # So how many lists: 1604

# 3.2.5

j = 5                           # Get jth list

b[j]                            # Look at the jth list or data-point in array

b[j][:4]                        # Within the jth list (data-point), look at

                                #  Ist four pixel values

# 3.3 How many values this data-point has

len(train['band_1'][0])     # 5625 : 75 X 75

# 3.4 Plot the image contained at the first data-point

Ist_point = b[0]

g = np.array(Ist_point).reshape(75,75)

imshow(g)

plt.show()

################### C. Define useful functions ##################

################  No data processing here #############





# 4.0 Expand each data-point of band_1/band_2 into

#     an array of 5625 points

#     For 1604 rows, we get a matrix of size

#     1604 X 5625



def ExpandBandDataPoints(df, colname):

    """df: A dataframe

       colname: Each colvalue is a 'list' of

                img-data.

                Extract this img-data into an array

                and return the matrix

    """

    # 4.1 An array of lists. Each point in the array

    #     is a list

    b =   df[colname].values

    # 4.2 Create a zero-filled-matrix of size

    #     1604 X 5625 images

    #     Each row of matrix will hold one image

    #     from the band

    x = np.zeros((1604, 5625))  # 1604 X 5625

    # 4.3 For every row-point

    for j in range(1604):

        # 4.4 For every jth list in 'b' ie b[j]

        #     Populate jth row of x, x[j,],

        #     with all elements of list b[j]

        jth_list = b[j]    # jth point of array is a list

        #    Try:  y=[1,2,3] ; np.array(y)

        x[j, :] = np.array(jth_list)

    return(x)

# 5.0 Given a matrix of 1605 X 5625 from above function,

#     plot first six images after reshaping each row

#     (or flattened) array to 75 X 75. That is plot

#     x[0], x[1], ...x[5] after reshaping each

#     to 75 X 75

def PlotImages(x):

    """Given a flattened image matrix

       Prints six images after reshaping

    """

    # 5.1 Create figure-window and axes

    _, ax = plt.subplots(nrows = 2, ncols= 3)

    # 5.2

    ax[0,0].imshow(x[0, :].reshape(75,75))

    ax[0,1].imshow(x[1, :].reshape(75,75))

    ax[0,2].imshow(x[2, :].reshape(75,75))

    ax[1,0].imshow(x[3, :].reshape(75,75))

    ax[1,1].imshow(x[4, :].reshape(75,75))

    ax[1,2].imshow(x[5, :].reshape(75,75))

    plt.show()

# 6.0 Save an one image file to disk just to look at its size

def SaveOneImg(x, filename):

    """x is matrix of flattened images

       Each image is one row

       There are as many images as there are rows

       Saves just one image in the very first row

       Returns saved file size

    """

    # 6.1 Min and Max values in this data

    lower = np.min(x[0,:])

    upper = np.max(x[0,:])

    # 6.2 Range of values

    range = upper - lower

    # 6.3 Normalize now

    trans = (x[0,:] - lower)/range           # Normalize image to values [0,1]

    imsave(filename, trans.reshape(75,75))   # Reshape image and save it to disk

    return os.path.getsize(filename)         # Return img size on disk

# 7.0 Singular value decomposition of bands

#     of train/test data

#     See SVD vs PCA below

def SingularValueDecomp(x, n_comp):

    """x is a matrix of flattened images

       returned from ExpandBandDataPoints().

       n_comp: Number of SVD components

       This function returns transformed matrix,

       transformation object and explained variance

    """

    # 7.1 Create an object to perform SVD

    svd = TruncatedSVD(n_components = n_comp)

    # 7.2 Fit and transform

    g = svd.fit_transform(x)

    # 7.3 How much variance is explained per-component

    ev1 = svd.explained_variance_ratio_

    # Return a tuple of three values

    return (g, svd, ev1)
################### D. Data processing ################



# 9.0 There are some 'na' values in 'inc_angle' column

#     We will fill these up with mean value. Note

#     that the word 'na' is string not np.nan



# 9.1 Following is a pandas Series with boolean values

train['inc_angle'] == 'na'





# 9.2 How many such rows are there?

np.sum(train['inc_angle'] == 'na')   # 133 only





# 9.3 OK. Get now the mean of non-na rows

m_value = np.mean(train.loc[train['inc_angle'] != 'na', 'inc_angle' ])

m_value       # 39.26870747790618



# 9.4 Replace 'na' with mean-value

train.loc[train['inc_angle'] == 'na', 'inc_angle' ] = m_value



# 9.5 Finally check

train['inc_angle']





# 9.6 Create a categorical variable from inc_angle

train['c_angle'] = 1

train.loc[train['inc_angle'] > m_value, 'c_angle'] = 2

################### E. Pre Analyses  ##################





### Our steps in Modeling

##  Brief steps: ################

#   1. Reshape single column, band_1, into an array 5625 columns

#   2. Reshape single column, band_2, into 5625 columns

#   3. Truncate 5625 cols of band_1 to 25 cols using SVD

#   4. Truncate 5625 cols of band_2 to 25 cols using SVD

#   5. Concatenate two expanded column. Result: 50 cols

#   6. Cut inc_angle into two, to create a categorical

#      variable having two categories

#   7. To (5) above, concatenate two morecolumns:

#      One: of 'inc_angle' and Two: categorical column

#      created in step (6)

#   8. Do modeling

#   9. While modeling tune parameters of model using Bayes

#      optimization technique

################





# 10.0 Examine train/test data

ExamineData(train)    # 1604 X 5   155MB

# 10.1

#ExamineData(test)     # 8424 X 5   817MB

## 11. Popualte a variable with flattened images

#      from band_1.

#      Band 1 and Band 2 are signals characterized

#      by radar backscatter produced from different

#      polarizations at a particular incidence angle.

band1 = ExpandBandDataPoints(train, "band_1")

band1.shape           # 1604 X 5625

# 12 Also plot six of the satellite images

PlotImages(band1)

## 13. Popualte another variable with flattened images from band_2

band2 = ExpandBandDataPoints(train, "band_2")

band2.shape



# 13.1 Also plot six images from it

PlotImages(band2)

## 14. Save one image from band1 into a file

#      Returned value is file size

filesize = SaveOneImg(band1, "band1.jpg")

filesize                # 1695 bytes



# 15. Perform Singular Value Decomposition

#    Get how much variance is explained per-component

#    in band1. Start with total of 500 components

_,_,ev1 = SingularValueDecomp(band1, 500)

_,_,ev2 = SingularValueDecomp(band2, 500)

# 16. Plot the cumulative sum of explained variances

#     component-wise for both the bands

plt.figure()

frac_band1 = np.cumsum(ev1)

frac_band2 = np.cumsum(ev2)

plt.plot(frac_band1[:200])         # 25 components appear OK

plt.plot(frac_band2[:200])         # 25 components appear OK

plt.show()
# 17. Next, let us , therefore, truncate to 25 components

#     Singular value decomposition of two bands of train data

#      So new flattened images are in comp1 and in comp2

comp1,svd1, _ = SingularValueDecomp(band1,25)

comp2,svd2, _ = SingularValueDecomp(band2, 25)

# 18. We can perform inverse_transform with just 25 svd components

#     and relook at images (75 X 75) to visually see how much of

#     information relating to original image does exist

im1 = svd1.inverse_transform(comp1)

im2 = svd2.inverse_transform(comp2)

im1.shape         # Shape remains as 1604 X 5625, as earlier

im2.shape
# 19. Now, relook at transformed and original images

#     Transformed first

PlotImages(im1)

# 19.1 Compare above with original. Not bad..

PlotImages(band1)
## 20. Does SVD compress image?

##     Save transformed image

#      And compare its size with that of original image

#      saved earlier

##     Obviously SVD compresses image

SaveOneImg(im1, "im1.jpg")      # Size: 713 bytes as against 1695 bytes

################### F. Modeling  ##################



### 21. Finally we horizontally stack flattened images

X = np.hstack((comp1, comp2))

X.shape

X[0]     # Check Ist row

X[:2]    # Check Ist two rows





# 21.1 Also stack inc_angle

X.shape         # 1604 X 50

X = np.hstack((X, train.inc_angle.values.reshape(X.shape[0],1)))

X.shape          # (1604,51)

X = np.hstack((X, train.c_angle.values.reshape(X.shape[0],1)))

X.shape      # 1604 X 52                Predictors



X[0]         # Check again Ist row

X[:2]       # Check again Ist two rows



y = train['is_iceberg'].values         # Target





# 21.2 Partition into train/test

X_train, X_test, y_train, y_test = train_test_split(

                                     X, y,

                                     test_size=0.30,

                                     random_state=42

                                     )

###############################################################################

################# Bayesian optimization-I with skopt ##########################

# Refer:

# https://www.kaggle.com/nanomathias/bayesian-optimization-of-xgboost-lb-0-9769

# Refer for LGBMRegressor:

#   https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMRegressor

# And  https://lightgbm.readthedocs.io/en/latest/Parameters.html



# Our modeling object for Classification

# Can be any modeler: RandoForest, xgboost, neuralnetwork

#                     svm etc



# 22. Store train and test data in lightgbm,

#     Dataset object





model = lgb.LGBMRegressor(                # Regressor will also perform classification

                          objective='binary',

                          metric='auc',   # This output must match with what

                                          #  we specify as input to Bayesian model

                          n_jobs=2,

                          verbose=0

                          )

# 22.1 Parameter search space for selected modeler

params = {

        'num_leaves': (5,45),              # Maximum tree leaves for base learners.

        'feature_fraction': (0.1, 0.9),	   # Randomly select part of features on each iteration

        'bagging_fraction': (0.8, 1),		  # Randomly select part of data without resampling

        'max_depth': (1, 50),              # Maximum tree depth for base learners, -1 means no limit.

        'learning_rate': (0.01, 1.0, 'log-uniform'), # Prob of interval 1 to 10 is same as 10 to 100

                                                     # Equal prob of selection from 0.01 to 0.1, 0.1

                                                     # to 1

                                                     #  Boosting learning rate.

        'min_child_samples': (1, 50),         # Minimum number of data needed in a child (leaf)

        'max_bin': (100, 1000),               # max number of bins that feature

                                              #  values will be bucketed in

                                              # small number of bins may reduce

                                              # training accuracy but may increase

                                              # general power (deal with over-fitting)

        'subsample': (0.01, 1.0, 'uniform'),  # Subsample ratio of the training instance (default: 1)

        'subsample_freq': (0, 10),            #   Frequence of subsample, <=0 means no enable (default = 0).

        'colsample_bytree': (0.01, 1.0, 'uniform'), #  Subsample ratio of columns when constructing each tree (default:1).

        'min_child_weight': (0, 10),         # Minimum sum of instance weight (hessian) needed in a child (leaf).

        'subsample_for_bin': (100000, 500000), #  Number of samples for constructing bins(default: 200000)

        'reg_lambda': (1e-9, 1000, 'log-uniform'),  # L2 regularization term on weights.

        'reg_alpha': (1e-9, 1.0, 'log-uniform'),

        'scale_pos_weight': (1e-6, 500, 'log-uniform'), #used only in binary application

                                                        # weight of labels with positive class

        'n_estimators': (50, 100)  # Number of boosted trees to fit (default: 100).

        }

# 22.2 Cross validation strategy for the modeler

#      Perform startified k-fold cross-validation

#      There is also RepeatedStratifiedKFold() class

#      that will repeat startified k-fold N-number

#      of times

#      Instantiate cross-vlidation object

"""

Examples of Cross-validation strategies:

    i)   Leave one out  : Very time consuming

    ii)  Leave P out    : For example, leave 2 out

    iii) kfold          : k-equal random folds

    iv)  StratifiedKFold : kfolds + stratification

    v)   ShuffleSplit  => Generate n-numbers of userdefined pairs

                          of (train,test). For examples, in each

                          (train,test) pair, let number of rows

                          of 'test' data be 30% of train data



"""



cvStrategy = StratifiedKFold(

                             n_splits=2,

                             shuffle=True,

                             random_state=42

                            )



# 22.3 Bayesian object instantiation

#     For API, refer: https://scikit-optimize.github.io/#skopt.BayesSearchCV

bayes_cv_tuner = BayesSearchCV(

                              estimator = model,    # rf, lgb, xgb, nn etc--Black box

                              search_spaces = params,  # Specify params as required by the estimator

                              scoring = 'roc_auc',  # Input to Bayes function

                                                    # modeler should return this

                                                    # peformence metric

                              cv = cvStrategy,      # Optional. Determines the cross-validation splitting strategy.

                                                    #           Can be cross-validation generator or an iterable,

                                                    #           Possible inputs for cv are: - None, to use the default 3-fold cv,

                                                    #           - integer, to specify the number of folds in a (Stratified)KFold,

                                                    #           - An object to be used as a cross-validation generator.

                              n_jobs = 2,           # Start two parallel threads for processing

                              n_iter = 50,        # Reduce to save time

                              verbose = 1,

                              refit = True,       #  Refit the best estimator with the entire dataset

                              random_state = 42

                               )

# 22.4 Start learning using Bayes tuner

start = time.time()

result = bayes_cv_tuner.fit(

                           X_train,       # Note that we use normal train data

                           y_train       #  rather than lgb train-data matrix

                           #callback=status_print

                           )



end = time.time()

(end - start)/60



# 22.5 So what are the results?

#      Use the following estimator in future

bayes_cv_tuner.best_estimator_

# 22.6 What parameters the best estimator was using?

best_params = pd.Series(bayes_cv_tuner.best_params_)

best_params
# 22.7 Best auc score for the above estimator

np.round(bayes_cv_tuner.best_score_, 4)



# 22.8 Summary of all models developed by Bayes process

allModels_summary = pd.DataFrame(bayes_cv_tuner.cv_results_)

allModels_summary.shape  # 50 X 26

allModels_summary.head()

### 23. Let us now use the best estimator

bst_bayes = bayes_cv_tuner.best_estimator_

bst_bayes

# 23.1 Train the best estimator

bst_bayes.fit(X_train, y_train)

# 23.2 Make predictions

pred = bst_bayes.predict(X_test)

pred



# 23.3 So what is auc score

fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)

auc(fpr, tpr)    # 94%

############## Learning Curve with lightgbm #############

## What is a Learning Curve--See note at the end.



# Ref: https://lightgbm.readthedocs.io/en/latest/Python-API.html#training-api

# 24. Store train and test data in lightgbm,

#     Dataset object





d_train = lgb.Dataset(X_train, label=y_train) # transformed train data

d_test = lgb.Dataset(X_test, label = y_test)  # test data



# 24.1 Watch error in these datasets as

#      modeling proceeds

watchlist = [d_train, d_test]

## 25. Build Lightgbm model

# Set parameters first

# Ref: http://lightgbm.readthedocs.io/en/latest/Python-Intro.html

#      https://lightgbm.readthedocs.io/en/latest/Parameters.html



params = { 'learning_rate': 0.25,

           'verbosity': -1,             # Be verbose when processing

           'categorical_feature' : [51],  # which cols are categorical

                                          # (specify index 0,1,2..)

            'nthread': 4,                 # USe CPU cores

            'max_depth': 7,            # limit the max depth for tree model

            'objective' : "binary",

            'metric' : ['auc', 'binary_logloss']

           }

## 26

start = time.time()

# 26.1

evals_result = {} # to record eval results for plotting

model = lgb.train(params,

                  train_set=d_train,

                  num_boost_round=1500,     # 1000 residuals are mapped to functions

                                            #   successively

                  valid_sets=watchlist,

                  early_stopping_rounds=100, # The goal of early stopping is to

                                            #  decide if any of the latest X rounds

                                            #  has improved performance versus a baseline,

                                            #  according to some metric.

                                            # The model will train until the validation

                                            #  score stops improving. Validation error

                                            #  needs to improve at least every

                                            #  early_stopping_rounds to continue training.

                 evals_result=evals_result, # Record evaluation results for plotting

                 verbose_eval=10

                 )

# 26.2

end = time.time()

end - start
# 27. Plot learning curve

print('Plot metrics during training...')

ax = lgb.plot_metric(evals_result, metric='binary_logloss')

plt.show()
# 28. Plot precision/Recall curve

#  Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

from sklearn.metrics import precision_recall_curve as pr

y_pred = model.predict(X_test)

precision,recall,_ = pr(y_test,y_pred)   # Reruns a tuple of three arrays

                                         # precision, recall, thresholds

plt.plot(precision,recall)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.show()
######## FEATURE IMPORTANCE ###########



# 29

# Column wise imporatnce. Default Criteria: "split".

# "split":  Result contains numbers of times feature is used in a model.

# “gain”:   Result contains total information-gains of splits

#           which use the feature

print('Plot feature importances...')

ax = lgb.plot_importance(bst_bayes, max_num_features=10)

ax.tick_params(labelsize=20)

plt.show()



# 29.1 Does not work. Needs 'graphviz'

ax= lgb.plot_tree(bst_bayes,

                  tree_index=9,

                  figsize=(40, 20),

                  show_info=['split_gain'])



plt.show()
#################### Bayesian-optimization-II Normal method ###################

# Ref: https://github.com/fmfn/BayesianOptimization



# 25. Create lightgbm dataset, a binary file

#     LightGBM binary file

#     Also saving Dataset into a LightGBM binary file will make loading faster:

d_train = lgb.Dataset(X_train, label=y_train) # transformed train data

d_test = lgb.Dataset(X_test, label = y_test)  # test data

# 25.1

#  Step 1: Create a function that when passed some parameters

#          evaluates results using cross-validation

def lgb_eval(num_leaves,feature_fraction, bagging_fraction,max_depth):

    # Specify complete list of parameters: static and dynamic

    # 25.2 Static: Parameters that need not be modified

    params = {'application':'binary',

              'num_boost_round':4000,

              'learning_rate':0.05,

              'early_stopping_round':100,

              'metric':'auc',

              'shuffle':True

              }



    # 25.3 Dynamic: Parameters that would be passed using arguments to function

    params["num_leaves"] = int(round(num_leaves))

    params['feature_fraction'] = feature_fraction

    params['bagging_fraction'] = bagging_fraction

    params['max_depth'] = int(round(max_depth))



    # 25.4 Now evaluate with above parameters

    cv_result = lgb.cv(params,

                       d_train,

                       nfold=4,

                       seed=0,

                       stratified=True,

                       #verbose_eval =200,

                       metrics=['auc'])



    # 25.5 Finally return maximum value of result

    return max(cv_result['auc-mean'])





# 26. Step 2: Define BayesianOptimization function has

#             two arguments

lgbBO = BayesianOptimization(

                             lgb_eval,               # Which function will evaluate



                             # Parameters to tune and to be

                             #   passed to above function

                             # Specify parameter range for each

                             {'num_leaves': (24, 45),

                              'feature_fraction': (0.1, 0.9),

                              'bagging_fraction': (0.8, 1),

                              'max_depth': (5, 8.99)

                              }

                             )



# 26.1. Gaussian process parameters

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian

                                 # process

# 27. Step 3: Start optimization

start = time.time()

lgbBO.maximize(init_points=2,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=25         # Total number of times the

                                 #   process is to repeated

               )

end = time.time()

f"Minutes: {(end -start)/60}"

# 28. Get results

lgbBO.max

lgbBO.max['params']['max_depth']

lgbBO.max['params']['bagging_fraction']

lgbBO.max['params']['feature_fraction']

lgbBO.max['params']['num_leaves']

# 29. Newly discovered parameter values

params_new = lgbBO.max['params']

params_new['num_leaves'] = int(params_new['num_leaves']) + 1

params_new['max_depth'] = int(params_new['max_depth']) + 1





# 29.1 Objective alias 'application'

params_new['objective'] = ['binary']    # Default regression



# 29.2 Metric parameters

params_new['metric'] = ['auc', 'binary_logloss']   # Multiple loss parameters





# 29.3 Watch error in these datasets as

#      modeling proceeds

watchlist = [d_train, d_test]

# 30

start = time.time()

model = lgb.train(params_new,

                  train_set=d_train,

                  num_boost_round=1000,     # 1000 residuals are mapped to functions

                                            #   successively

                  valid_sets=watchlist,

                  early_stopping_rounds=20, # The goal of early stopping is to

                                            #  decide if any of the latest X rounds

                                            #  has improved performance versus a baseline,

                                            #  according to some metric.

                                            # The model will train until the validation

                                            #  score stops improving. Validation error

                                            #  needs to improve at least every

                                            #  early_stopping_rounds to continue training.

                  evals_result=evals_result, # Record evaluation results for plotting

                  verbose_eval=10)





end = time.time()

end - start

# 30.1

model.best_score



# 30.2 Save model to a text file for later use

model.save_model('model.txt',

                 num_iteration=model.best_iteration

                 )



# 30.3 Delete model

del model



# 30.4 Load back saved model

bst = lgb.Booster(model_file='model.txt')  #init model



# 30.5 If early stopping is enabled during training,

#      get predictions from the best iteration with

#      bst.best_iteration



lgb_pred = bst.predict(X_test,

                       num_iteration=bst.best_iteration)  > 0.5

lgb_pred

y_test



# 30.6 Now accuracy

np.sum(lgb_pred == y_test)/y_test.size

# 30.7 So what is auc score

fpr, tpr, thresholds = roc_curve(y_test, lgb_pred, pos_label=1)

auc(fpr, tpr)





#####################################################

################ SVD vs PCA ###################

"""

Ref:

   https://stats.stackexchange.com/a/87536/78454



SVD is slower than PCA but is often considered to be the preferred

method because of its higher numerical accuracy.

As you state in the question, principal component analysis (PCA) can

be carried out either by SVD of the centered data matrix X.



Matlab's help records this:

Principal component algorithm that pca uses to perform the

principal component analysis [...]:

[PCA uses two methods, svd and eigenvalue BUT unlike in SVD

PCA is done (on column-wise) centered data]







 i)    'svd' -- Default. Singular value decomposition (SVD) of X.

                         Slower but more accurate

 ii)   'eig' -- Eigenvalue decomposition (EIG) of the covariance matrix.

                The EIG algorithm is faster than SVD when the number of

                observations, n, exceeds the number of variables, p, but

                is less accurate because the condition number of the

                covariance is the square of the condition number of X.

                Faster but less accurate





"""



"""

Learning Curve--General definition

Ref: https://stackoverflow.com/a/13715276



    One line:  How a model performs as some hyperparameter is varied



    A learning curve conventionally depicts improvement in performance

    on the vertical axis when there are changes in another parameter

    (on the horizontal axis), such as training set size (in machine learning)

    or iteration/time (in both machine and biological learning). One salient

    point is that many parameters of the model are changing at different points

    on the plot.



    """

###############################
