from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import pandas as pd

import numpy as np





import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg





from sklearn.decomposition import PCA

from sklearn import preprocessing

from sklearn import ensemble

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_error
# %%writefile utilityfunctions.py

# Utility Functions



import numpy as np 

import pandas as pd 



from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import FunctionTransformer



from sklearn.model_selection import cross_val_score



from sklearn.isotonic import IsotonicRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import NuSVR

from sklearn.linear_model import HuberRegressor



from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.svm import LinearSVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor



import xgboost as xgb

import lightgbm as lgb



import time



import warnings

warnings.filterwarnings('ignore')



def summary(df):

    """

    Display data summary

    """

    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name', 'dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    display(summary)

    return summary





def stats(data_frame):

    """

    Collect stats about all columns of a dataframe, their types

    and descriptive statistics, return them in a Pandas DataFrame



    :param data_frame: the Pandas DataFrame to show statistics for

    :return: a new Pandas DataFrame with the statistics data for the

    given DataFrame.

    """

    stats_column_names = ('column', 'dtype', 'nan_cts', 'val_cts',

                          'min', 'max', 'mean', 'stdev', 'skew', 'kurtosis')

    stats_array = []

    for column_name in sorted(data_frame.columns):

        col = data_frame[column_name]

        if is_numeric_column(col):

            stats_array.append(

                [column_name, col.dtype, col.isna().sum(), len(col.value_counts()),

                 col.min(), col.max(), col.mean(), col.std(), col.skew(),

                 col.kurtosis()])

        else:

            stats_array.append(

                [column_name, col.dtype, col.isna().sum(), len(col.value_counts()),

                 0, 0, 0, 0, 0, 0])

    stats_df = pd.DataFrame(data=stats_array, columns=stats_column_names)

    return stats_df





def of_type(stats_data_frame, column_dtype):

    """

    Filter on columns of a given dtype ('object', 'int64', 'float64', etc)



    :param stats_data_frame: a DataFrame produced by the stats() function (above)

    :param column_dtype: a valid column dtype string ('object', 'int64', 'float64', ...)

    :return: the stats_data_frame that was passed in

    """

    return stats_data_frame[stats_data_frame['dtype'] == column_dtype]





def sort(data_frame, column_name, ascending=False):

    """

    Shorthand for sorting a data frame by one column's values.

    Useful with the status dataframe columns.



    :param data_frame: data_frame whose contents are to be sorted

    :param column_name: String name of the column to sort by

    :param ascending: if True, sort in ascending order (default, False)

    :return: a copy of the data_frame, sorted as specified

    """

    return data_frame.sort_values(column_name, ascending=ascending)





def is_numeric_column(df_column):

    """

    Answer whether a column of a data_frame is numeric



    :param df_column: Any column from a Pandas DataFrame

    :return: True if it's in one of the standard numeric types

    """

    numeric_types = (np.int16, np.float16, np.int32, np.float32,

                     np.int64, np.float64)

    return df_column.dtype in numeric_types



# Auto Regression

def log_transform(x):

    return np.log1p(x)



def inverse_log_transform(x):

    return np.expm1(x)





def getClassifiers():



    """

    Provide lists of regression classifiers and their names.

    """

    n_jobs       = -1

    random_state =  42



    classifiers = [

                   DummyRegressor(),

                   #IsotonicRegression(random_state=random_state),

                   KNeighborsRegressor(n_neighbors=7),

                   LinearRegression(n_jobs=n_jobs), 

                   Ridge(random_state=random_state), 

                   Lasso(random_state=random_state), 

                   ElasticNet(random_state=random_state),

                   KernelRidge(),

                   HuberRegressor(),

                   SGDRegressor(random_state=random_state),

                   SVR(kernel="linear"),

                   LinearSVR(random_state=1),

                   NuSVR(C=1.0, nu=0.1),

                   DecisionTreeRegressor(random_state=random_state),

                   RandomForestRegressor(n_jobs=n_jobs, random_state=random_state),

                   GradientBoostingRegressor(random_state=random_state),

                   lgb.LGBMRegressor(n_jobs=n_jobs, random_state=random_state),

                   xgb.XGBRegressor(objective="reg:squarederror", n_jobs=n_jobs, random_state=random_state),

    ]



    clf_names = [

                "DummyRegressor       ",

                #"IsotonicRegression   ",

                "KNeighborsRegressor  ",

                "LinearRegression     ", 

                "Ridge                ",

                "Lasso                ",

                "ElasticNet           ",

                "KernelRidge          ",

                "HuberRegressor       ",

                "SGDRegressor         ",

                "SVR                  ",

                "LinearSVR            ",

                "NuSVR                ",

                "DecisionTreeRegressor",

                "RandomForest         ", 

                "GBMRegressor         ", 

                "LGBMRegressor        ", 

                "XGBoostRegressor     ",

    ]



    return clf_names, classifiers







def prepareData(df, target_name):



    """

    Separate descriptive variables and target variable.

    Separate numerical and categorical columns.

    """



    if target_name is not None:

        X = df.drop(target_name, axis=1)

        y = df[target_name]

    else:

        X = df

        y = None



    # get list of numerical & categorical columns in order to process these separately in the pipeline 

    num_cols = X.select_dtypes("number").columns

    cat_cols = X.select_dtypes("object").columns

    

    return X, y, num_cols, cat_cols





def getPipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y):



    """

    Create Pipeline with a separate pipe for categorical and numerical data.

    Automatically impute missing values, scale and then one hot encode.

    """



    # the numeric transformer gets the numerical data acording to num_cols

    # first step: the imputer imputes all missing values to the provided strategy argument

    # second step: all numerical data gets stanadard scaled 

    if log_x == False:

        numeric_transformer = Pipeline(steps=[

            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),

            ('scaler', StandardScaler())])

    # if log_x is "True" than log transform feature values

    else:

        numeric_transformer = Pipeline(steps=[

            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),

            ('log_transform', FunctionTransformer(np.log1p)),

            ('scaler', StandardScaler()),

            ])

    

    # the categorical transformer gets all categorical data according to cat_cols

    # first step: imputing missing values

    # second step: one hot encoding all categoricals

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    

    # the column transformer creates one Pipeline for categorical and numerical data each

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numeric_transformer, num_cols),

            ('cat', categorical_transformer, cat_cols)])

    

    # return the whole pipeline for the classifier provided in the function call

    if log_y == False:

        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    # if log_y is "True" than use a TransformedTargetRegressor with log and inverse log functions for "y"

    else:

        transformed_classifier = TransformedTargetRegressor(regressor=classifier, 

            func=log_transform, inverse_func=inverse_log_transform)

        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', transformed_classifier)])





def scoreModels(df, target_name, sample_size=None, 

    impute_strategy="mean", scoring_metric="r2", log_x=False, log_y=False, verbose=True):



    """

    This function yields error scores for a large variety of common regression classifiers on provided training data. 



    """



    

    if sample_size is not None:

        df = df.sample(sample_size)

  

    # retrieve X, y and separated columns names for numerical and categorical data

    X, y, num_cols, cat_cols = prepareData(df, target_name)



    scores = []



    clf_names, classifiers = getClassifiers()

    if verbose == True:

        print(f"Classifier             Metric ({scoring_metric})")

        print("-"*30)

    for clf_name, classifier in zip(clf_names, classifiers):

        start_time = time.time()

        

        # create a pipeline for each classifier

        clf = getPipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)

                

        # crossvalidate classifiers on training data

        cv_score = cross_val_score(clf, X, y, cv=3, scoring=scoring_metric)

        

        if verbose == True:

            print(f"{clf_name} {cv_score.mean(): .4f}  |  {(time.time() - start_time):.2f} secs")

        

        scores.append([clf_name.strip(), cv_score.mean()])



    scores = pd.DataFrame(scores, columns=["Classifier", scoring_metric]).sort_values(scoring_metric, ascending=False)

    

    # just for good measure: add the mean of all scores to dataframe

    scores.loc[len(scores) + 1, :] = ["mean_all", scores[scoring_metric].mean()]



    return scores.reset_index(drop=True)

    





def trainModels(df, target_name, 

    impute_strategy="mean", log_x=False, log_y=False, verbose=True): 



    """

    This function trains a large variety of common regression classifiers on provided training data. 

    

    """



    X, y, num_cols, cat_cols = prepareData(df, target_name)



    pipelines = []



    if verbose == True:

        print(f"Classifier            Training time")

        print("-"*35)

    

    clf_names, classifiers = getClassifiers()

    for clf_name, classifier in zip(clf_names, classifiers):

        start_time = time.time()

        clf = getPipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)

        clf.fit(X, y)

        if verbose == True:

            print(f"{clf_name}     {(time.time() - start_time):.2f} secs")

        pipelines.append(clf)

    

    return pipelines







def predictFromModels(df_test, pipelines):



    """

    This function makes predictions with a list of pipelines. 

    

    """

    

    X_test, _ , _, _ = prepareData(df_test, None)

    predictions = []

    

    for pipeline in pipelines:

        preds = pipeline.predict(X_test)

        predictions.append(preds)

        

    df_predictions = pd.DataFrame(predictions).T

    clf_names, _ = getClassifiers()

    df_predictions.columns = [clf_name.strip() for clf_name in clf_names]



    return df_predictions



print("Feed Data")

feed = pd.read_csv("../input/froth-flotation/feed.csv")

feed.shape

feed.head()

#summary(feed)



print("Floatation Data")

floatation = pd.read_csv("../input/froth-flotation/flotation.csv")

floatation.shape

floatation.head()

#summary(floatation)



print("Scoring Dataset")

test = pd.read_csv("../input/froth-flotation/scoringdataset.csv")

test.shape

test.head()

#summary(test)



print("Submission File")

submission = pd.read_csv("../input/froth-flotation/SubmissionFormat.csv")

submission.shape

submission.head()


data = pd.merge(feed, floatation, on='date')



data.head()

data.shape
test.columns



data.columns
df = data.drop(['Unnamed: 0_x', 'X1_x', 'Unnamed: 0_y', 'X1_y'], axis=1)
stats(df)
df.shape

test.shape
from tabulate import tabulate



scores = scoreModels(df, "% Silica Concentrate", sample_size=1000, verbose=False)

print()

print(tabulate(scores, showindex=False, floatfmt=".3f", headers="keys"))