import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn import linear_model, ensemble

from sklearn.metrics import mean_squared_error, mean_absolute_error



import tensorflow as tf



from tqdm.notebook import tqdm



import os

from PIL import Image
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

df = pd.read_csv(base_path + 'train.csv')

# df.sample(5,  random_state=1)

df.head()
def get_weeks_passed(df):

    min_week_dict = df.groupby('Patient').min('Weeks')['Weeks'].to_dict()

    df['MinWeek'] =  df['Patient'].map(min_week_dict)

    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']

    return df
def get_baseline_FVC(df):

    _df = (

        df

        .loc[df.Weeks == df.MinWeek][['Patient','FVC']]

        .rename({'FVC': 'FirstFVC'}, axis=1)

        .groupby('Patient')

        .first()

#         .reset_index()

    )

    

    first_FVC_dict = _df.to_dict()['FirstFVC']

    df['FirstFVC'] =  df['Patient'].map(first_FVC_dict)

    

    return df


def calculate_height(row):

    if row['Sex'] == 'Male':

        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])

    else:

        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])

    

df = get_weeks_passed(df)

df = get_baseline_FVC(df)

df['Height'] = df.apply(calculate_height, axis=1)

df['FullFVC'] = df['FVC']/df['Percent']*100
df
# import the necessary Encoders & Transformers

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.compose import ColumnTransformer



# define which attributes shall not be transformed, are numeric or categorical

no_transform_attribs = ['Patient','FVC']

num_attribs = ['Percent', 'Age', 'WeeksPassed', 'FirstFVC','Height', 'Weeks', 'MinWeek', 'FullFVC']

cat_attribs = ['Sex', 'SmokingStatus']

from sklearn.base import BaseEstimator, TransformerMixin



class NoTransformer(BaseEstimator, TransformerMixin):

    """Passes through data without any change and is compatible with ColumnTransformer class"""

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X
## transform features into series



# create an instance of the ColumnTransformer

datawrangler = ColumnTransformer(([

     # the No-Transformer does not change the data and is applied to all no_transform_attribs 

     ('original', NoTransformer(), no_transform_attribs),

     # Apply StdScaler to the numerical attributes, here you can change to e.g. MinMaxScaler()   

     ('MinMax', MinMaxScaler(), num_attribs),

     # OneHotEncoder all categorical attributes.   

     ('cat_encoder', OneHotEncoder(), cat_attribs),

    ]))



transformed_data_series = []

transformed_data_series = datawrangler.fit_transform(df)
## put transformed series into dataframe



# get column names for non-categorical data

new_col_names = no_transform_attribs + num_attribs



# extract possible values from the fitted transformer

categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]

new_col_names += categorical_values



# create Dataframe based on the extracted Column-Names

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)

train_sklearn_df.head()
from sklearn.model_selection import train_test_split



csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']



X = train_sklearn_df[csv_features_list].astype(float)



y = train_sklearn_df[['FVC']].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.linear_model import HuberRegressor

from sklearn.ensemble import GradientBoostingRegressor
# # huber = HuberRegressor(max_iter=200)

# huber = GradientBoostingRegressor(random_state=0, loss='huber')

# huber.fit(X_train, y_train)

# preds = huber.predict(X_test)
# predict confidence # https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python-ab3899f992ed



# Set lower and upper quantile

LOWER_ALPHA = 0.25

UPPER_ALPHA = 0.75

# Each model has to be separate

lower_huber = GradientBoostingRegressor(loss="quantile",                   

                                        alpha=LOWER_ALPHA)

upper_huber = GradientBoostingRegressor(loss="quantile",

                                        alpha=UPPER_ALPHA)



# The mid model will use the default loss

mid_huber = GradientBoostingRegressor(loss="huber")



# Fit models

lower_huber.fit(X_train, y_train)

mid_huber.fit(X_train, y_train)

upper_huber.fit(X_train, y_train)



# Record actual values on test set

# preds = y_test





# Predict

preds_lower = lower_huber.predict(X_test)

preds_mid = mid_huber.predict(X_test)

preds_upper = upper_huber.predict(X_test)



preds = pd.DataFrame({'lower':preds_lower, 'mid':preds_mid, 'upper':preds_upper})

mse = mean_squared_error(

    y_test,

    preds['mid'],

    squared=False

)



mae = mean_absolute_error(

    y_test,

    preds['mid']

)



print('MSE Loss: {0:.2f}'.format(mse))

print('MAE Loss: {0:.2f}'.format(mae))
def competition_metric(trueFVC, predFVC, predSTD):

    clipSTD = np.clip(predSTD, 70 , 9e9)  

    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  

    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))

    



print(

    'Competition metric with variable confidence: ', 

    competition_metric(np.ravel(y_test.values), preds['mid'], preds['upper']-preds['lower']) 

)



print(

    'Competition metric with static confidence: ', 

    competition_metric(np.ravel(y_test.values), preds['mid'], 285) 

)
# save model weights



import _pickle as cPickle



with open('/kaggle/working/lower_huber.pkl', 'wb') as f:

    cPickle.dump(lower_huber, f)



with open('/kaggle/working/mid_huber.pkl', 'wb') as f:

    cPickle.dump(mid_huber, f)

    

with open('/kaggle/working/upper_huber.pkl', 'wb') as f:

    cPickle.dump(upper_huber, f)





with open('/kaggle/working/datawrangler.pkl', 'wb') as f:

    cPickle.dump(datawrangler, f)





class NoTransformer(BaseEstimator, TransformerMixin):

    """Passes through data without any change and is compatible with ColumnTransformer class"""

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X

        



def load_huber_models(lower_huber_path, mid_huber_path, upper_huber_path, datawrangler_path):

    

    '''

    function to load saved huber modles and saved datawrangler

    

    Param

    -----

    lower_huber_path: string

    mid_huber_path: string

    upper_huber_path: string

    datawrangler_path: string

    

    Return

    ------

    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1

    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'

    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9

    datawrangler: saved columntransformer from sklearn

    

    '''

    

    with open(lower_huber_path, 'rb') as f:

        lower_huber = cPickle.load(f)



    with open(mid_huber_path, 'rb') as f:

        mid_huber = cPickle.load(f)



    with open(upper_huber_path, 'rb') as f:

        upper_huber = cPickle.load(f)



    with open(datawrangler_path, 'rb') as f:

        datawrangler = cPickle.load(f)



    return lower_huber, mid_huber, upper_huber, datawrangler

    



def huber_predict(lower_huber, mid_huber, upper_huber, datawrangler, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_start=-12, week_end=134):



    '''

    function to predict FVC value and confidence

    

    Param

    -----

    lower_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.1

    mid_huber: saved GradientBoostRegressor from sklearn, with loss = 'huber'

    upper_huber: saved GradientBoostRegressor from sklearn, with loss = 'quantile', alpha = 0.9

    datawrangler: saved columntransformer from sklearn

    

    Week: integer 

        Number of weeks after CT Scan, can be negative

    FVC: integer

        Initial FVC measurement at above `week`

    Percent: integer

        Initial percentage measurement at above `week`

    Age: integer

        Age at above `week`

    Sex: string

        'Male' or 'Female'

    SmokingStatus: string

        'Currently smokes', 'Ex-smoker', or 'Never smoked'

    week_start: integer

        start week to predict

    week_end: interger

        end week to predict

    

    Return

    ------

    df: DataFrame

        DataFrame with user inputs, engineered features, and predictions in running weeks

    

    '''

    

    

    MinWeek, FirstFVC, FullFVC, Height = _engineer_feature(Week, FVC, Percent, Age, Sex)



    df = _create_df_with_running_weeks(Patient,

                                        Week,

                                        FVC,

                                        Percent,

                                        Age,

                                        Sex,

                                        SmokingStatus,

                                        MinWeek,

                                        FirstFVC,

                                        FullFVC,

                                        Height,

                                        week_start,

                                        week_end)



    df, df_transformed = _wrangle_data(df, datawrangler)

    

    df = _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber)

    

    return df

        

def _engineer_feature(Week, FVC, Percent, Age, Sex):

    '''

    function to calculate MinWeek, FullFVC, and Height from patient details

    

    '''



    MinWeek = min(Week, 0)

    FirstFVC = FVC

    FullFVC = (FVC/Percent)*100



    if Sex == 'Male':

        Height = FirstFVC / (27.63 - 0.112 * Age)

    else:

        Height = FirstFVC / (21.78 - 0.101 * Age)



    return MinWeek, FirstFVC, FullFVC, Height

    

def _create_df_with_running_weeks(Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus, MinWeek, FirstFVC, FullFVC, Height, week_start, week_end):



    '''

    function to put patient details, engineered features, and running list of weeks into DataFrame

    

    '''



    Weeks = list(range(week_start, week_end))

    df = pd.DataFrame({'Weeks':Weeks})

    df['Patient'] = Patient

    df['Sex'] = Sex

    df['Age'] = Age

    df['SmokingStatus'] = SmokingStatus

    df['MinWeek'] = MinWeek

    df['FirstFVC'] = FVC

    df['FullFVC'] = FullFVC

    df['Height'] = Height

    df['Percent'] = Percent

    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']

    df['FVC'] = 0 # dummy FVC, to be predicted



    return df



def _wrangle_data(df, datawrangler):

    '''

    function to transform patient details into suitable format for models' ingestion

    

    '''



    transformed_data_series = datawrangler.transform(df)



    ## put transformed series into dataframe



    # define which attributes shall not be transformed, are numeric or categorical

    no_transform_attribs = ['Patient','FVC']

    num_attribs = ['Percent', 'Age', 'WeeksPassed', 'FirstFVC','Height', 'Weeks', 'MinWeek', 'FullFVC']

    cat_attribs = ['Sex', 'SmokingStatus']



    # get column names for non-categorical data

    new_col_names = no_transform_attribs + num_attribs



    # extract possible values from the fitted transformer

    categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]

    new_col_names += categorical_values



    # create Dataframe based on the extracted Column-Names

    df_transformed = pd.DataFrame(transformed_data_series, columns=new_col_names)



    return df, df_transformed





def _get_predictions(df, df_transformed, lower_huber, mid_huber, upper_huber):

    

    '''

    function to predict lower, upper and mid FVC and confidence interval for patients

    

    '''

    csv_features_list = ['FullFVC','Age','Weeks','MinWeek','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']

    df_transformed = df_transformed[csv_features_list]



    preds_lower = lower_huber.predict(df_transformed)

    preds_mid = mid_huber.predict(df_transformed)

    preds_upper = upper_huber.predict(df_transformed)

    

    df['Lower'] = preds_lower

    df['Upper'] = preds_upper

    df['FVC'] = preds_mid

    df['Confidence'] = abs(preds_upper - preds_lower)



    return df
## Model inputs

Patient = 'Albert'

Age = 69

Sex = 'Female'

Week = -4 # Number of weeks after CT Scan, can be negative

FVC = 3000

Percent = 78

SmokingStatus = 'Never smoked'

image_folder = 'path/to/folder/containing/dcm/images'
# load saved models

lower_huber, mid_huber, upper_huber, datawrangler = load_huber_models('/kaggle/working/lower_huber.pkl',

                                                                    '/kaggle/working/mid_huber.pkl',

                                                                    '/kaggle/working/upper_huber.pkl',

                                                                    '/kaggle/working/datawrangler.pkl')



# generate prediction

df = huber_predict(lower_huber, mid_huber, upper_huber, datawrangler, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus)

df = df[['Patient','Weeks','FVC','Lower','Upper']]

df
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

df = pd.read_csv(base_path + 'test.csv')

df.tail(20)
# lower_huber, mid_huber, upper_huber, datawrangler = load_huber_models('/kaggle/working/lower_huber.pkl',

#                                                                     '/kaggle/working/mid_huber.pkl',

#                                                                     '/kaggle/working/upper_huber.pkl',

#                                                                     '/kaggle/working/datawrangler.pkl')







for i, row in df.iterrows():

    if i == 0:

        df_predict = huber_predict(lower_huber,

                            mid_huber,

                            upper_huber,

                            datawrangler,

                            row['Patient'],

                            row['Weeks'],

                            row['FVC'],

                            row['Percent'],

                            row['Age'],

                            row['Sex'],

                            row['SmokingStatus']

                           )

    else:

        df_interim = huber_predict(lower_huber,

                    mid_huber,

                    upper_huber,

                    datawrangler,

                    row['Patient'],

                    row['Weeks'],

                    row['FVC'],

                    row['Percent'],

                    row['Age'],

                    row['Sex'],

                    row['SmokingStatus']

                   )

    



        df_predict = df_predict.append(df_interim)

df_predict['Patient_Week'] = df_predict['Patient'] + '_' + df_predict['Weeks'].astype(str)

df_predict = df_predict.sort_values('Patient').sort_values('Weeks', ignore_index=True)
df_submission = df_predict[['Patient_Week', 'FVC']] #, 'Confidence'

df_submission['Confidence'] = 285

df_submission.to_csv('/kaggle/working/submission.csv', index=False)
df_predict[['Patient','FVC','Lower','Upper']]
# base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

# df_sub = pd.read_csv(base_path + 'sample_submission.csv')

# # df_submission[['Patient','Weeks']] = df_submission['Patient_Week'].str.split("_",expand=True,)

# # df = df.drop(['Weeks', 'FVC','Confidence'],axis=1)



# df_sub
## for single prediction

# df = huber_predict(lower_huber, mid_huber, upper_huber, Patient, Week, FVC, Percent, Age, Sex, SmokingStatus, week_range=101)



# df_submission = df.apply(lambda x: huber_predict(lower_huber,

#                                             mid_huber,

#                                             upper_huber,

#                                             x['Patient'],

#                                             x['Weeks'],

#                                             x['FVC'],

#                                             x['Percent'],

#                                             x['Age'],

#                                             x['Sex'],

#                                             x['SmokingStatus'],

#                                             week_range = 133

#                                            ), axis=1)