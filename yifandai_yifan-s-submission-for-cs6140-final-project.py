# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
pos_cash_balance = pd.read_csv('../input/POS_CASH_balance.csv')

previous_app = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
def agg_numeric(df, group_var, df_name):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg.head()
app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED'] / app_train['DAYS_BIRTH']
app_train.shape
app_train = app_train.join(bureau_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
app_train = app_train.rename(index=str, columns={"SK_ID_CURR1": "SK_ID_CURR"})
app_train
app_test['CREDIT_INCOME_PERCENT'] = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']
app_test['ANNUITY_INCOME_PERCENT'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']
app_test['CREDIT_TERM'] = app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']
app_test['DAYS_EMPLOYED_PERCENT'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']
app_test = app_test.join(bureau_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
app_test = app_test.rename(index=str, columns={"SK_ID_CURR1": "SK_ID_CURR"})

def normalize_categorical(df, group_var, col_name):
    
    """Computes counts and normalized counts for each observation
    of `group_var` for each unique category in every categorical variable
    
    Parameters 
    ----------
    df - DataFrame for which we will calculate count
    
    group_var  = string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    col_name = string
            Variable added to the front of column names to keep track of columns
            
            """
    # select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])                                              
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (col_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical
bureau_counts = normalize_categorical(bureau, group_var = 'SK_ID_CURR', col_name = 'bureau')
bureau_counts.head()
data_bureau_agg=bureau.groupby(by='SK_ID_CURR').mean()
data_credit_card_balance_agg=credit_card_balance.groupby(by='SK_ID_CURR').mean()
data_previous_application_agg=previous_app.groupby(by='SK_ID_CURR').mean()
data_installments_payments_agg=installments_payments.groupby(by='SK_ID_CURR').mean()
data_POS_CASH_balance_agg=pos_cash_balance.groupby(by='SK_ID_CURR').mean()

def merge(df):
    df = df.join(data_bureau_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    df = df.join(bureau_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
    df = df.join(data_credit_card_balance_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')    
    df = df.join(data_previous_application_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2')   
    df = df.join(data_installments_payments_agg, how='left', on='SK_ID_CURR', lsuffix='1', rsuffix='2') 
    
    return df

train = merge(app_train)
test = merge(app_test)
display(train.head())
#combining the data
ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train.TARGET.values

#train_df = train_df.drop

all_data = pd.concat([train, test]).reset_index(drop=True)
all_data.drop(['TARGET'], axis=1, inplace=True)
# Now we will convert days employed and days registration and days id publish to a positive no. 
def correct_birth(df):
    
    df['DAYS_BIRTH'] = round((df['DAYS_BIRTH'] * (-1))/365)
    return df

def convert_abs(df):
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
    df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
    df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])
    df['DAYS_LAST_PHONE_CHANGE'] = abs(df['DAYS_LAST_PHONE_CHANGE'])
    return df

# Now we will fill misisng values in OWN_CAR_AGE. 
#Most probably there will be missing values if the person does not own a car. So we will fill with 0

def missing(df):
    
    features = ['previous_loan_counts','NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI','OWN_CAR_AGE']
    
    for f in features:
        df[f] = df[f].fillna(0 )
    return df

def transform_app(df):
    df = correct_birth(df)
    df = convert_abs(df)
    df = missing(df)
    return df

   

all_data = transform_app(all_data)

# counting no of phones given by the company and delete the irrelevant features
all_data['NO_OF_CLIENT_PHONES'] = all_data['FLAG_MOBIL'] + all_data['FLAG_EMP_PHONE'] + all_data['FLAG_WORK_PHONE']
all_data.head()
# add a feature to determine if client's permanent city does not match with contact/work city
all_data['FLAG_CLIENT_OUTSIDE_CITY'] = np.where((all_data['REG_CITY_NOT_WORK_CITY']==1) & (all_data['REG_CITY_NOT_LIVE_CITY']==1),1,0)
all_data.head()
 # add a feature to determine if client's permanent city does not match with contact/work region
all_data['FLAG_CLIENT_OUTSIDE_REGION'] = np.where((all_data['REG_REGION_NOT_LIVE_REGION']==1) & (all_data['REG_REGION_NOT_WORK_REGION']==1),1,0)
all_data.head()
# deleting useless features
def delete(df):
   # useless=['FLAG_MOBIL', 'FLAG_EMP_PHONE' ,'FLAG_WORK_PHONE','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION']
    #for feature in useless:
     return df.drop(['FLAG_MOBIL', 'FLAG_EMP_PHONE' ,'FLAG_WORK_PHONE','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION'], axis=1)
def transform(df):
   # df = convert_abs(df)
    df = delete(df)
   
    return df

all_data = transform(all_data)
all_data.head()
# delete Ids

def delete_id(df):
    return df.drop(['SK_ID_CURR', 'SK_ID_PREV','SK_ID_BUREAU'], axis = 1)

all_data = delete_id(all_data)
def miss_numerical(df):
    
    features = ['previous_loan_counts','NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_MEDI','OWN_CAR_AGE']
    numerical_features = all_data.select_dtypes(exclude = ["object"] ).columns
    #print(numerical_features)
    for f in numerical_features:
        #print(f)
        if f not in features:
            df[f] = df[f].fillna(df[f].median())
      
    return df

def miss_categorical(df):
    
    categorical_features = all_data.select_dtypes(include = ["object"]).columns
    
    for f in categorical_features:
        df[f] = df[f].fillna(df[f].mode()[0])
        
    return df

def transform_feature(df):
    df = miss_numerical(df)
    df = miss_categorical(df)
    #df = fill_cabin(df)
    return df

all_data = transform_feature(all_data)


all_data.head()
from sklearn.preprocessing import MinMaxScaler

def encoder(df):
    scaler = MinMaxScaler()
    numerical = all_data.select_dtypes(exclude = ["object"]).columns
    features_transform = pd.DataFrame(data= df)
    features_transform[numerical] = scaler.fit_transform(df[numerical])
    display(features_transform.head(n = 5))
    return df

all_data = encoder(all_data)
# Converting into categorical features

# Create a label encoder object
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le_count = 0


# Iterate through the columns
for col in all_data:
    if all_data[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(all_data[col].unique())) <= 2:
            # Train on the training data
            le.fit(all_data[col])
            # Transform both training and testing data
            all_data[col] = le.transform(all_data[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
           
print('%d columns were label encoded.' % le_count)
all_data = pd.get_dummies(all_data)
display(all_data.shape)
train = all_data[:ntrain]
test = all_data[ntrain:]

print("Training shape", train.shape)
print("Testing shape", test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train, y_train, test_size = 0.1, random_state = 100)
print("X Training shape", X_train.shape)
print("X Testing shape", X_test.shape)
print("Y Training shape", Y_train.shape)
print("Y Testing shape", Y_test.shape)
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0, class_weight='balanced', C=500)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict_proba(X_test)[:,1]

print('Train/Test split results:')
print("ROC",  roc_auc_score(Y_test, Y_pred))
pred_test = logreg.predict_proba(train)
submission = pd.DataFrame({'SK_ID_CURR' : app_train['SK_ID_CURR'], 
                           'TARGET' : pred_test[:,0]})
pd.DataFrame(submission, columns=['SK_ID_CURR','TARGET'],index=None).to_csv('homecreditada.csv')
