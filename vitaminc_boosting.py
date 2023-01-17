import pandas as pd

import numpy as np

import datetime as dt

import pandas_profiling



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce



from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

import seaborn as sns
LOCAL = '../input/'



train_features = pd.read_csv(LOCAL +'train_features.csv')

train_labels = pd.read_csv(LOCAL + 'train_labels.csv')

test_features = pd.read_csv(LOCAL + 'test_features.csv')

sample_submission = pd.read_csv(LOCAL + 'sample_submission.csv')



assert train_features.shape == (59400, 40)

assert train_labels.shape == (59400, 2)

assert test_features.shape == (14358, 40)

assert sample_submission.shape == (14358, 2)
train_features.profile_report()
print("Initial search space of our training data is {}".format(train_features.shape[0] * train_features.shape[1]))
train_labels['status_group'].value_counts()
# extracting numerical column names and categorical column names from the feature columns

num_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()

cat_cols = train_features.select_dtypes(exclude=[np.number]).columns.tolist()



# Sanity check to make sure the number of numerical features and categorical features

# add up to the total number of features

assert (len(num_cols) + len(cat_cols)) == train_features.shape[1]



# from pandas profiling we know permit and public_meeting are boolean features.  And we make a list of rejected to reject.

bool_cols = ['permit', 'public_meeting']

rejected = ['quantity_group', 'recorded_by']



# print out numerical features and categorical features

print(num_cols)

print(cat_cols)
print("{} numerical features and {} categorical features have missing values, of which {} are boolean features".format(

    train_features[num_cols].isnull().any().sum(), 

    train_features[cat_cols].isnull().any().sum(),

    train_features[bool_cols].isnull().any().sum()))
def preproc(df: pd.DataFrame) -> pd.DataFrame:

    """

    Naive preprocessing the input data by fixing missing values and dropping rejected features.

    Boolean features and categorical features are processed differently.

    

    Parameters

    ----------

    df : pandas.DataFrame

    

    Returns

    ----------

    df : pandas.DataFrame

    """

    

    # drop the rejected columns first.

    df = df.drop(columns=rejected)

    

    # since boolean features are included in categorical features, lets fillna of those first.

    # we will prudently assume nan/missing permits or public_meetings as False. 

    df[bool_cols] = df[bool_cols].fillna(False)

    

    # with boolean part of the categorical features fixed, we can proceed to fill missing values.

    # only categorial features has missing values, so we can just fillna directly on the whole df

    df = df.fillna('unknown')

    

    # date_recorded is in datetime format. We need to change that into numbers for the algorithm

    # to work. Let's change the definition of date_recorded to number of days since its recorded

    # between then and today. 

    today = dt.datetime.today()   

    df.date_recorded = pd.to_datetime(df.date_recorded,format = '%Y-%m-%d')

    df.date_recorded = (df.date_recorded - today).dt.days.abs()

        

    return df
X = preproc(train_features)

y = train_labels.status_group

print("search space is now {}".format(X.shape[0] * X.shape[1]))
logreg = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500)

encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')

scaler = StandardScaler(with_mean=False)



pipe = Pipeline(steps=[('encoder', encoder),

                       ('scaler', scaler),

                       ('logreg', logreg)

                       ])
%time pipe.fit(X, y);

pipe.score(X, y)
pipe.score(X, y)
X_FI = X[num_cols+['date_recorded']]

X_FI.columns

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



%time forest.fit(X_FI, y)

importances = forest.feature_importances_



# standard deviation of importance of each features

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)



# sort the importances by rank

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_FI.shape[1]):

    print("%d. %s, feature #%d (%f)" % (f + 1, X_FI.columns[indices[f]], indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_FI.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_FI.shape[1]), indices)

plt.xlim([-1, X_FI.shape[1]])

plt.show()
sns.relplot(x='longitude', y='latitude', hue='gps_height',data=train_features, alpha=0.1);
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:

    """

    Detailed cleaning of numeric dataframes. This function first change the 0s to np.nan and fill

    iteratively fillna with the means of all categorical filters down to only using the means

    of the least granular categorical feature, 'basin'. This function than also fix construction 

    year by changing the 0s to base year 1960.

    

    We elect to keep the original id, region_code, district_code, and num_private.  Also, this 

    function does not changing anything to amount_tsh since its not clear what that represents.  

    

    amount_tsh : Total static head (amount water available to waterpoint)            

    

    Parameters

    ----------

    df : pandas.DataFrame

    

    Returns

    ----------

    df : pandas.DataFrame

    """

    

    # construct overall location keys. The list is ordered from lowest cardinality to the highest so

    # we can iterate fillna from the most granular means to lowest cardinality region codes.    

    loc_keys = ['basin', 'region_code', 'district_code', 'lga', 'ward']

    

    # amount_tsh : Total static head (amount water available to waterpoint)        

    # Not changing anything to amount_tsh since its not clear what that represents.  Also not changing

    # district_code, region_code, id, and num_private.

    

    df.longitude = df.longitude.replace(0, np.nan)

    df.latitude = df.latitude.replace(-2.000000e-08, np.nan)

    df.gps_height = df.gps_height.replace(0, np.nan)

    df.population = df.population.replace(0, np.nan)

    

    # Iteratively fillna with the means from the most granular area split to the most overall area split.

    # https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group

    for i in range(len(loc_keys), 0, -1):

        df['longitude'] = df['longitude'].fillna(df.groupby(loc_keys[:i])['longitude'].transform('mean'))

        df['latitude'] = df['latitude'].fillna(df.groupby(loc_keys[:i])['latitude'].transform('mean'))

        df['gps_height'] = df['gps_height'].fillna(df.groupby(loc_keys[:i])['gps_height'].transform('mean'))

        df['population'] = df['population'].fillna(df.groupby(loc_keys[:i])['population'].transform('mean'))

    

    # construction years has a lot of 0s. Outside of 0s, it starts at 1960. We will make the

    # assumption that construction year for all the 0s/unkowns as the base year 1960.

    df['construction_year'] = df['construction_year'].apply(lambda x: 1960 if x == 0 else x)

    

    return df
X = preproc(train_features)

X = clean_numeric(X)

y = train_labels.status_group

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
sns.relplot(x='longitude', y='latitude', hue='gps_height',data=X, alpha=0.1);
categorical_counts = X_train.select_dtypes(exclude='number').nunique().sort_values()

categorical_counts.plot.bar();

high_cardi_cols = categorical_counts[categorical_counts > 150].index.tolist()

low_cardi_cols = categorical_counts[categorical_counts <= 150].index.tolist()
def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:

    """

    Detailed cleaning of numeric dataframes. We change everything categorical labels to lower case

    features. We also decide to collapse all unknown with other.

    

    Parameters

    ----------

    df : pandas.DataFrame

    

    Returns

    ----------

    df : pandas.DataFrame

    """

    

    # lower case everything

    cat_cols = df.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()

    df[cat_cols] = df[cat_cols].applymap(lambda x: x.lower())

    

    # since for some categories there are both unknown and other categories, we feel we can collapse

    # unknown with other since other can incoroporate unknown.

    df = df.replace({'unknown':'other'})

    

    return df
X = preproc(train_features)

X = clean_numeric(X)

X = clean_categorical(X)

y = train_labels.status_group

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=45, stratify=y)
logreg = LogisticRegression(solver='lbfgs', multi_class='ovr', random_state=45, max_iter=2000, C=100000.0)

encoder = ce.OrdinalEncoder()#(use_cat_names=True)

scaler = StandardScaler(with_mean=False)



lr_pipe = Pipeline(steps=[('encoder', encoder),

                          ('scaler', scaler),

                          ('logreg', logreg)

                         ])
%time lr_pipe.fit(X_train, y_train)

lr_pipe.score(X_train, y_train)
lr_pipe.score(X_val, y_val)
X = preproc(train_features)

X = clean_numeric(X)

X = clean_categorical(X)

y = train_labels.status_group

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=45, stratify=y)
xgbc = XGBClassifier(random_state=45, objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 

                      num_class = 3, maximize = False, eval_metric = 'merror', eta = .1, verbosity=1,

                      max_depth = 20, colsample_bytree = .4)

encoder = ce.OrdinalEncoder()



xgbc_pipe = Pipeline(steps=[('encoder', encoder),

                            ('xgbc', xgbc)])
%time xgbc_pipe.fit(X_train, y_train)

xgbc_pipe.score(X_train, y_train)
xgbc_pipe.score(X_val, y_val)