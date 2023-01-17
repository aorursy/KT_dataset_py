import numpy as np

import pandas as pd



import time

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error
# Read in data

X = pd.read_csv('../input/hdb-resale-price-prediction/train.csv', index_col='id')

X_test_full = pd.read_csv('../input/hdb-resale-price-prediction/test.csv', index_col='id')

df_add_coords = pd.read_csv('../input/testing-google-maps-geocoding-api/address_coordinates.csv')

df_distance = pd.read_csv('../input/distance-from-hdb-block-address-to-nearest-station/address_to_nearest_stn_dist.csv')

# train_indices = X.index

# test_indices = X_test_full.index



# Replace C'WEALTH with COMMONWEALTH

X['street_name'] = X['street_name'].replace('C\'WEALTH', 'COMMONWEALTH', regex=True)

X_test_full['street_name'] = X_test_full['street_name'].replace('C\'WEALTH', 'COMMONWEALTH', regex=True)



# Add address column to X and X_test_full

X['address'] = X['block'] + ' ' + X['street_name']

X_test_full['address'] = X_test_full['block'] + ' ' + X_test_full['street_name']



# Add lat and long to X and X_test_full

X = X.reset_index().merge(df_add_coords, on='address').set_index('id')

X = X.reset_index().merge(df_distance, on='address').set_index('id')

X_test_full = X_test_full.reset_index().merge(df_add_coords, on='address').set_index('id')

X_test_full = X_test_full.reset_index().merge(df_distance, on='address').set_index('id')



# Define dictionaries for changing labels of categorical variables

dictionaries = [

    {

        'name': 'num_rooms',

        'features': ['flat_type'],

        'lookup': {

            '1 ROOM': 'type1',

            '2 ROOM': 'type2',

            '3 ROOM': 'type3',

            '4 ROOM': 'type4',

            '5 ROOM': 'type5',

            'EXECUTIVE': 'type6',

            'MULTI-GENERATION': 'type7'

        }

    }

]





for df in [X, X_test_full]:

    

    # Change flat_type

    for dictionary in dictionaries:

        for feature in dictionary["features"]:

            df[feature] = df[feature].map(dictionary["lookup"])

    

    # Convert remaining lease years and storey range to numeric format

    df['remaining_lease_years'] = pd.to_numeric(df['remaining_lease'].apply(lambda x: x.split(" ")[0]))

    df['storey_range_numerical'] = pd.to_numeric(df['storey_range'].apply(lambda x: x.split(" ")[0]))

    df.drop(['remaining_lease','storey_range'], axis=1, inplace=True)



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['resale_price'], inplace=True)

y = np.log1p(X.resale_price)              

X.drop(['resale_price'], axis=1, inplace=True)



# Train-val split

X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                            random_state=4)



# Select categorical columns

categorical_cols = ['flat_model', 'flat_type']



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns

                  if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

selected_cols = categorical_cols + numerical_cols

X_train = X_train_full[selected_cols].copy()

X_val = X_val_full[selected_cols].copy()

X_test  = X_test_full[selected_cols].copy()
# Preprocessing for numerical data

numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scale',   StandardScaler(with_mean=False))

])



# Preprocessing for categorical data (drop first class of each feature during one hot encoding)

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot',  OneHotEncoder(handle_unknown='ignore')),

    ('scale',   StandardScaler(with_mean=False))

])



# Bundle both preprocessing

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer,   numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
metric = lambda y1_real, y2_real: np.sqrt(mean_squared_error(y1_real, y2_real))

y_trfm = lambda y: np.expm1(y)



def get_score(model, X, y):

    # Predict

    preds = model.predict(X)

    # Transform

    preds  = y_trfm(preds)

    y = y_trfm(y)

    return metric(preds, y)
# Build pipeline

model = GradientBoostingRegressor(

    n_estimators=1000,

    learning_rate=0.02,

    random_state=42

    )

pipeline_gbr = Pipeline(steps=[('preprocessor', preprocessor),

                               ('model', model)])



params = {

    'model__subsample': [0.5, 0.75, 1.0],

    'model__max_features': [0.4, 0.6, 0.8, 1.0],

    'model__max_depth': [4, 6, 8, 10]

}

gbr = GridSearchCV(pipeline_gbr, params, cv=5, scoring='neg_mean_squared_error',

                   n_jobs=-2)

time_start = time.time()

gbr.fit(X_train, y_train)

print('Time taken for hyperparameter tuning: {:.2f} min'.

      format((time.time() - time_start) / 60))

get_score(gbr, X_val, y_val)
# Show selected params

gbr.best_params_
preds_test = gbr.predict(X_test)

preds_test = y_trfm(preds_test)



output = pd.DataFrame({'id': X_test.index,

                       'resale_price': preds_test})

output.to_csv('submission.csv', index=False)