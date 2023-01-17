!pip install xgboost
import datetime as dt

import calendar

import random



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras

import tensorflow as tf

from xgboost import XGBClassifier
# Load in data

# Email specs

df_train = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/train.csv', index_col='user_id', 

                       na_values=['Never checkout', 'Never open', 'Never login'])

# User specs

df_users = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/users.csv', index_col='user_id')

# Merge User and Email specs into one

df_train = pd.merge(df_train, df_users, how='left', left_index=True, right_index=True)
df_train.head()
# Process data

# Create month, day and time data columns

df_train['grass_date'] = pd.to_datetime(df_train.grass_date, format='%Y-%m-%d 00:00:00+08:00')

df_train['day'] = [calendar.day_name[i.weekday()] for i in df_train['grass_date'].to_list()] 

# Remove some columns

df_train.drop(columns=['grass_date', 'row_id'], inplace=True)
# Check for missing data.

df_train.isna().sum()
# First examine the distributions of 'open flag'

email_open_distribution = df_train.open_flag.value_counts()

ax = sns.barplot(x=email_open_distribution.index, y=email_open_distribution.values)
# Plot distribution of categorical vars using boxplots

# Columns containing categorical valuess

categorical_columns = ['country_code', 'attr_1', 'attr_2', 'attr_3', 'domain', 'day']

figure = plt.figure(figsize=(20, 10))

NUM_COLS = 2

NUM_ROWS = np.ceil(len(categorical_columns)/NUM_COLS)

NUM_PLOTS = 1 

for columns in categorical_columns:

    print('Plotting {}...'.format(columns))

    ax = plt.subplot(NUM_ROWS, NUM_COLS, NUM_PLOTS)

    sns.barplot(x=columns, y='open_flag', data=df_train)

    NUM_PLOTS = NUM_PLOTS + 1

plt.tight_layout()
# Plot distribution of continuous vars

# Columns containing continuous values

all_columns = df_train.columns

continuous_columns = list(set(all_columns).difference(set(categorical_columns)))

# Columns containing continuous values

figure = plt.figure(figsize=(20, 20))

NUM_COLS = 3

NUM_ROWS = np.ceil(len(continuous_columns)/NUM_COLS)

NUM_PLOTS = 1 

for column in continuous_columns:

    print('Plotting {}...'.format(column))

    ax = plt.subplot(NUM_ROWS, NUM_COLS, NUM_PLOTS)

    sns.boxplot(x='open_flag', y=column, data=df_train)

    ax.set_yscale('log')

    NUM_PLOTS = NUM_PLOTS + 1

plt.tight_layout()
def process_training(df_train, use_imputer=False):

    '''

    Process training data and seperate into train and test dataset

    '''

    df_current = df_train.copy()

    # Choose following columns as independent variables

#     chosen_columns = ['country_code', 'day', 'attr_1', 'attr_3', 'domain', 

#                       'last_open_day', 'open_count_last_60_days', 'open_count_last_10_days', 'open_count_last_30_days',

#                       'open_flag']

    chosen_columns = ['country_code', 'day', 'attr_3', 'domain', 

                      'last_open_day', 'open_count_last_60_days', 'open_count_last_10_days', 'open_count_last_30_days',

                      'open_flag']

    df_current = df_current[chosen_columns]



    # Downsample the majority to be the same as the minority (randomly), 

    # this is to rebalance the dataset

    # Copy an instance of the data

    df_data_for_sampling = df_current.copy()

    df_data_for_sampling.reset_index(inplace=True)

    # Random sampling

    sample_size_minority = df_data_for_sampling[df_data_for_sampling.open_flag==1].shape[0]

    sample_size_downsample_index = random.choices(df_data_for_sampling[df_data_for_sampling.open_flag==0].index, 

                                                  k=sample_size_minority)

    sample_size_downsample = df_data_for_sampling.loc[sample_size_downsample_index, :]

    # Combined data from both open_flag categories

    df_current = pd.concat([df_data_for_sampling[df_data_for_sampling.open_flag==1], sample_size_downsample])

    df_current.set_index('user_id', inplace=True)

    

    # For category var, change values to str

#     categorical_columns = ['country_code', 'day', 'attr_1', 'attr_3', 'domain']

    categorical_columns = ['country_code', 'day', 'attr_3', 'domain']

    for columns in categorical_columns:

        df_current[columns] = df_current[columns].astype('str')

    # OneHotEncoding

    df_onehot = pd.get_dummies(df_current[categorical_columns])

    df_current = pd.concat([df_current, df_onehot], axis=1)

    df_current.drop(columns=categorical_columns, inplace=True)

    

    # Dealing with NAs

    if use_imputer == True:

        # Use an imputer

        print('Imputing...')

        df_independent_var = df_current.drop(columns=['open_flag'])

        imputer = KNNImputer(n_neighbors=2)

        independent_var_imputed = imputer.fit_transform(df_independent_var)

        # Put in pandas df

        df_independent_var_imputed = pd.DataFrame(independent_var_imputed, 

                                                  index=df_independent_var.index,

                                                  columns=df_independent_var.columns)

        # Put back the target var

        df_independent_var_imputed['open_flag'] = df_current['open_flag']

        df_current = df_independent_var_imputed

    else:

        # Drop NA data

        df_current.dropna(inplace=True)

    

    # Seperate target from independent vars

    independent_vars = list(df_current.columns)

    independent_vars.remove('open_flag')

    X = df_current[independent_vars]

    y = df_current['open_flag']

    # Split data into train and test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

    return df_current, X_train, X_test, y_train, y_test



processed_train, X_train, X_test, y_train, y_test = process_training(df_train, use_imputer=True)
# Decision Tree

model_dt = DecisionTreeClassifier(random_state=1)

model_dt.fit(X_train, y_train)

# Predict train

predict_train = model_dt.predict(X_train)

print(classification_report(y_train, predict_train))

# Predict test

predict_test_dt = model_dt.predict(X_test)

print(classification_report(y_test, predict_test_dt))
# Random Forest

model_rf = RandomForestClassifier(n_estimators=500)

model_rf.fit(X_train, y_train)



# Predict train

predict_train = model_rf.predict(X_train)

print(classification_report(y_train, predict_train))

# Predict test

predict_test_rf = model_rf.predict(X_test)

print(classification_report(y_test, predict_test_rf))
# Train SVM classifier

model_svm = SVC()

model_svm.fit(X_train, y_train)



# Predict train

predict_train = model_svm.predict(X_train)

print(classification_report(y_train, predict_train))

# Predict test

predict_test_svm = model_svm.predict(X_test)

print(classification_report(y_test, predict_test_svm))
# Use XGboost

model_xgb = XGBClassifier()

model_xgb.fit(X_train, y_train)



# Predict train

predict_train = model_xgb.predict(X_train)

print(classification_report(y_train, predict_train))

# Predict test

predict_test_xgb = model_xgb.predict(X_test)

print(classification_report(y_test, predict_test_xgb))
# Train NN

def scalar_to_vector(ind, dim):

    np_vector = np.zeros(dim)

    np_vector[ind] = 1

    return np_vector

# Convert ys to vector to train NN

y_train_vector = np.asarray([scalar_to_vector(i, 2) for i in y_train.to_list()])

y_test_vector = np.asarray([scalar_to_vector(i, 2) for i in y_test.to_list()])

# Scale Xs

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



# Set up NN

model_NN = keras.Sequential()

model_NN.add(keras.layers.Dense(18, activation='relu', input_shape=(X_train.shape[1],)))

model_NN.add(keras.layers.Dropout(rate=0.2))

model_NN.add(keras.layers.Dense(2))

model_NN.summary()

# Compile model_NN before training

model_NN.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])

# Start training

history = model_NN.fit(X_train_scaled, y_train_vector,

#                     validation_data=(image_batch_val, label_batch_val),

                    validation_split=0.2,

                    epochs=30)



# Plot performance

plt.plot(history.epoch, history.history['accuracy'], label='Training Accuracy')

plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')

# plt.title('Performance with Pre-Trained model_NN ({}) without dropout'.format(pretrain_name))

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim(0, 1)

plt.legend()



def NN_predict(pred):

    # For NN, apply softmax and obtain the highest probability

    predict_label = tf.nn.softmax(pred)

    predict_label = np.argmax(predict_label, axis=1)

    return predict_label

# Predict train

predict_train = NN_predict(model_NN.predict(X_train_scaled))

print(classification_report(y_train, predict_train))

# Predict test

predict_test_NN = NN_predict(model_NN.predict(X_test_scaled))

print(classification_report(y_test, predict_test_NN))
# # Generate forecast from a poor man's ensemble

# predict_test_ensemble = np.asarray(list(zip(*[predict_test_NN, predict_test_rf, predict_test_xgb])))

# predict_test_ensemble = np.asarray([max(i) for i in predict_test_ensemble])

# print(classification_report(y_test, predict_test_ensemble))
# Use models on competition data

# Load in data

# Email specs

df_test_competition = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/test.csv', index_col='user_id', 

                       na_values=['Never checkout', 'Never open', 'Never login'])

# User specs

df_users = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/users.csv', index_col='user_id')

# Merge User and Email specs into one

df_test_competition = pd.merge(df_test_competition, df_users, how='left', left_index=True, right_index=True)



# Process data

# Create month, day and time data columns

df_test_competition['grass_date'] = pd.to_datetime(df_test_competition.grass_date, format='%Y-%m-%d 00:00:00+08:00')

df_test_competition['day'] = [calendar.day_name[i.weekday()] for i in df_test_competition['grass_date'].to_list()] 

# Remove some columns

df_test_competition.drop(columns=['grass_date'], inplace=True)

df_test_competition.set_index('row_id', inplace=True)
df_test_competition.isna().sum()
def process_test(df_test):

    '''

    Process training data and seperate into train and test dataset

    '''

    df_current = df_test.copy()

    # Choose following columns as independent variables

    chosen_columns = ['country_code', 'day', 'attr_3', 'domain', 

                      'last_open_day', 'open_count_last_60_days', 'open_count_last_10_days', 'open_count_last_30_days']

    df_current = df_current[chosen_columns]

    # For category var, change values to str

    categorical_columns = ['country_code', 'day', 'attr_3', 'domain']

    for columns in categorical_columns:

        df_current[columns] = df_current[columns].astype('str')

    # OneHotEncoding

    df_onehot = pd.get_dummies(df_current[categorical_columns])

    df_current = pd.concat([df_current, df_onehot], axis=1)

    df_current.drop(columns=categorical_columns, inplace=True)

    return df_current

# Process test data similar to train

processed_test_competition = process_test(df_test_competition)



# Some missing values in vars, need to imput

print('Imputing...')

imputer = KNNImputer(n_neighbors=2)

processed_test_competition_imputed = imputer.fit_transform(processed_test_competition)



# Put in pandas df

processed_test_competition_imputed = pd.DataFrame(processed_test_competition_imputed, 

                                                  index=processed_test_competition.index,

                                                  columns=processed_test_competition.columns)

processed_test_competition_imputed.head()
# Check whether imputer works

processed_test_competition_imputed.isna().sum()
# # Use poor man's ensemble

# predictions = []

# counter = 1

# for model in [model_NN, model_rf, model_xgb, model_dt, model_svm]:

#     if counter == 1:

#         # For NN model, need to do additional processing

#         # Invoke scaler

#         processed_test_competition_imputed_scaled = scaler.transform(processed_test_competition_imputed)

#         # Model prediction

#         predict_competition = model.predict(processed_test_competition_imputed_scaled)

#         predict_competition = NN_predict(predict_competition)

#     else:

#         # Model prediction

#         predict_competition = model.predict(processed_test_competition_imputed)

#     predictions.append(predict_competition)

#     counter = counter + 1

# # ensemble predict

# predict_competition = np.asarray(list(zip(*predictions)))

# predict_competition = np.asarray([max(i) for i in predict_competition])
use_NN = True

if use_NN:

    # Invoke scaler

    processed_test_competition_imputed_scaled = scaler.transform(processed_test_competition_imputed)

    # Model prediction

    predict_competition = model_NN.predict(processed_test_competition_imputed_scaled)

    predict_competition = NN_predict(predict_competition)

else:

    # Model prediction

    predict_competition = model.predict(processed_test_competition_imputed)

# Export predictions with row_id

processed_test_competition_imputed['open_flag'] = predict_competition

processed_test_competition_imputed['open_flag'].to_csv('./submission_NN.csv')