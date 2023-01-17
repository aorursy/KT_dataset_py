# Import required libraries:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Set our random seed:

SEED = 17

PATH_TO_DIR = '../input/amazoncom-employee-access-challenge/'



print(os.listdir(PATH_TO_DIR))
# Import data:

train = pd.read_csv(PATH_TO_DIR + 'train.csv')
y = train['ACTION']

train = train[['RESOURCE', 'MGR_ID', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']]
logit = LogisticRegression(random_state=SEED)

rf = RandomForestClassifier(random_state=SEED)
# Split dataset into train and validation subsets:

X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, random_state=SEED)
# We create a helper function to get the scores for each encoding method:

def get_score(model, X, y, X_val, y_val):

    model.fit(X, y)

    y_pred = model.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    return score
# Lets have a quick look at our data:

X_train.head(5)
X_train.info()
# Discover the number of categories within each categorical feature:

len(X_train.RESOURCE.unique()), len(X_train.MGR_ID.unique()), len(X_train.ROLE_FAMILY_DESC.unique()), len(X_train.ROLE_FAMILY.unique()),len(X_train.ROLE_CODE.unique())
# Create a list of each categorical column name:

columns = [i for i in X_train.columns]
%%time

baseline_logit_score = get_score(logit, X_train, y_train, X_val, y_val)

print('Logistic Regression score without feature engineering:', baseline_logit_score)
%%time

baseline_rf_score = get_score(rf, X_train, y_train, X_val, y_val)

print('Random Forest score without feature engineering:', baseline_rf_score)
from sklearn.preprocessing import OneHotEncoder



one_hot_enc = OneHotEncoder(sparse=False)
print('Original number of features: \n', X_train.shape[1], "\n")

data_ohe_train = (one_hot_enc.fit_transform(X_train))

data_ohe_val = (one_hot_enc.transform(X_val))

print('Features after OHE: \n', data_ohe_train.shape[1])
%%time

ohe_logit_score = get_score(logit, data_ohe_train, y_train, data_ohe_val, y_val)

print('Logistic Regression score with one-hot encoding:', ohe_logit_score)
%%time

ohe_rf_score = get_score(rf, data_ohe_train, y_train, data_ohe_val, y_val)

print('Random Forest score with one-hot encoding:', ohe_rf_score)
# Install category_encoders:

# pip install category_encoders
from category_encoders import HashingEncoder
n_components_list = [100, 500, 1000, 5000, 10000]

n_components_list_str = [str(i) for i in n_components_list]
fh_logit_scores = []



# Iterate over different n_components:

for n_components in n_components_list:

    

    hashing_enc = HashingEncoder(cols=columns, n_components=n_components).fit(X_train, y_train)

    

    X_train_hashing = hashing_enc.transform(X_train.reset_index(drop=True))

    X_val_hashing = hashing_enc.transform(X_val.reset_index(drop=True))

    

    fe_logit_score = get_score(logit, X_train_hashing, y_train, X_val_hashing, y_val)

    fh_logit_scores.append(fe_logit_score)
plt.figure(figsize=(8, 5))

plt.plot(n_components_list_str, fh_logit_scores, linewidth=3)

plt.title('n_compontents vs roc_auc for feature hashing with logistic regression')

plt.xlabel('n_components')

plt.ylabel('score')

plt.show;
hashing_enc = HashingEncoder(cols=columns, n_components=10000).fit(X_train, y_train)



X_train_hashing = hashing_enc.transform(X_train.reset_index(drop=True))

X_val_hashing = hashing_enc.transform(X_val.reset_index(drop=True))
X_train_hashing.head()
%%time

hashing_logit_score = get_score(logit, X_train_hashing, y_train, X_val_hashing, y_val)

print('Logistic Regression score with feature hashing:', hashing_logit_score)
%%time

hashing_rf_score = get_score(rf, X_train_hashing, y_train, X_val_hashing, y_val)

print('Random Forest score with feature hashing:', hashing_rf_score)
# Create example dataframe with numbers ranging from 1 to 5:

example_df = pd.DataFrame([1,2,3,4,5], columns=['example'])



from category_encoders import BinaryEncoder



example_binary = BinaryEncoder(cols=['example']).fit_transform(example_df)



example_binary
binary_enc = BinaryEncoder(cols=columns).fit(X_train, y_train)
X_train_binary = binary_enc.transform(X_train.reset_index(drop=True))

X_val_binary = binary_enc.transform(X_val.reset_index(drop=True))

# note: category_encoders implementations can't handle shuffled datasets. 
print('Features after Binary Encoding: \n', X_train_binary.shape[1])
%%time

be_logit_score = get_score(logit, X_train_binary, y_train, X_val_binary, y_val)

print('Logistic Regression score with binary encoding:', be_logit_score)
%%time

binary_rf_score = get_score(rf, X_train_binary, y_train, X_val_binary, y_val)

print('Random Forest score with binary encoding:', binary_rf_score)
from category_encoders import TargetEncoder



targ_enc = TargetEncoder(cols=columns, smoothing=8, min_samples_leaf=5).fit(X_train, y_train)
X_train_te = targ_enc.transform(X_train.reset_index(drop=True))

X_val_te = targ_enc.transform(X_val.reset_index(drop=True))
X_train_te.head()
%%time

te_logit_score = get_score(logit, X_train_te, y_train, X_val_te, y_val)

print('Logistic Regression score with target encoding:', te_logit_score)
%%time

te_rf_score = get_score(rf, X_train_te, y_train, X_val_te, y_val)

print('Random Forest score with target encoding:', te_rf_score)
targ_enc = TargetEncoder(cols=columns, smoothing=8, min_samples_leaf=5).fit(X_train, y_train)



X_train_te = targ_enc.transform(X_train.reset_index(drop=True))

X_val_te = targ_enc.transform(X_val.reset_index(drop=True))
%%time

me_logit_score = get_score(logit, X_train_te, y_train, X_val_te, y_val)

print('Logistic Regression score with target encoding with regularization:', me_logit_score)
%%time

me_rf_score = get_score(rf, X_train_te, y_train, X_val_te, y_val)

print('Random Forest score with target encoding with regularization:', me_rf_score)
from sklearn.model_selection import KFold



# Create 5 kfold splits:

kf = KFold(random_state=17, n_splits=5, shuffle=False)
# Create copy of data:

X_train_te = X_train.copy()

X_train_te['target'] = y_train
all_set = []



for train_index, val_index in kf.split(X_train_te):

    # Create splits:

    train, val = X_train_te.iloc[train_index], X_train_te.iloc[val_index]

    val=val.copy()

    

    # Calculate the mean of each column:

    means_list = []

    for col in columns:

        means_list.append(train.groupby(str(col)).target.mean())

    

    # Calculate the mean of each category in each column:

    col_means = []

    for means_series in means_list:

        col_means.append(means_series.mean())

    

    # Encode the data:

    for column, means_series, means in zip(columns, means_list, col_means):

        val[str(column) + '_target_enc'] = val[str(column)].map(means_series).fillna(means) 

    

    list_of_mean_enc = [str(column) + '_target_enc' for column in columns]

    list_of_mean_enc.extend(columns)

    

    all_set.append(val[list_of_mean_enc].copy())



X_train_te=pd.concat(all_set, axis=0)
# Apply encodings to validation set:

X_val_te = pd.DataFrame(index=X_val.index)

for column, means in zip(columns, col_means):

    enc_dict = X_train_te.groupby(column).mean().to_dict()[str(column) + '_target_enc']

    X_val_te[column] = X_val[column].map(enc_dict).fillna(means)
# Create list of target encoded columns:

list_of_target_enc = [str(column) + '_target_enc' for column in columns]
%%time

kf_reg_logit_score = get_score(logit, X_train_te[list_of_target_enc], y_train, X_val_te, y_val)

print('Logistic Regression score with kfold-regularized target encoding:', kf_reg_logit_score)
%%time

kf_reg_rf_score = get_score(rf, X_train_te[list_of_target_enc], y_train, X_val_te, y_val)

print('Random Forest score with kfold-regularized target encoding:', kf_reg_rf_score)
from category_encoders import WOEEncoder



woe_enc = WOEEncoder(cols=columns, random_state=17).fit(X_train, y_train)
X_train_woe = woe_enc.transform(X_train.reset_index(drop=True))

X_val_woe = woe_enc.transform(X_val.reset_index(drop=True))
X_train_woe.head()
%%time

woe_logit_score = get_score(logit, X_train_woe, y_train, X_val_woe, y_val)

print('Logistic Regression score with woe encoding:', woe_logit_score)
%%time

woe_rf_score = get_score(rf, X_train_woe, y_train, X_val_woe, y_val)

print('Random Forest score with woe encoding:', woe_rf_score)