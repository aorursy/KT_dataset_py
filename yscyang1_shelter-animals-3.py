import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
train_df = pd.read_feather("../input/shelter/train_df")
def get_subset(df, train_percent=.6, validate_percent=.2, copy = True, seed=None):
    if copy:
        df_copy = df.copy()
    perm = np.random.RandomState(seed).permutation(df_copy.index)
    length = len(df_copy.index)
    train_end = int(train_percent * length)
    validate_end = int(validate_percent * length) + train_end
    train = df_copy.iloc[perm[:train_end]]
    validate = df_copy.iloc[perm[train_end:validate_end]]
    test = df_copy.iloc[perm[validate_end:]]
    
    return train, validate, test
train_speed, val_speed, test_speed = get_subset(train_df)
X_train_speed = train_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_train_speed = train_speed['Outcome1']
X_val_speed = val_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_val_speed = val_speed['Outcome1']
from sklearn.ensemble import RandomForestClassifier
def print_score(model, X_t, y_t, X_v, y_v, oob = False):
    print('Training Score: {}'.format(model.score(X_t, y_t)))
    print('Validation Score: {}'.format(model.score(X_v, y_v)))
    if oob:
        if hasattr(model, 'oob_score_'):
            print("OOB Score:{}".format(model.oob_score_))
rf_speed = RandomForestClassifier(n_estimators=60, min_samples_leaf=7, max_features=0.3, min_samples_split= 20, bootstrap=False, n_jobs=-1)
rf_speed.fit(X_train_speed, y_train_speed)
print_score(rf_speed, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
def get_feat_imp(model, df):
    tmp = pd.DataFrame({'Feature':  np.array(df.columns), 'Importance': np.array(model.feature_importances_)})
    return tmp.sort_values(by = ['Importance'], ascending = False)
if_df = get_feat_imp(rf_speed, X_train_speed)
if_df.head()
if_df.plot('Feature', 'Importance', kind = 'barh',legend=False, figsize=(10,6))
train_if, val_if, _ = get_subset(train_df, seed = 55)
X_train_if = train_if[if_df[if_df['Importance']>0.01]['Feature'].values]
X_val_if = val_if[if_df[if_df['Importance']>0.01]['Feature'].values]
y_train_if = train_if['Outcome1']
y_val_if = val_if['Outcome1']
rf_if = RandomForestClassifier(n_estimators=60, min_samples_leaf=7, max_features=0.3, min_samples_split= 20, bootstrap=False, n_jobs=-1)
rf_if.fit(X_train_if, y_train_if)
print_score(rf_if, X_train_if, y_train_if, X_val_if, y_val_if)
rf_speed_fi = get_feat_imp(rf_if, X_train_if)
print(rf_speed_fi.head())
rf_speed_fi.plot('Feature', 'Importance', kind = 'barh',legend=False, figsize=(10,6))
from sklearn.utils import shuffle
def shuffle_col(df, col_name):
    df_copy = df.copy()
    # Reset index of copy because index from get_subset wasn't reset, causes nan problems later
    df_copy.reset_index(inplace=True, drop=True)
    df_new = df_copy.drop(col_name, axis=1)
    shuf = shuffle(df[col_name])
    shuf.reset_index(inplace=True, drop=True)
    df_new[col_name] = shuf
    return df_new
shuf_df = shuffle_col(X_train_if, 'Sex')
rf_if.fit(shuf_df, y_train_if)
print_score(rf_if, shuf_df, y_train_if, X_val_if, y_val_if)
shuf_df = shuffle_col(X_train_if, 'Age')
rf_if.fit(shuf_df, y_train_if)
print_score(rf_if, shuf_df, y_train_if, X_val_if, y_val_if)
shuf_df = shuffle_col(X_train_if, 'Datehour')
rf_if.fit(shuf_df, y_train_if)
print_score(rf_if, shuf_df, y_train_if, X_val_if, y_val_if)
shuf_df = shuffle_col(X_train_if, 'Name')
rf_if.fit(shuf_df, y_train_if)
print_score(rf_if, shuf_df, y_train_if, X_val_if, y_val_if)
shuf_df = shuffle_col(X_train_if, 'Breed')
rf_if.fit(shuf_df, y_train_if)
print_score(rf_if, shuf_df, y_train_if, X_val_if, y_val_if)
train_speed['Name'].value_counts(ascending = False)[:5]
fig, ((axis1, axis2), (axis3, axis4)) = plt.subplots(2,2,figsize=(10,7))
order = ['Transfer', 'Adoption', 'Return_to_owner', 'Euthanasia', 'Died']

sns.countplot(x = 'Name', hue = 'Outcome1', data = train_speed[train_speed['Name']==5048], hue_order= order, ax = axis1)
sns.countplot(x = 'Name', hue = 'Outcome1', data = train_speed[train_speed['Name']==540], hue_order=order, ax = axis2)
sns.countplot(x = 'Name', hue = 'Outcome1', data = train_speed[train_speed['Name']==4542], hue_order=order, ax = axis3)
sns.countplot(x = 'Name', hue = 'Outcome1', data = train_speed[train_speed['Name']==1305], hue_order=order, ax = axis4)

axis2.get_legend().remove()
axis3.get_legend().remove()
axis4.get_legend().remove()
print('Number of categories in Sex: {}'.format(train_if['Sex'].nunique()))
print('Categories in Sex: {}'.format(train_if['Sex'].unique()))
print('Number of categories in Name: {}'.format(train_if['Name'].nunique()))
def oneHotEncode(df, max_cat):
    for col in df.columns.values:
        if df[col].nunique() < max_cat:
            test = pd.get_dummies(df[col], prefix = col)
            df = pd.concat([df, test], axis=1)
        df.drop(col, axis = 1, inplace=True)
    return df
tmp = oneHotEncode(train_if[['Name', 'Animal', 'Sex', 'Age', 'Breed', 'Color']], 6)
train_if2 = pd.concat([train_if, tmp], axis = 1)
train_if2.drop(['Sex', 'Animal'], axis = 1, inplace = True)
tmp = oneHotEncode(val_if[['Name', 'Animal', 'Sex', 'Age', 'Breed', 'Color']], 6)
val_if2 = pd.concat([val_if, tmp], axis = 1)
val_if2.drop(['Animal', 'Sex'], axis = 1, inplace = True)
X_train_if2 = train_if2.drop(['Outcome1', 'Outcome2'], axis = 1)
y_train_if2 = train_if2['Outcome1']
X_val_if2 = val_if2.drop(['Outcome1', 'Outcome2'], axis = 1)
y_val_if2 = val_if2['Outcome1']
rf_if2 = RandomForestClassifier(n_estimators=60, min_samples_leaf=7, max_features=0.3, min_samples_split= 20, bootstrap=False, n_jobs=-1)
rf_if2.fit(X_train_if2, y_train_if2)
print_score(rf_if2, X_train_if2, y_train_if2, X_val_if2, y_val_if2)
rf_speed_fi2 = get_feat_imp(rf_if2, X_train_if2)
print(rf_speed_fi2.head())
rf_speed_fi2.plot('Feature', 'Importance', kind = 'barh',legend=False, figsize=(10,6))