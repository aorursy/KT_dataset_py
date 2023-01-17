# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

df_test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

df.head()
def convert_to_int(x):

    try:

        xx = x

        if x.find('\\') > -1:

            xx = x.split('\\')[1]

        s = int(str(xx), 16)

        return s

    except:

        return x



#df.nom_5 = df.nom_5.apply(lambda x: convert_to_int(x))

#df.nom_6 = df.nom_6.apply(lambda x: convert_to_int(x))

#df.nom_7 = df.nom_7.apply(lambda x: convert_to_int(x))

#df.nom_8 = df.nom_8.apply(lambda x: convert_to_int(x))

#df.nom_9 = df.nom_9.apply(lambda x: convert_to_int(x))



#df.iloc[:,11:16].apply(lambda x: convert_to_int(x))

#df.tail(25)
def set_temp(x):

    if x == 'Cold' or x == 'Freezing':

        return -1

    if x == 'Hot' or x == 'Lava Hot' or x == 'Boiling Hot':

        return 1

    return 0



df['ord_2_ex'] = df.ord_2.apply(lambda x: set_temp(x))

df_test['ord_2_ex'] = df_test.ord_2.apply(lambda x: set_temp(x))
df["ord_5a"] = df["ord_5"].str[0]

df["ord_5b"] = df["ord_5"].str[1]



df_test["ord_5a"] = df_test["ord_5"].str[0]

df_test["ord_5b"] = df_test["ord_5"].str[1]
from sklearn import preprocessing



def encode_df(df):

    df['bin_3_enc'] = preprocessing.LabelEncoder().fit_transform(df['bin_3'])

    df['bin_4_enc'] = preprocessing.LabelEncoder().fit_transform(df['bin_4'])

    df['nom_0_enc'] = preprocessing.LabelEncoder().fit_transform(df['nom_0'])

    df['nom_1_enc'] = preprocessing.LabelEncoder().fit_transform(df['nom_1'])

    df['nom_2_enc'] = preprocessing.LabelEncoder().fit_transform(df['nom_2'])

    df['nom_3_enc'] = preprocessing.LabelEncoder().fit_transform(df['nom_3'])

    df['nom_4_enc'] = preprocessing.LabelEncoder().fit_transform(df['nom_4'])

    df['ord_1_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_1'])

    df['ord_2_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_2'])

    df['ord_3_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_3'])

    df['ord_4_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_4'])

    #df['ord_5_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_5'])

    df['ord_5a_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_5a'])

    df['ord_5b_enc'] = preprocessing.LabelEncoder().fit_transform(df['ord_5b'])



    #df2 = df.drop(['bin_3','bin_4','nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 

    #               'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_5a', 'ord_5b'], axis=1)



    df2 = df.drop(['bin_3','bin_4','nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4',

                'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'ord_5a', 'ord_5b'], axis=1)

    

    return df2
df2 = encode_df(df)

df2_test = encode_df(df_test)



## https://www.kaggle.com/subinium/lightgbm-is-powerful

from category_encoders.target_encoder import TargetEncoder



te = TargetEncoder()

nom_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

df2[nom_cols] = te.fit_transform(X = df2[nom_cols], y = df['target'])

df2_test[nom_cols] = te.transform(X = df2_test[nom_cols])





df2.head()
count_classes = pd.value_counts(df2.target, sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("histogram")

plt.xlabel("traget")

plt.ylabel("Frequency")
true_txn_count = len(df2[df2.target == 1])

print('Number of traget rows is:' + str(true_txn_count))

true_txn_indices = np.array(df2[df2.target == 1].index)

false_txn_indices = df[df.target == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_txn_indices = np.random.choice(false_txn_indices, true_txn_count, replace = True)

random_txn_indices = np.array(random_txn_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([true_txn_indices,random_txn_indices])



# Under sample dataset

under_sample_data = df2.iloc[under_sample_indices,:]



X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'target']

y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'target']



# Showing ratio

print("Percentage of false txn: ", len(under_sample_data[under_sample_data.target == 0])/len(under_sample_data))

print("Percentage of true txn: ", len(under_sample_data[under_sample_data.target == 1])/len(under_sample_data))

print("Total number of txn in resampled data: ", len(under_sample_data))
from sklearn.preprocessing import StandardScaler



y = under_sample_data['target'].copy()

X = under_sample_data.copy()

X.drop(['target','id'], inplace=True, axis=1)

columns = X.columns.copy()
plt.figure(figsize=(16,16))

sns.heatmap(X.corr())
X.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
import lightgbm as lgb

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

print(X_train.shape)


lgb_train = lgb.Dataset(X_train, y_train)

lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)



#params = {'num_leaves':7, 'objective':'binary','max_depth':11,'learning_rate':.1,'max_bin':20,

#    'feature_fraction': .9,'bagging_fraction': 0.8,'bagging_freq': 10,'verbose': 0,

#         'min_data_in_leaf' : 24, }





params = {'num_leaves':8, 'objective':'binary','max_depth':10,'learning_rate':.1,'max_bin':12,

    'feature_fraction': .9,'bagging_fraction': 0.8,'bagging_freq': 15,'verbose': 0,

         'min_data_in_leaf' : 12, }

params['metric'] = ['auc', 'binary_logloss']



evals_result = {}  # to record eval results for plotting



print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=800,

                valid_sets=[lgb_train, lgb_test],

                #feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],

                evals_result=evals_result,

                verbose_eval=100)

print('Plotting metrics recorded during training...')

ax = lgb.plot_metric(evals_result, metric='auc')

plt.show()



ax = lgb.plot_metric(evals_result, metric='binary_logloss')

plt.show()



print('Plotting feature importances...')

ax = lgb.plot_importance(gbm, max_num_features=10)

plt.show()



#print('Plotting split value histogram...')

#ax = lgb.plot_split_value_histogram(gbm, feature='f22', bins='auto')

#plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score



#predicting on test set

y_pred = gbm.predict(X_test)



#calculating accuracy

#print(f'accuracy score:{accuracy_score(y_pred, y_test)}')

print(f'roc auc:{roc_auc_score(y_test,y_pred)}')
df2_test.head()
X2 = df2_test.copy()

X2.drop(['id'], inplace=True, axis=1)

X2 = scaler.fit_transform(X2)



y2_pred = gbm.predict(X2)



sub_df = pd.DataFrame()

sub_df['id'] = df2_test['id']

sub_df['target'] = y2_pred



sub_df.head()
sub_df.to_csv('submission.csv', index=False)