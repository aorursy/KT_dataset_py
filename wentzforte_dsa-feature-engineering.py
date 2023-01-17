import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
def missing_zero_values_table(df):

        zero_val = (df == 0.00).astype(int).sum(axis=0)

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

        mz_table = mz_table.rename(

        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']

        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)

        mz_table['Data Type'] = df.dtypes

        mz_table = mz_table[mz_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      

            "There are " + str(mz_table.shape[0]) + " columns that have missing values.")

        return mz_table
def show_corr(df):

    fig = plt.subplots(figsize = (20,20))

    sb.set(font_scale=1.5)

    sb.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

    plt.show()
def change_types(cols):

    for col in cols:

        labels = train[col].unique()

        map_labels = dict(zip(labels, range(0,len(labels))))

        train[col] = train[col].map(map_labels)

        train[col] = train[col].astype(float)

        

        test[col] = test[col].map(map_labels)

        test[col] = test[col].astype(float)
def change_categorical():

    for col in features_cat:      

        if col in features_ignore:

            continue

        features_ignore.append(col)

        for item in train[col].unique():

            if not item is np.NaN:            

                for i in list(set(str(item))):

                    new_col = col +'_'+ str.upper(str(i).replace(',',''))

                    if not new_col in train.columns:

                        train[new_col] = 0

                        test[new_col] = 0

                        #print('add', new_col)

                    train[new_col] = train[col].apply(lambda x: float(str(x).find(item) >= 0))                 

                    test[new_col] = test[col].apply(lambda x: float(str(x).find(item) >= 0))        
def remove_outlier(df, col):

    if col in features_ignore:

        return

    

    if df[col].dtype == object:

        return   

    

    fig = plt.subplots(figsize = (20,3))

    

    _std = round(df[col].std(), 5)

    _mean= round(df[col].mean(), 5)    

    _min = round(_mean - df[col].std()*3, 5)

    if _min < 0:

        _min = 0.00001

    _max = round(_mean + df[col].std()*3, 5)

    if _max > 20:

        _max = 19.99999

        

    plt.hist(df[col], bins=100)

    

    df.loc[(df[col] < 0) | (df[col] < _min) | (df[col] > _max)] = _mean

    df[col] = round(df[col], 5)

    

    plt.hist(df[col], bins=100)    

    print('Process Remove Outlier', col, '| mean:', _mean, '| std:', _std,

          '| min:', _min, '| max:', _max, '-> min:', df[col].min(), '| max:', df[col].max())

    plt.show()    
train = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv")

test = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv")
missing_zero_values_table(train)
train.head()
test.head()
train.describe().T
test.describe().T
train.dtypes
features_cat = []

for col in train.columns:

    if train[col].dtype == 'O':

        features_cat.append(col)

print(features_cat)
features_ignore = ['ID', 'v22', 'v10', 'v109', 'v104', 'v105', 'v15', 'v121', 'v114', 'v29', 'v26', 

                   'v25', 'v41', 'v11', 'v46', 'v33', 'v26', 'v54', 'v17', 'v20', 'v41', 

                   'v32', 'v64','v67', 'v63', 'v55', 'v32', 'v63', 'v92', 'v41', 'v118']
train.describe().T
%%time

#changeTypes(features_cat)

change_categorical()
train.describe().T
features = []

for col in test.columns:

    if col not in features_ignore:

        features.append(col)

features_train = features + ['target']

print(features_train)
missing_zero_values_table(train)
train.columns[2:132]
for col in train.columns[2:132]:

    remove_outlier(train, col)
for col in test.columns[1:132]:

    remove_outlier(test, col)
train.head()
train.to_csv('fe_train.csv', index=False)
test.to_csv('fe_test.csv', index=False)