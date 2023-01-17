# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler

from sklearn.impute import SimpleImputer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_item = pd.read_csv('../input/train_ctrUa4K.csv')

df_test = pd.read_csv('../input/test_lAUu6dG.csv')

df_item

# Any results you write to the current directory are saved as output.
features_cat_data = ['Gender','Married', 'Education', 'Self_Employed', 'Property_Area']

features_num_cat_data = ['Dependents', 'Credit_History','LoanAmount','Loan_Amount_Term']

features=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome', 'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History', 'Property_Area']

label = 'Loan_Status'
imp_num = SimpleImputer(strategy='mean')

imp_cat = SimpleImputer(strategy='most_frequent')

def missing_values_filling(df, imputator, columns, test=False):

    df = df.copy()

    df['Dependents'] = df['Dependents'].apply(lambda x: int(x.strip('+')) if isinstance(x, str) else x)

    if not test:

        imputator.fit(df[columns])

    df[columns] = imputator.transform(df[columns])

    return df
label_encoder = preprocessing.OrdinalEncoder()

# ros = RandomOverSampler(random_state=0)



def features_processing(df_input, test=False):

    df = df_input.copy()   

    df = missing_values_filling(df, imp_num, features_num_cat_data)    

    df = missing_values_filling(df, imp_cat, features_cat_data)     

    if not test:

        label_encoder.fit(df[features_cat_data])

    df[features_cat_data] = label_encoder.transform(df[features_cat_data])   

#     if not test:

#         X_resampled, y_resampled = ros.fit_resample(df[features], df[label])

#         df = pd.DataFrame(data=X_resampled, columns=features)

#         df[label] = y_resampled

    return df
df_train = features_processing(df_item)
x_train, x_valid,y_train,y_valid = train_test_split(df_train[features],df_train[label], test_size=0.10, random_state=42)
model = LogisticRegression(random_state=50, solver='lbfgs', max_iter=5000)

clf = model.fit(x_train,y_train)
y_predict = clf.predict(x_valid)
clf.score(x_valid,y_valid)
confusion_matrix(y_valid, y_predict)
df_test_data = features_processing(df_test, test=True)
df_test_data[label]=model.predict(df_test_data[features])
id_cal = 'Loan_ID'

df_test_data[[id_cal,label]].to_csv("submission.csv", index=False)