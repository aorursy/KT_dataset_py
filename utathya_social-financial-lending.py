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
# import libs



import numpy as np

import os

import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import norm 

import matplotlib.pyplot as plt

from collections import Counter



import math



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

import operator

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
sfl_train = pd.read_csv("/kaggle/input/finance/application_train.csv", encoding="utf-8", engine='c')

sfl_test = pd.read_csv("/kaggle/input/finance/application_test.csv", encoding="utf-8", engine='c')
# Imputing the mode of CNT_FAM_MEMBERS based on CNT_CHILDREN to train and test

sfl_train.loc[sfl_train.loc[:, "CNT_FAM_MEMBERS"].isnull(), "CNT_FAM_MEMBERS"] = sfl_train.loc[sfl_train.loc[:, "CNT_CHILDREN"] == 0, "CNT_FAM_MEMBERS"].mode()[0]

sfl_test.loc[sfl_test.loc[:, "CNT_FAM_MEMBERS"].isnull(), "CNT_FAM_MEMBERS"] = sfl_test.loc[sfl_test.loc[:, "CNT_CHILDREN"] == 0, "CNT_FAM_MEMBERS"].mode()[0]



# Imputing the common value "Unaccompanied" to NAME_TYPE_SUITE on train and test

sfl_train.loc[sfl_train.loc[:, "NAME_TYPE_SUITE"].isnull(), "NAME_TYPE_SUITE"] = "Unaccompanied"

sfl_test.loc[sfl_test.loc[:, "NAME_TYPE_SUITE"].isnull(), "NAME_TYPE_SUITE"] = "Unaccompanied"



# Assigning occupation type based on most common value when organisation type is of a particular value in train

# Creating the dictionary of corresponding occupation to organisation type

# Initialising list

ls_occ = []

ls_org = []



# Running a loop for all organisation types

for val in sfl_train.loc[:, "ORGANIZATION_TYPE"].unique():

    ls_org.append(str(val))

    ls_occ.append(str(sfl_train.iloc[sfl_train[sfl_train.iloc[:, 40].isin([val])].index, 28].mode()[0]))



# Creating a dictionary with key as organisation type and value as the most common occupation

dic_org = dict(zip(ls_org, ls_occ))



# Assigning XNA occupation to XNA organisation

dic_org["XNA"] = "XNA"



# Assigning occupation type based on most common value when organisation type is of a particular value in test

# Assigning XNA to XNA

sfl_test.loc[sfl_test.loc[:, "ORGANIZATION_TYPE"].isin(["XNA"]), "OCCUPATION_TYPE"] = "XNA"



# Creating the dictionary of corresponding occupation to organisation type

# Initialising list

ls_occ = []

ls_org = []



# Running a loop for all organisation types

for val in sfl_test.loc[:, "ORGANIZATION_TYPE"].unique():

    ls_org.append(str(val))

    ls_occ.append(str(sfl_test.iloc[sfl_test[sfl_test.iloc[:, 39].isin([val])].index, 27].mode()[0]))



# Creating a dictionary with key as organisation type and value as the most common occupation

dic_org = dict(zip(ls_org, ls_occ))



# Imputing Occupation for train

for key, val in dic_org.items():

    sfl_train["OCCUPATION_TYPE"][sfl_train.iloc[:, 28].isnull() & sfl_train.iloc[:, 40].isin([key])] = val



# Imputing Occupation for test

for key, val in dic_org.items():

    sfl_test["OCCUPATION_TYPE"][sfl_test.iloc[:, 27].isnull() & sfl_test.iloc[:, 39].isin([key])] = val

    

# Combining Ext_SOURCE_1, _2 & _3 into EXT_SOURCE in train

z = []

for index, row in sfl_train.iloc[:, [41,42,43]].iterrows():

    total = 0

    count = 0

    

    # Checking if Null

    if not math.isnan(row[0]):

        total += row[0]

        count +=1

    if not math.isnan(row[1]):

        total += row[1]

        count +=1

    if not math.isnan(row[2]):

        total += row[2]

        count +=1

    if count == 0:

        count = 1

    

    # Adppending Average

    z.append(total/count)

    

# Combining Ext_SOURCE_1, _2 & _3 into EXT_SOURCE in test

z_tst = []

for index, row in sfl_test.iloc[:, [40,41,42]].iterrows():

    total = 0

    count = 0

    

    # Checking if Null

    if not math.isnan(row[0]):

        total += row[0]

        count +=1

    if not math.isnan(row[1]):

        total += row[1]

        count +=1

    if not math.isnan(row[2]):

        total += row[2]

        count +=1

    if count == 0:

        count = 1

        

    # Adppending Average

    z_tst.append(total/count)

    

# Dealing with own car age in train. Converting Null to 0 and all ages moved to higher value

sfl_train.iloc[:, 21] = sfl_train.iloc[:, 21] + 1

sfl_train.loc[sfl_train.iloc[:, 21].isnull(), "OWN_CAR_AGE"] = 0



# Dealing with own car age in test. Converting Null to 0 and all ages moved to higher value

sfl_test.iloc[:, 20] = sfl_test.iloc[:, 20] + 1

sfl_test.loc[sfl_test.iloc[:, 20].isnull(), "OWN_CAR_AGE"] = 0



# Dealing with values to remove in train

ls_rem = []

ls_rem = [41, 42, 43, 9, 10, 11]

ls_rem.extend(range(44,95))

ls_rem.extend(range(116,122))



# Dealing with values to remove in test

ls_tst_rem = []

ls_tst_rem = [40, 41, 42, 8, 9, 10]

ls_tst_rem.extend(range(43,94))

ls_tst_rem.extend(range(115,121))



# Dropping column and adding EXT source in train

sfl_train.drop(sfl_train.iloc[:, ls_rem].columns, axis=1, inplace=True)

sfl_train.loc[:, "EXT_SOURCE"] = z



# Dropping column and adding EXT source in test

sfl_test.drop(sfl_test.iloc[:, ls_tst_rem].columns, axis=1, inplace=True)

sfl_test.loc[:, "EXT_SOURCE"] = z_tst



# Assigning mode to one row in DAYS_LAST_PHONE_CHANGE in test set

sfl_test.loc[sfl_test["DAYS_LAST_PHONE_CHANGE"].isnull(), "DAYS_LAST_PHONE_CHANGE"] = sfl_test["DAYS_LAST_PHONE_CHANGE"].mode()[0]
# Removing columns based on close corelation in train

ls_rem = [6, 14, 15, 27, 33, 36, 43, 45]



# Dropping column

sfl_train.drop(sfl_train.iloc[:, ls_rem].columns, axis=1, inplace=True)



# Removing columns based on close corelation in test

ls_rem = [5, 13, 14, 26, 32, 35, 42, 44]



# Dropping column

sfl_test.drop(sfl_test.iloc[:, ls_rem].columns, axis=1, inplace=True)
# Standardising Train and test values

# Converting Train categorical values to numerical

# Initialising new dataframes

sfl_train_std = sfl_train.copy()

sfl_test_std = sfl_test.copy()



# Reassigning numeric values in train

sfl_train_std["NAME_CONTRACT_TYPE"] = sfl_train_std["NAME_CONTRACT_TYPE"].map({"Cash loans" : 1, "Revolving loans" : 0})

sfl_train_std["FLAG_OWN_CAR"] = sfl_train_std["FLAG_OWN_CAR"].map({"Y" : 1, "N" : 0})

sfl_train_std["FLAG_OWN_REALTY"] = sfl_train_std["FLAG_OWN_REALTY"].map({"Y" : 1, "N" : 0})

sfl_train_std["CODE_GENDER"] = sfl_train_std["CODE_GENDER"].map({"F" : 1, "M" : 0, "XNA" : 2})



# Reassigning numeric values in test

sfl_test_std["NAME_CONTRACT_TYPE"] = sfl_test_std["NAME_CONTRACT_TYPE"].map({"Cash loans" : 1, "Revolving loans" : 0})

sfl_test_std["FLAG_OWN_CAR"] = sfl_test_std["FLAG_OWN_CAR"].map({"Y" : 1, "N" : 0})

sfl_test_std["FLAG_OWN_REALTY"] = sfl_test_std["FLAG_OWN_REALTY"].map({"Y" : 1, "N" : 0})

sfl_test_std["CODE_GENDER"] = sfl_test_std["CODE_GENDER"].map({"F" : 1, "M" : 0, "XNA" : 2})



# Reassigning using Label Encoder in both train and test

le = LabelEncoder()

sfl_train_std["NAME_INCOME_TYPE"] = le.fit_transform(sfl_train_std["NAME_INCOME_TYPE"])

sfl_test_std["NAME_INCOME_TYPE"] = le.transform(sfl_test_std["NAME_INCOME_TYPE"])



le = LabelEncoder()

sfl_train_std["NAME_EDUCATION_TYPE"] = le.fit_transform(sfl_train_std["NAME_EDUCATION_TYPE"])

sfl_test_std["NAME_EDUCATION_TYPE"] = le.transform(sfl_test_std["NAME_EDUCATION_TYPE"])



le = LabelEncoder()

sfl_train_std["NAME_FAMILY_STATUS"] = le.fit_transform(sfl_train_std["NAME_FAMILY_STATUS"])

sfl_test_std["NAME_FAMILY_STATUS"] = le.transform(sfl_test_std["NAME_FAMILY_STATUS"])



le = LabelEncoder()

sfl_train_std["NAME_HOUSING_TYPE"] = le.fit_transform(sfl_train_std["NAME_HOUSING_TYPE"])

sfl_test_std["NAME_HOUSING_TYPE"] = le.transform(sfl_test_std["NAME_HOUSING_TYPE"])



le = LabelEncoder()

sfl_train_std["OCCUPATION_TYPE"] = le.fit_transform(sfl_train_std["OCCUPATION_TYPE"])

sfl_test_std["OCCUPATION_TYPE"] = le.transform(sfl_test_std["OCCUPATION_TYPE"])



le = LabelEncoder()

sfl_train_std["ORGANIZATION_TYPE"] = le.fit_transform(sfl_train_std["ORGANIZATION_TYPE"])

sfl_test_std["ORGANIZATION_TYPE"] = le.transform(sfl_test_std["ORGANIZATION_TYPE"])



le = LabelEncoder()

sfl_train_std["WEEKDAY_APPR_PROCESS_START"] = le.fit_transform(sfl_train_std["WEEKDAY_APPR_PROCESS_START"])

sfl_test_std["WEEKDAY_APPR_PROCESS_START"] = le.transform(sfl_test_std["WEEKDAY_APPR_PROCESS_START"])
# Removing columns based on Extra trees classifier information. Bottom 24 percentile is removed in train and test

sfl_train_std.drop(["FLAG_MOBIL","FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_21", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_10"], axis=1, inplace=True)

sfl_test_std.drop(["FLAG_MOBIL","FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_21", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_10"], axis=1, inplace=True)
# Standardising the train and test data

# Combining train and test

frames = [sfl_train_std.drop(["SK_ID_CURR", "TARGET"], axis=1), sfl_test_std.drop("SK_ID_CURR", axis=1)]

df_std = pd.concat(frames, keys=['train', 'test'])

#Standardising

df_std.iloc[:,:] = preprocessing.scale(df_std.iloc[:,:])



# Seperating Train and Test

sfl_train_std.iloc[:, 2:] = df_std.xs('train', level=0)

sfl_test_std.iloc[:, 1:] = df_std.xs('test', level=0)
# Based on exploration set cutoff at 0.80 and predict output using Naive Bayes

# First create arrays holding input and output data

# Creating an object for Label Encoder and fitting on target strings

y_train = sfl_train_std["TARGET"].values



# Creating train set

x_train = sfl_train_std.drop(["SK_ID_CURR","TARGET"], axis=1).values



# Creating test set

x_test = sfl_test_std.drop(["SK_ID_CURR"], axis=1).values



# Creating model

clf = GaussianNB()



# Feeding test data

clf.fit(x_train, y_train)



#Recording results

predictions = clf.predict_proba(x_test)



# Applying cutoff

sfl_sub = pd.DataFrame({'SK_ID_CURR':sfl_test["SK_ID_CURR"].tolist(),'Prob':predictions[:, 1]})

sfl_sub["TARGET"] = sfl_sub['Prob'].apply(lambda row: 1 if row > 0.80 else 0)

sfl_sub.drop("Prob", axis=1, inplace=True)



# Creating submission file

# sfl_sub.to_csv("submission.csv", index=False)