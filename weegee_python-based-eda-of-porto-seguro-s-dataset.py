%pylab inline



#working with the numbers

import numpy as np

import pandas as pd

#visualization

import seaborn as sns

sns.set(style="whitegrid")

import missingno as msn

#machine learning

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, KFold

from sklearn.linear_model import LogisticRegression



import xgboost as xgb

import lightgbm as lgb



#other

import gc
df_train = pd.read_csv(r"input\train.csv")

df_test = pd.read_csv(r"input\test.csv")



print("Train data:")

print("Columns: {}".format(len(df_train.columns)))

print("Rows: {}". format(len(df_train)))

print("Test data:")

print("Columns: {}".format(len(df_test.columns)))

print("Rows: {}". format(len(df_test)))
display(df_train.head())

display(df_test.head())
df_train.dtypes 
df_train2 = df_train.replace(-1, np.NaN)

df_test2 = df_test.replace(-1, np.NaN)
sorted_traindata = msn.nullity_sort(df_train2, sort='descending')

msn.matrix(sorted_traindata)
msn.heatmap(df_train2)
sorted_testdata = msn.nullity_sort(df_test2, sort='descending')

msn.matrix(sorted_testdata)
binary_train = [c for c in df_train2.columns if c.endswith("bin")]

categorical_train = [c for c in df_train2.columns if c.endswith("cat")]



binary_test = [c for c in df_test2.columns if c.endswith("bin")]

categorical_test = [c for c in df_test2.columns if c.endswith("cat")]
plt.figure(figsize=(17,20))

for i, c in enumerate(binary_train):

    ax = plt.subplot(6,3,i+1)

    sns.countplot(df_train2[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.grid(False)
plt.figure(figsize=(17,20))

for i, c in enumerate(binary_test):

    ax = plt.subplot(6,3,i+1)

    sns.countplot(df_test2[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.grid(False)
print("Training Data")

for i in categorical_train:

    print(i)

    print(df_train2[i].isnull().sum())

print('\n"Test Data')   

for i in categorical_test:

    print(i)

    print(df_test2[i].isnull().sum())
to_drop = ["ps_car_03_cat","ps_car_05_cat"]

df_train2.drop(to_drop, axis=1, inplace=True)

df_test2.drop(to_drop, axis=1, inplace=True)

categorical_train = [i for i in categorical_train if i not in to_drop]

categorical_test = [i for i in categorical_test if i not in to_drop]
for i in categorical_train:

    print(i)

    print(df_train2[i].value_counts())
new_bin = ["ps_ind_04_cat","ps_car_02_cat","ps_car_07_cat", "ps_car_08_cat"]

for i in new_bin:

    print(i)

    print(df_test2[i].value_counts())
binary_train.append(new_bin)

binary_test.append(new_bin)

categorical_train = [i for i in categorical_train if i not in new_bin]

categorical_test = [i for i in categorical_test if i not in new_bin]
plt.figure(figsize=(17,10))

for i, c in enumerate(categorical_train):

    ax = plt.subplot(3,3,i+1)

    sns.countplot(df_train2[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.grid(False)
plt.figure(figsize=(17,10))

for i, c in enumerate(categorical_test):

    ax = plt.subplot(3,3,i+1)

    sns.countplot(df_test2[c])

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    plt.grid(False)
df_train3 = df_train2.apply(lambda x:x.fillna(x.value_counts().index[0]))

df_test3 = df_test2.apply(lambda x:x.fillna(x.value_counts().index[0]))
enc = OneHotEncoder()

enc.fit_transform(df_train3[categorical_train])

enc.fit_transform(df_test3[categorical_test])
continuous_train = [i for i in df_train3.columns if 

                    ((i not in binary_train) and (i not in categorical_train) and (i not in ["target", "id"]))]

continuous_test = [i for i in df_test3.columns if 

                   ((i not in binary_test) and (i not in categorical_test) and (i != "id"))]
corr = np.corrcoef(df_train3.transpose())

sns.heatmap(corr)
sns.clustermap(corr)
gc.collect()