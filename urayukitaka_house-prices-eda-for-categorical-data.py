# Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.simplefilter("ignore")

# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Statistics library

from scipy import stats

from scipy.stats import norm



# Data preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Regression

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
# data loading

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
# Focus on category data

dtype = pd.DataFrame({"columns":df_train.dtypes.index,

                     "dtype":df_train.dtypes})

dtype["dtype"] = [str(i) for i in dtype["dtype"]]

dtype
# columns

obj_columns = dtype.query("dtype=='object'")["columns"].values



# categorical data

df_cate = df_train[obj_columns]

df_cate.head()
# data info

df_cate.info()
# null values

df_cate.isnull().sum()
# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(df_train["SalePrice"], fit=norm, ax=ax[0])

ax[0].set_xlabel("SalePrice")

ax[0].set_ylabel("count(normalized)")

ax[0].set_title("SalePrice distribution")



# Probability  plot

stats.probplot(df_train['SalePrice'], plot=ax[1])
# Distribution

log_SelePrice = np.log10(df_train["SalePrice"])

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(log_SelePrice, fit=norm, ax=ax[0])

ax[0].set_xlabel("SalePrice")

ax[0].set_ylabel("count(normalized)")

ax[0].set_title("log10(SalePrice distribution)")



# Probability  plot

stats.probplot(log_SelePrice, plot=ax[1])
# Create sale price log columns

df_train["log_SalePrice"] = np.log10(df_train["SalePrice"])
# visualization of categorical data

# distlibution

fig, ax = plt.subplots(9,5, figsize=(25,50))

for i in range(0,43):

    if i < 5:

        sns.countplot(df_cate.iloc[:,i], ax=ax[0,i])

    elif i >= 5 and i < 10:

        sns.countplot(df_cate.iloc[:,i], ax=ax[1,i-5])

    elif i >= 10 and i < 15:

        sns.countplot(df_cate.iloc[:,i], ax=ax[2,i-10])

    elif i >= 15 and i < 20:

        sns.countplot(df_cate.iloc[:,i], ax=ax[3,i-15])

    elif i >= 20 and i < 25:

        sns.countplot(df_cate.iloc[:,i], ax=ax[4,i-20])

    elif i >= 25 and i <30:

        sns.countplot(df_cate.iloc[:,i], ax=ax[5,i-25])

    elif i >= 30 and i <35:

        sns.countplot(df_cate.iloc[:,i], ax=ax[6,i-30])

    elif i >= 35 and i <40:

        sns.countplot(df_cate.iloc[:,i], ax=ax[7,i-35])

    else:

        sns.countplot(df_cate.iloc[:,i], ax=ax[8,i-40])
fig, ax = plt.subplots(9,5, figsize=(25,50))

for i in range(0,43):

    if i < 5:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[0,i])

    elif i >= 5 and i < 10:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[1,i-5])

    elif i >= 10 and i < 15:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[2,i-10])

    elif i >= 15 and i < 20:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[3,i-15])

    elif i >= 20 and i < 25:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[4,i-20])

    elif i >= 25 and i <30:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[5,i-25])

    elif i >= 30 and i <35:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[6,i-30])

    elif i >= 35 and i <40:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[7,i-35])

    else:

        sns.violinplot(df_cate.iloc[:,i], df_train["log_SalePrice"], ax=ax[8,i-40])
# create df

df_cha = df_train.copy()

df_cha = df_cha[np.insert(obj_columns,0,"log_SalePrice")]

df_cha.head()
# define function

# Change to mean value function

def change_cate_mean(df):

    df_c = df.copy()

    col_list_m = ["log_SalePrice"]

    for i in range(1,len(df.columns)):

        # mean values

        index = df.iloc[:,i].value_counts().index

        for j in range(len(index)):

            mean = df[df.iloc[:,i]==index[j]]["log_SalePrice"].mean()

            df.iloc[:,i][df.iloc[:,i]==index[j]] = mean

        # update column name

        col = df.columns[i]+'_m'

        col_list_m.append(col) 

    # create out put df

    df_c = df.copy()

    df_c.columns = col_list_m

    return df_c



# Change to std value function

def change_cate_std(df):

    col_list_s = ["log_SalePrice"]

    for i in range(1,len(df.columns)):

        index = df.iloc[:,i].value_counts().index

        for j in range(len(index)):

            std = df[df.iloc[:,i]==index[j]]["log_SalePrice"].std()

            df.iloc[:,i][df.iloc[:,i]==index[j]] = std

        # update column name

        col = df.columns[i]+'_s'

        col_list_s.append(col)

    # create out put df

    df_c = df.copy()

    df_c.columns = col_list_s

    return df_c



# Change to float value function

def change_float(df):

    for i in range(1,len(df.columns)):

        lists = []

        for j in range(len(df.iloc[:,i])):

            flo = float(df.iloc[:,i][j])

            lists.append(flo)

        df.iloc[:,i] = lists

    return df



# Fill by median function

def fill_median(df):

    for i in range(1,len(df.columns)):

        median = df.iloc[:,i].median()

        df.iloc[:,i].fillna(median, inplace=True)

    return df



# create df fix

df_cha_fix = pd.concat([

    fill_median(change_float(change_cate_mean(df_cha))), fill_median(change_float(change_cate_std(df_cha))).iloc[:,1:] ], axis=1)

df_cha_fix.head()
# Correlation

matrix = df_cha_fix.iloc[:,1:]



# Visualization

plt.figure(figsize=(16,16))

hm = sns.heatmap(matrix.corr(), vmax=1, vmin=-1, cmap="bwr", square=True)
# corrlation coefficient

coef = df_cha_fix.corr().round(3)
fig, ax = plt.subplots(9,5, figsize=(25,50))

for i in range(0,43):

    if i < 5:

        ax[0,i].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[0,i].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[0,i].set_ylabel("log10(SalePrice)")

        ax[0,i].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 5 and i < 10:

        ax[1,i-5].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[1,i-5].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[1,i-5].set_ylabel("log10(SalePrice)")

        ax[1,i-5].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 10 and i < 15:

        ax[2,i-10].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[2,i-10].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[2,i-10].set_ylabel("log10(SalePrice)")

        ax[2,i-10].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 15 and i < 20:

        ax[3,i-15].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[3,i-15].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[3,i-15].set_ylabel("log10(SalePrice)")

        ax[3,i-15].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 20 and i < 25:

        ax[4,i-20].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[4,i-20].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[4,i-20].set_ylabel("log10(SalePrice)")

        ax[4,i-20].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 25 and i <30:

        ax[5,i-25].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[5,i-25].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[5,i-25].set_ylabel("log10(SalePrice)")

        ax[5,i-25].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 30 and i <35:

        ax[6,i-30].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[6,i-30].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[6,i-30].set_ylabel("log10(SalePrice)")

        ax[6,i-30].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 35 and i <40:

        ax[7,i-35].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[7,i-35].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[7,i-35].set_ylabel("log10(SalePrice)")

        ax[7,i-35].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    else:

        ax[8,i-40].scatter(df_cha_fix.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[8,i-40].set_xlabel("{}".format(df_cha_fix.columns[i+1]))

        ax[8,i-40].set_ylabel("log10(SalePrice)")

        ax[8,i-40].set_title("R:{}".format(coef["log_SalePrice"][1+i]))
# Select variables with border of correlation with SalePrice values

# Check the accuracy(R2 score)

R2_train_score_list =[]

R2_test_score_list = []

variable_count_list = []

R2_valid_df = pd.DataFrame({})

# Roop

for i in range(0,8):

    col_select = coef[(coef["log_SalePrice"] > 0.1*i) | (coef["log_SalePrice"] < -0.1*i)].index

    # calc R2 score of Ridge regression

    # Dataset

    X = df_cha_fix[col_select].iloc[:,1:]

    y = df_cha["log_SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    # Scaling

    msc = MinMaxScaler()

    msc.fit(X)

    X_train_msc = msc.fit_transform(X_train)

    X_test_msc = msc.fit_transform(X_test)

    # Training and score

    ridge = Ridge().fit(X_train_msc, y_train)

    R2_train_score = ridge.score(X_train_msc, y_train)

    R2_test_score = ridge.score(X_test_msc, y_test)

    variable_count = len(col_select)-1

    R2_train_score_list.append(R2_train_score)

    R2_test_score_list.append(R2_test_score)

    variable_count_list.append(variable_count)

    



# create df

R2_valid_df["valid"] = ["All",">0.1|<-0.1",">0.2|<-0.2",">0.3|<-0.3",">0.4|<-0.4",">0.5|<-0.5",">0.6|<-0.6",">0.7|<-0.7"]

R2_valid_df["R2_train_score"] = R2_train_score_list

R2_valid_df["R2_test_score"] = R2_test_score_list

R2_valid_df["R2_variable_count"] = variable_count_list



# Check

R2_valid_df
# result_variables

result_columns = coef[(coef["log_SalePrice"] > 0.1*2) | (coef["log_SalePrice"] < -0.1*2)].index
df_result = df_cha_fix[result_columns]

# corrlation coefficient

coef = df_result.corr().round(3)



fig, ax = plt.subplots(7,5, figsize=(25,40))

for i in range(0,33):

    if i < 5:

        ax[0,i].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[0,i].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[0,i].set_ylabel("log10(SalePrice)")

        ax[0,i].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 5 and i < 10:

        ax[1,i-5].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[1,i-5].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[1,i-5].set_ylabel("log10(SalePrice)")

        ax[1,i-5].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 10 and i < 15:

        ax[2,i-10].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[2,i-10].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[2,i-10].set_ylabel("log10(SalePrice)")

        ax[2,i-10].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 15 and i < 20:

        ax[3,i-15].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[3,i-15].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[3,i-15].set_ylabel("log10(SalePrice)")

        ax[3,i-15].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 20 and i < 25:

        ax[4,i-20].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[4,i-20].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[4,i-20].set_ylabel("log10(SalePrice)")

        ax[4,i-20].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    elif i >= 25 and i <30:

        ax[5,i-25].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[5,i-25].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[5,i-25].set_ylabel("log10(SalePrice)")

        ax[5,i-25].set_title("R:{}".format(coef["log_SalePrice"][1+i]))

    else:

        ax[6,i-30].scatter(df_result.iloc[:,i+1], df_cha_fix["log_SalePrice"])

        ax[6,i-30].set_xlabel("{}".format(df_result.columns[i+1]))

        ax[6,i-30].set_ylabel("log10(SalePrice)")

        ax[6,i-30].set_title("R:{}".format(coef["log_SalePrice"][1+i]))