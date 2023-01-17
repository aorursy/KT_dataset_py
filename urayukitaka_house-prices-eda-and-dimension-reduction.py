# Basic libraries

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter('ignore')



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install factor_analyzer
# Statistics library

from scipy.stats import norm

from scipy import stats

import scipy



# Data preprocessing

import datetime

import re

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



# Visualization

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns



# Dimension reduction

from factor_analyzer import FactorAnalyzer

from sklearn.manifold import Isomap

from sklearn.manifold import TSNE

from sklearn import cluster
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
# data size

print("train data size:{}".format(df_train.shape))

print("test data size:{}".format(df_test.shape))
# data info

print("*"*50)

print("train data information")

print(df_train.info())
# summery

df_train.describe()
# Distribution

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(df_train["SalePrice"], ax=ax[0])

ax[0].set_xlabel("SalePrice")

ax[0].set_ylabel("count(normalized)")

ax[0].set_title("SalePrice distribution")



# Probability  plot

stats.probplot(df_train['SalePrice'], plot=ax[1])
# define function

# quantile_list

def quantile(data):

    quat_list = []

    for i in range(0,9):

        quat = data.quantile((i+1)*0.1)

        quat_list.append(quat)

    return quat_list



# decyl flag

def decyl(x):

    if x["SalePrice"] < quat_list[0]:

        res = 10

    elif x["SalePrice"] < quat_list[1] and x["SalePrice"] >= quat_list[0]:

        res = 9

    elif x["SalePrice"] < quat_list[2] and x["SalePrice"] >= quat_list[1]:

        res = 8

    elif x["SalePrice"] < quat_list[3] and x["SalePrice"] >= quat_list[2]:

        res = 7

    elif x["SalePrice"] < quat_list[4] and x["SalePrice"] >= quat_list[3]:

        res = 6

    elif x["SalePrice"] < quat_list[5] and x["SalePrice"] >= quat_list[4]:

        res = 5

    elif x["SalePrice"] < quat_list[6] and x["SalePrice"] >= quat_list[5]:

        res = 4

    elif x["SalePrice"] < quat_list[7] and x["SalePrice"] >= quat_list[6]:

        res = 3

    elif x["SalePrice"] < quat_list[8] and x["SalePrice"] >= quat_list[7]:

        res = 2

    else:

        res = 1

    return res



# quat_list

quat_list = quantile(df_train["SalePrice"])

# decyl

df_train["decyl"] = df_train.apply(decyl, axis=1)
# decyl data frame

decyl_df = pd.DataFrame(data=df_train.groupby("decyl").SalePrice.sum()).reset_index().sort_values(by="SalePrice", ascending=False)



# Ratio

decyl_df["ratio"] = decyl_df["SalePrice"] / decyl_df["SalePrice"].sum()

decyl_df["ratio_cumsum"] = decyl_df["ratio"].cumsum()*100



# Price band

price_band = pd.DataFrame({})

list_min = [df_train["SalePrice"].min()]

list_max = []



for i in range(0,9):

    quat = df_train["SalePrice"].quantile((i+1)*0.1)

    list_min.append(quat)

    list_max.append(quat)

list_max.append(df_train["SalePrice"].max())



decyl_df["min"] = sorted(list_min, reverse=True)

decyl_df["max"] = sorted(list_max, reverse=True)



decyl_df.head(10)
# Visualization decyl analysis

plt.figure(figsize=(10,6))

plt.bar(decyl_df["decyl"], decyl_df["max"]/1000-decyl_df["min"]/1000, bottom=decyl_df["min"]/1000, color="green")

plt.xlabel("decyl")

plt.xticks(range(1,11))

plt.ylabel("Price band(K)")

plt.yticks([0,200,400,600,800])
# Visualization decyl analysis

def barcumsum(x1, x2, y1, y2):

    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.bar(x1,y1, color="blue", label='Sale Price')

    ax1.set_xticks(range(1,11))

    ax1.set_ylim([0,60])

    ax1.grid()

    ax1.set_ylabel("Sale Price sum(M)")

    plt.legend(loc="upper left")

    ax2 = ax1.twinx()

    ax2.set_ylim([0, 110])

    ax2.set_ylabel("Ratio(%)")

    ax2.plot(x2,y2, color="red", label="Ratio", marker='o')

    ax1.set_xlabel("decyl")

    plt.legend(loc="upper right")

    

    

x1 = decyl_df["decyl"]

x2 = decyl_df["decyl"]

y1 = decyl_df["SalePrice"]/1000000

y2 = decyl_df["ratio_cumsum"]

barcumsum(x1, x2, y1, y2)
# R:year, M:Sale Price

RM_pivot = pd.pivot_table(df_train, index="decyl", columns="YrSold", values="SalePrice", aggfunc="sum").sort_index(by="decyl")

# color palette 

cm = sns.light_palette("red", as_cmap=True)

RM_pivot.style.background_gradient(cmap=cm)
# dtype object

dtype = pd.DataFrame({"columns":df_train.dtypes.index,

                     "dtype":df_train.dtypes})

dtype["dtype"] = [str(i) for i in dtype["dtype"]]

dtype
# columns

num_columns = dtype.query('dtype=="int64" | dtype=="float64"')["columns"].values[1:-1]

obj_columns = dtype.query('dtype=="object"')["columns"].values



print("numerical_values_count:{}".format(len(num_columns)))

print("object_values_count:{}".format(len(obj_columns)))
# correlation

matrix = df_train[num_columns].fillna(0) # tempolary fill na =0



plt.figure(figsize=(16,16))

hm = sns.heatmap(matrix.corr(), vmax=1, vmin=-1, cmap="bwr", square=True)
fig, ax = plt.subplots(1,3,figsize=(20,6))



ax[0].scatter(df_train["TotalBsmtSF"], df_train["1stFlrSF"], color="pink")

ax[0].set_xlabel("TotalBsmtSF")

ax[0].set_ylabel("1stFlrSF")



ax[1].scatter(df_train["GrLivArea"], df_train["TotRmsAbvGrd"], color="pink")

ax[1].set_xlabel("GrLivArea")

ax[1].set_ylabel("TotRmsAbvGrd")



ax[2].scatter(df_train["GarageCars"], df_train["GarageArea"], color="pink")

ax[2].set_xlabel("GarageCars")

ax[2].set_ylabel("GarageArea")
# Correlation coefficient

coef = matrix.corr().iloc[36,:].round(3)



# skewness

skew = []

for i in range(df_train[num_columns].shape[1]):

    median = df_train[num_columns].iloc[:,i].median()

    s = scipy.stats.skew(df_train[num_columns].iloc[:,i].fillna(median))

    skew.append(s)



# kurtosis

kurt = []

for i in range(df_train[num_columns].shape[1]):

    median = df_train[num_columns].iloc[:,i].median()

    s = scipy.stats.kurtosis(df_train[num_columns].iloc[:,i].fillna(median))

    kurt.append(s)





ske_kur = pd.DataFrame(skew, index=df_train[num_columns].columns, columns=["skew"])

ske_kur["kurt"] = kurt



skew = ske_kur["skew"].values.round(2)

kurt = ske_kur["kurt"].values.round(2)
fig, ax = plt.subplots(18,4, figsize=(25,100))

plt.subplots_adjust(hspace=0.3, wspace=0.3)

for i in range(0,36):

    if i < 4:

        ax[0,i].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="red")

        ax[0,i].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[0,i].set_ylabel("SalePrice")

        ax[0,i].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[1,i], kde=False, color="red")

        ax[1,i].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[1,i].set_ylabel("Count")

        

    elif i >= 4 and i < 8:

        ax[2,i-4].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="blue")

        ax[2,i-4].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[2,i-4].set_ylabel("SalePrice")

        ax[2,i-4].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[3,i-4], kde=False, color="blue")

        ax[3,i-4].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[3,i-4].set_ylabel("Count")

        

    elif i >= 8 and i < 12:

        ax[4,i-8].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="green")

        ax[4,i-8].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[4,i-8].set_ylabel("SalePrice")

        ax[4,i-8].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[5,i-8], kde=False, color="green")

        ax[5,i-8].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[5,i-8].set_ylabel("Count")

        

    elif i >= 12 and i < 16:

        ax[6,i-12].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="pink")

        ax[6,i-12].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[6,i-12].set_ylabel("SalePrice")

        ax[6,i-12].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[7,i-12], kde=False, color="pink")

        ax[7,i-12].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[7,i-12].set_ylabel("Count")

        

    elif i >= 16 and i < 20:

        ax[8,i-16].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="orange")

        ax[8,i-16].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[8,i-16].set_ylabel("SalePrice")

        ax[8,i-16].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[9,i-16], kde=False, color="orange")

        ax[9,i-16].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[9,i-16].set_ylabel("Count")

        

    elif i >= 20 and i < 24:

        ax[10,i-20].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="purple")

        ax[10,i-20].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[10,i-20].set_ylabel("SalePrice")

        ax[10,i-20].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[11,i-20], kde=False, color="purple")

        ax[11,i-20].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[11,i-20].set_ylabel("Count")

        

    elif i >= 24 and i < 28:

        ax[12,i-24].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="sienna")

        ax[12,i-24].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[12,i-24].set_ylabel("SalePrice")

        ax[12,i-24].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[13,i-24], kde=False, color="sienna")

        ax[13,i-24].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[13,i-24].set_ylabel("Count")

        

    elif i >= 28 and i < 32:

        ax[14,i-28].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="cyan")

        ax[14,i-28].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[14,i-28].set_ylabel("SalePrice")

        ax[14,i-28].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[15,i-28], kde=False, color="cyan")

        ax[15,i-28].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[15,i-28].set_ylabel("Count")

        

    else:

        ax[16,i-32].scatter(df_train[num_columns].iloc[:,i], df_train["SalePrice"], color="gray")

        ax[16,i-32].set_title("{}".format(df_train[num_columns].columns[i] + " :R" + str(coef[i])))

        ax[16,i-32].set_ylabel("SalePrice")

        ax[16,i-32].set_xlabel("{}".format(df_train[num_columns].columns[i]))

        sns.distplot(df_train[num_columns].iloc[:,i], ax=ax[17,i-32], kde=False, color="gray")

        ax[17,i-32].set_title("{}".format("skew:"+ str(skew[i]) + "/" + "kurt:" + str(kurt[i])))

        ax[17,i-32].set_ylabel("Count")
columns = df_train[num_columns[:-1]].columns



# fill by median

for i in range(len(columns)):

    median = df_train[columns[i]].median()

    df_train[columns[i]].fillna(median, inplace=True)
# calculate eigen values

eigen_vals = sorted(np.linalg.eigvals(df_train[num_columns[:-1]].corr()), reverse=True)



# plot

plt.figure(figsize=(10,6))

plt.plot(eigen_vals, "s-")

plt.xlabel("factor")

plt.ylabel("eigenvalue")
# Create instance

fa = FactorAnalyzer(n_factors=5, rotation="promax", impute="drop")

fa.fit(df_train[num_columns[:-1]].fillna(0))



result = fa.loadings_

colnames = df_train[num_columns[:-1]].columns



# Visualization by heatmap

plt.figure(figsize=(15,15))

hm = sns.heatmap(result,

                cbar=True,

                annot=True,

                cmap="bwr",

                fmt=".2f",

                annot_kws={"size":10},

                yticklabels=colnames,

                vmax=1,

                vmin=-1,

                center=0)

plt.xlabel("factors")

plt.ylabel("numerical values")
# factor point dataframe

factor_point = pd.DataFrame(fa.transform(df_train[num_columns[:-1]]), columns=["factor1", "factor2", "factor3", "factor4", "factor5"])

factor_point.head()
# Visualization by plot

x1 = factor_point["factor1"]

y1 = factor_point["factor2"]

c = df_train["SalePrice"]



plt.figure(figsize=(10,8))

plt.scatter(x1, y1, c=c)

plt.xlabel("factor1\nSize of the first floor of the house factor")

plt.ylabel("factor2\nNewness from outside factor")

plt.colorbar()
# Create instance

iso = Isomap(n_components=2)



# Fitting

iso.fit(df_train[num_columns[:-1]])

iso_projected = iso.transform(df_train[num_columns[:-1]])
# Visualization by plot

x1 = iso_projected[:,0]

y1 = iso_projected[:,1]

c = df_train["SalePrice"]



plt.figure(figsize=(10,8))

plt.scatter(x1, y1, c=c)

plt.xlabel("iso_axis1")

plt.ylabel("iso_axis2")

plt.colorbar()
# Create instance

tsne = TSNE(n_components=2, random_state=10)



# Fitting

tsne_projected = tsne.fit_transform(df_train[num_columns[:-1]])
# Visualization by plot

x1 = tsne_projected[:,0]

y1 = tsne_projected[:,1]

c = df_train["SalePrice"]



plt.figure(figsize=(10,8))

plt.scatter(x1, y1, c=c)

plt.xlabel("iso_axis1")

plt.ylabel("iso_axis2")

plt.colorbar()
# Scaling parameters with standarlizing

# Create instance

sc = StandardScaler()

# Fitting

sc.fit(df_train[num_columns[:-1]])



# Create prameters

X = sc.fit_transform(df_train[num_columns[:-1]])
# kmeans and visualization with violinplots cluster numbers.

fig, ax = plt.subplots(2,3, figsize=(24,12))

for i in range(2,8):

    # Create instance

    kmeans = cluster.KMeans(n_clusters=i, max_iter=30, init="random", random_state=10)

    

    # Fitting

    kmeans.fit(X)

    

    # cluster label

    labels = kmeans.predict(X)

    

    # Combine with df_train

    df_train["cluster_label"] = labels



    # Confirming by violinplot, if they can separate habing relationship with SalePrice

    if i >= 2 and i < 5:

        sns.violinplot("cluster_label","SalePrice", data=df_train, ax=ax[0,i-2])

        ax[0,i-2].set_xlabel("cluster")

        ax[0,i-2].set_ylabel("Price")

    else:

        sns.violinplot("cluster_label","SalePrice", data=df_train, ax=ax[1,i-5])

        ax[1,i-5].set_xlabel("cluster")

        ax[1,i-5].set_ylabel("Price")
# decided cluster values 6

kmeans = cluster.KMeans(n_clusters=6, max_iter=30, init="random", random_state=10)

    

# Fitting

kmeans.fit(X)

    

# cluster label

labels = kmeans.predict(X)



# Combine with df_train

df_train["cluster_label"] = labels



# Cisualization

plt.figure(figsize=(10,6))

sns.violinplot("cluster_label","SalePrice", data=df_train)
# Visualization by plot

x1 = tsne_projected[:,0]

y1 = tsne_projected[:,1]

c_clust = labels



fig, ax = plt.subplots(1,2,figsize=(20, 6))

cs = ax[0].scatter(x1, y1, c=c_clust, cmap='Set1_r')

ax[0].set_xlabel("iso_axis1")

ax[0].set_ylabel("iso_axis2")

fig.colorbar(cs, ax=ax[0])

ax[0].set_title("Cluster label")



cp = ax[1].scatter(x1, y1, c=c)

ax[1].set_xlabel("iso_axis1")

ax[1].set_ylabel("iso_axis2")

fig.colorbar(cp, ax=ax[1])

ax[1].set_title("Price")
# Create dataframe with scaling values

compare = pd.DataFrame(X, columns=num_columns[:-1])

compare["cluster_label"] = labels

compare = compare.groupby("cluster_label").mean()[num_columns[:-1]].reset_index()

compare = compare.iloc[:,1:].T

compare.head()
# Visualization by plot

plt.figure(figsize=(20,10))

plt.plot(compare.index, compare[0])

plt.plot(compare.index, compare[3])

plt.plot(compare.index, compare[4])

plt.xticks(rotation=90)

plt.xlabel("Parameter")

plt.ylabel("Values(Standarlized)")

plt.legend(["cluster_0", "cluster_3", "cluster_4"])