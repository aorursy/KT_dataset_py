# Basic library

import numpy as np 

import pandas as pd

import time

import warnings

warnings.simplefilter("ignore")



# Statistics library

from scipy.stats import norm

from scipy import stats

import scipy



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Data loading

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
# feat = open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt")

# feat.readlines()

# Display omitted
# Confirming the data

train.head()
test.head()
# Data size

print("train data size:{}".format(train.shape))

print("test data size:{}".format(test.shape))
# Null check and data type check

# train.info()

# Display omitted
# test.info()

# Display omitted
train["MSZoning"].dtypes=="object"
# Data type, null check functions

def data_check(data_series):

    print("Data_type:{}".format(data_series.dtypes))

    print("Null counts:{}, Null_Ratio:{}%".format(data_series.isnull().sum(), round(data_series.isnull().sum()/len(data_series)*100,2)))

    print("Data describe;\n{}".format(data_series.describe()))

    if data_series.dtypes=="object":

        print("Unique categorical values:{}".format(data_series.unique()))

    else:

        pass



# Visualization functions

# Numerical values distribution

def dist_plot(data_series, label, title):

    # Axes subplots

    fig, ax = plt.subplots(1,2,figsize=(20,6))

    # Distribution

    sns.distplot(data_series, ax=ax[0], fit=norm, kde=False)

    ax[0].set_xlabel(label)

    ax[0].set_ylabel("Frequency")

    ax[0].set_title("{}\nskewness:{}_kurtsis:{}".format(title, 

                                                         round(scipy.stats.skew(data_series.dropna()),2), 

                                                         round(scipy.stats.kurtosis(data_series.dropna()),2)))

    # Probability plot

    stats.probplot(data_series, plot=ax[1])

    

# correlation check with SalePrice

def plot_corr(data_series, target_series, label, target_label, title):

    plt.figure(figsize=(6,6))

    plt.scatter(data_series, target_series)

    plt.xlabel(label)

    plt.ylabel(target_label)

    plt.title(title)



# barplot for Categorical value count and box plot vs SalePrice    

def bar_and_box_plot(data_series, target_series, label, target_label, title):

    fig, ax = plt.subplots(1,2, figsize=(20,6))

    # bar plot of categorical value

    count = data_series.value_counts()

    ax[0].bar(count.index, count)

    ax[0].set_xlabel(label)

    ax[0].set_ylabel("count")

    ax[0].set_title(title)

    ax[0].tick_params(axis="x", labelrotation=90)

    

    # box plot vs target label

    sns.boxplot(data_series, target_series, ax=ax[1], showfliers=False)

    sns.stripplot(data_series, target_series, ax=ax[1], jitter=True, color="black")

    ax[1].set_xlabel(label)

    ax[1].set_ylabel(target_label)

    ax[1].set_title("box_plot_"+title+"vs"+target_label)

    ax[1].tick_params(axis="x", labelrotation=90)
# Data check

data_check(train["SalePrice"])
# Raw data distribution

dist_plot(data_series=train["SalePrice"], label="SalePrice", title="SalePrice")
# log scaling distribution

dist_plot(data_series=np.log(train["SalePrice"]), label="SalePrice with log scale", title="SalePrice with log scale")
# Data check

# train data

data_check(train["MSSubClass"])
# test data

data_check(test["MSSubClass"])
# Raw data distribution

dist_plot(data_series=train["MSSubClass"], label="MSSubClass", title="MSSubClass")
# Plot vs SalePrice

plot_corr(data_series=train["MSSubClass"], target_series=train["SalePrice"], label="MSSubClass", target_label="SalePrice", title="MSSubClass vs SapePrice")
# Data check

data_check(train["MSZoning"])
# Data check

data_check(test["MSZoning"])
# Plot vs SalePrice

bar_and_box_plot(train["MSZoning"], np.log(train["SalePrice"]), label="MSZoning", target_label="SalePrice with log scale", title="MSZoning")
# Data check

data_check(train["LotFrontage"])
# Data check

data_check(test["LotFrontage"])
# Raw data distribution

dist_plot(data_series=train["LotFrontage"], label="LotFrontage", title="LotFrontage")
# log scaling distribution

dist_plot(data_series=np.log(train["LotFrontage"]), label="LotFrontage with log scale", title="LotFrontage with log scale")
# Plot vs SalePrice

plot_corr(data_series=train["LotFrontage"], target_series=train["SalePrice"], label="LotFrontage", target_label="SalePrice", title="LotFrontage vs SapePrice")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["LotFrontage"]), target_series=np.log(train["SalePrice"]), label="LotFrontage with log scale", target_label="SalePrice with log scale", title="LotFrontage vs SapePrice")
train["LotFrontage"].fillna(0, inplace=True)

test["LotFrontage"].fillna(0, inplace=True)
# Data check

data_check(train["LotArea"])
# Data check

data_check(test["LotArea"])
# Raw data distribution

dist_plot(data_series=train["LotArea"], label="LotArea", title="LotArea")
# log scaling distribution

dist_plot(data_series=np.log(train["LotArea"]), label="LotArea with log scale", title="LotArea with log scale")
# Plot vs SalePrice

plot_corr(data_series=train["LotArea"], target_series=train["SalePrice"], label="LotArea", target_label="SalePrice", title="LotArea vs SapePrice")
# Plot vs SalePrice with log scale

plot_corr(data_series=np.log(train["LotArea"]), target_series=np.log(train["SalePrice"]), label="LotArea with log scale", target_label="SalePrice with log scale", title="LotArea vs SapePrice")
# Data check

data_check(train["Street"])
# Test data check

data_check(test["Street"])
# Plot vs SalePrice

bar_and_box_plot(train["Street"], np.log(train["SalePrice"]), label="Street", target_label="SalePrice with log scale", title="Street")
# Data check

data_check(train["Alley"])
# Data check

data_check(test["Alley"])
# Fill na by "No_alley"

train["Alley"].fillna("No_alley", inplace=True)

test["Alley"].fillna("No_alley", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["Alley"], np.log(train["SalePrice"]), label="Alley", target_label="SalePrice with log scale", title="Alley")
# Data check

data_check(train["LotShape"])
# Data check

data_check(test["LotShape"])
# Plot vs SalePrice

bar_and_box_plot(train["LotShape"], np.log(train["SalePrice"]), label="LotShape", target_label="SalePrice with log scale", title="LotShape")
# Data check

data_check(train["LandContour"])
# Data check

data_check(test["LandContour"])
# Plot vs SalePrice

bar_and_box_plot(train["LandContour"], np.log(train["SalePrice"]), label="LandContour", target_label="SalePrice with log scale", title="LandContour")
# Data check

data_check(train["Utilities"])
# Data check

data_check(test["Utilities"])
# Plot vs SalePrice

bar_and_box_plot(train["Utilities"], np.log(train["SalePrice"]), label="Utilities", target_label="SalePrice with log scale", title="Utilities")
# Data check

data_check(train["LotConfig"])
# Data check

data_check(test["LotConfig"])
# Plot vs SalePrice

bar_and_box_plot(train["LotConfig"], np.log(train["SalePrice"]), label="LotConfig", target_label="SalePrice with log scale", title="LotConfig")
# Data check

data_check(train["LandSlope"])
# Data check

data_check(test["LandSlope"])
# Plot vs SalePrice

bar_and_box_plot(train["LandSlope"], np.log(train["SalePrice"]), label="LandSlope", target_label="SalePrice with log scale", title="LandSlope")
# Data check

data_check(train["Neighborhood"])
# Data check

data_check(test["Neighborhood"])
# Plot vs SalePrice

bar_and_box_plot(train["Neighborhood"], np.log(train["SalePrice"]), label="Neighborhood", target_label="SalePrice with log scale", title="Neighborhood")
# Data check

data_check(train["Condition1"])
# Data check

data_check(test["Condition1"])
# Plot vs SalePrice

bar_and_box_plot(train["Condition1"], np.log(train["SalePrice"]), label="Condition1", target_label="SalePrice with log scale", title="Condition1")
# Data check

data_check(train["Condition2"])
# Data check

data_check(test["Condition2"])
# Plot vs SalePrice

bar_and_box_plot(train["Condition2"], np.log(train["SalePrice"]), label="Condition2", target_label="SalePrice with log scale", title="Condition2")
# Data check

data_check(train["BldgType"])
# Data check

data_check(test["BldgType"])
# Plot vs SalePrice

bar_and_box_plot(train["BldgType"], np.log(train["SalePrice"]), label="BldgType", target_label="SalePrice with log scale", title="BldgType")
# Data check

data_check(train["HouseStyle"])
# Data check

data_check(test["HouseStyle"])
# Plot vs SalePrice

bar_and_box_plot(train["HouseStyle"], np.log(train["SalePrice"]), label="HouseStyle", target_label="SalePrice with log scale", title="HouseStyle")
# Data check

data_check(train["OverallQual"])
# Data check

data_check(test["OverallQual"])
# Raw data distribution

dist_plot(data_series=train["OverallQual"], label="OverallQual", title="OverallQual")
# Plot vs SalePrice

plot_corr(data_series=train["OverallQual"], target_series=np.log(train["SalePrice"]), label="OverallQual", target_label="SalePrice with log scale", title="OverallQual vs SapePrice")
# Data check

data_check(train["OverallCond"])
# Data check

data_check(test["OverallCond"])
# Raw data distribution

dist_plot(data_series=train["OverallCond"], label="OverallCond", title="OverallCond")
# Plot vs SalePrice

plot_corr(data_series=train["OverallCond"], target_series=np.log(train["SalePrice"]), label="OverallCond", target_label="SalePrice with log scale", title="OverallCond vs SapePrice")
# Data check

data_check(train["YearBuilt"])
# Data check

data_check(test["YearBuilt"])
# Raw data distribution

dist_plot(data_series=train["YearBuilt"], label="YearBuilt", title="YearBuilt")
# Plot vs SalePrice

plot_corr(data_series=train["YearBuilt"], target_series=np.log(train["SalePrice"]), label="YearBuilt", target_label="SalePrice with log scale", title="YearBuilt vs SapePrice")
# Data check

data_check(train["YearRemodAdd"])
# Data check

data_check(test["YearRemodAdd"])
# Raw data distribution

dist_plot(data_series=train["YearRemodAdd"], label="YearRemodAdd", title="YearRemodAdd")
# Plot vs SalePrice

plot_corr(data_series=train["YearRemodAdd"], target_series=np.log(train["SalePrice"]), label="YearRemodAdd", target_label="SalePrice with log scale", title="YearRemodAdd vs SapePrice")
# Data check

data_check(train["RoofStyle"])
# Data check

data_check(test["RoofStyle"])
# Plot vs SalePrice

bar_and_box_plot(train["RoofStyle"], np.log(train["SalePrice"]), label="RoofStyle", target_label="SalePrice with log scale", title="RoofStyle")
# Data check

data_check(train["RoofMatl"])
# Data check

data_check(test["RoofMatl"])
# Plot vs SalePrice

bar_and_box_plot(train["RoofMatl"], np.log(train["SalePrice"]), label="RoofMatl", target_label="SalePrice with log scale", title="RoofMatl")
# Data check

data_check(train["Exterior1st"])
# Data check

data_check(test["Exterior1st"])
# Plot vs SalePrice

bar_and_box_plot(train["Exterior1st"], np.log(train["SalePrice"]), label="Exterior1st", target_label="SalePrice with log scale", title="Exterior1st")
test["Exterior1st"].fillna("Other", inplace=True)
# Data check

data_check(train["Exterior2nd"])
# Data check

data_check(test["Exterior2nd"])
# Plot vs SalePrice

bar_and_box_plot(train["Exterior2nd"], np.log(train["SalePrice"]), label="Exterior2nd", target_label="SalePrice with log scale", title="Exterior2nd")
# Data check

data_check(train["MasVnrType"])
# Data check

data_check(test["MasVnrType"])
# Plot vs SalePrice

bar_and_box_plot(train["MasVnrType"], np.log(train["SalePrice"]), label="MasVnrType", target_label="SalePrice with log scale", title="MasVnrType")
# Data check

data_check(train["MasVnrArea"])
# Data check

data_check(test["MasVnrArea"])
# Raw data distribution

dist_plot(data_series=train["MasVnrArea"], label="MasVnrArea", title="MasVnrArea")
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train.query("MasVnrArea!=0")["MasVnrArea"]), label="MasVnrArea with log scale", title="MasVnrArea")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["MasVnrArea"]), target_series=np.log(train["SalePrice"]), label="MasVnrArea with log scale", target_label="SalePrice with log scale", title="MasVnrArea vs SapePrice")
train["MasVnrArea"].fillna(0, inplace=True)

test["MasVnrArea"].fillna(0, inplace=True)
# Data check

data_check(train["ExterQual"])
# Data check

data_check(test["ExterQual"])
# Plot vs SalePrice

bar_and_box_plot(train["ExterQual"], np.log(train["SalePrice"]), label="ExterQual", target_label="SalePrice with log scale", title="ExterQual")
# Data check

data_check(train["ExterCond"])
# Data check

data_check(test["ExterCond"])
# Plot vs SalePrice

bar_and_box_plot(train["ExterCond"], np.log(train["SalePrice"]), label="ExterCond", target_label="SalePrice with log scale", title="ExterCond")
# Data check

data_check(train["Foundation"])
# Data check

data_check(test["Foundation"])
# Plot vs SalePrice

bar_and_box_plot(train["Foundation"], np.log(train["SalePrice"]), label="Foundation", target_label="SalePrice with log scale", title="Foundation")
# Data check

data_check(train["BsmtQual"])
# Data check

data_check(test["BsmtQual"])
# Plot vs SalePrice

bar_and_box_plot(train["BsmtQual"], np.log(train["SalePrice"]), label="BsmtQual", target_label="SalePrice with log scale", title="BsmtQual")
train["BsmtQual"].fillna("None", inplace=True)

test["BsmtQual"].fillna("None", inplace=True)
# Data check

data_check(train["BsmtCond"])
# Data check

data_check(test["BsmtCond"])
# Plot vs SalePrice

bar_and_box_plot(train["BsmtCond"], np.log(train["SalePrice"]), label="BsmtCond", target_label="SalePrice with log scale", title="BsmtCond")
train["BsmtCond"].fillna("None", inplace=True)

test["BsmtCond"].fillna("None", inplace=True)
# Data check

data_check(train["BsmtExposure"])
# Data check

data_check(test["BsmtExposure"])
# Plot vs SalePrice

bar_and_box_plot(train["BsmtExposure"], np.log(train["SalePrice"]), label="BsmtExposure", target_label="SalePrice with log scale", title="BsmtExposure")
train["BsmtExposure"].fillna("None", inplace=True)

test["BsmtExposure"].fillna("None", inplace=True)
# Data check

data_check(train["BsmtFinType1"])
# Data check

data_check(test["BsmtFinType1"])
# Plot vs SalePrice

bar_and_box_plot(train["BsmtFinType1"], np.log(train["SalePrice"]), label="BsmtFinType1", target_label="SalePrice with log scale", title="BsmtFinType1")
train["BsmtFinType1"].fillna("None", inplace=True)

test["BsmtFinType1"].fillna("None", inplace=True)
# Data check

data_check(train["BsmtFinSF1"])
# Data check

data_check(test["BsmtFinSF1"])
# Raw data distribution

dist_plot(data_series=train["BsmtFinSF1"], label="BsmtFinSF1", title="BsmtFinSF1")
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train.query("BsmtFinSF1!=0")["BsmtFinSF1"]), label="BsmtFinSF1 with log scale", title="BsmtFinSF1")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["BsmtFinSF1"]), target_series=np.log(train["SalePrice"]), label="BsmtFinSF1 with log scale", target_label="SalePrice with log scale", title="BsmtFinSF1 vs SapePrice")
test["BsmtFinSF1"].fillna(0, inplace=True)
# Data check

data_check(train["BsmtFinType2"])
# Data check

data_check(test["BsmtFinType2"])
# Plot vs SalePrice

bar_and_box_plot(train["BsmtFinType2"], np.log(train["SalePrice"]), label="BsmtFinType2", target_label="SalePrice with log scale", title="BsmtFinType2")
train["BsmtFinType2"].fillna("None", inplace=True)

test["BsmtFinType2"].fillna("None", inplace=True)
# Data check

data_check(train["BsmtFinSF2"])
# Data check

data_check(test["BsmtFinSF2"])
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train.query("BsmtFinSF2!=0")["BsmtFinSF2"]), label="BsmtFinSF2 with log scale", title="BsmtFinSF2")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["BsmtFinSF2"]), target_series=np.log(train["SalePrice"]), label="BsmtFinSF2 with log scale", target_label="SalePrice with log scale", title="BsmtFinSF2 vs SapePrice")
test["BsmtFinSF2"].fillna(0, inplace=True)
# Data check

data_check(train["BsmtUnfSF"])
# Data check

data_check(test["BsmtUnfSF"])
# Raw data distribution

dist_plot(data_series=train["BsmtUnfSF"], label="BsmtUnfSF", title="BsmtUnfSF")
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train.query("BsmtUnfSF!=0")["BsmtUnfSF"]), label="BsmtUnfSF with log scale", title="BsmtUnfSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["BsmtUnfSF"]), target_series=np.log(train["SalePrice"]), label="BsmtUnfSF with log scale", target_label="SalePrice with log scale", title="BsmtUnfSF vs SapePrice")
test["BsmtUnfSF"].fillna(0, inplace=True)
# Data check

data_check(train["TotalBsmtSF"])
# Data check

data_check(test["TotalBsmtSF"])
# Raw data distribution

dist_plot(data_series=train["TotalBsmtSF"], label="TotalBsmtSF", title="TotalBsmtSF")
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train.query("TotalBsmtSF!=0")["TotalBsmtSF"]), label="TotalBsmtSF with log scale", title="TotalBsmtSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["TotalBsmtSF"]), target_series=np.log(train["SalePrice"]), label="TotalBsmtSF with log scale", target_label="SalePrice with log scale", title="TotalBsmtSF vs SapePrice")
# Data check

data_check(train["Heating"])
# Data check

data_check(test["Heating"])
# Plot vs SalePrice

bar_and_box_plot(train["Heating"], np.log(train["SalePrice"]), label="Heating", target_label="SalePrice with log scale", title="Heating")
# Data check

data_check(train["HeatingQC"])
# Data check

data_check(test["HeatingQC"])
# Plot vs SalePrice

bar_and_box_plot(train["HeatingQC"], np.log(train["SalePrice"]), label="HeatingQC", target_label="SalePrice with log scale", title="HeatingQC")
# Data check

data_check(train["CentralAir"])
# Data check

data_check(test["CentralAir"])
# Plot vs SalePrice

bar_and_box_plot(train["CentralAir"], np.log(train["SalePrice"]), label="CentralAir", target_label="SalePrice with log scale", title="CentralAir")
# Data check

data_check(train["Electrical"])
# Data check

data_check(test["Electrical"])
# Plot vs SalePrice

bar_and_box_plot(train["Electrical"], np.log(train["SalePrice"]), label="Electrical", target_label="SalePrice with log scale", title="Electrical")
train["Electrical"].fillna("SBrkr", inplace=True)
# Data check

data_check(train["1stFlrSF"])
# Data check

data_check(test["1stFlrSF"])
# Raw data distribution

dist_plot(data_series=train["1stFlrSF"], label="1stFlrSF", title="1stFlrSF")
# Raw data distribution

dist_plot(data_series=np.log(train["1stFlrSF"]), label="1stFlrSF", title="1stFlrSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["1stFlrSF"]), target_series=np.log(train["SalePrice"]), label="1stFlrSF with log scale", target_label="SalePrice with log scale", title="1stFlrSF vs SapePrice")
# Data check

data_check(train["2ndFlrSF"])
# Data check

data_check(test["2ndFlrSF"])
# Raw data distribution

dist_plot(data_series=train["2ndFlrSF"], label="2ndFlrSF", title="2ndFlrSF")
# Distribution of numbers other than 0 with log scale

dist_plot(data_series=np.log(train[train["2ndFlrSF"]!=0]["2ndFlrSF"]), label="2ndFlrSF with log scale", title="2ndFlrSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["2ndFlrSF"]), target_series=np.log(train["SalePrice"]), label="2ndFlrSF with log scale", target_label="SalePrice with log scale", title="2ndFlrSF vs SapePrice")
# Data check

data_check(train["LowQualFinSF"])
# Data check

data_check(test["LowQualFinSF"])
# Raw data distribution

dist_plot(data_series=train["LowQualFinSF"], label="LowQualFinSF", title="LowQualFinSF")
# Raw data distribution

dist_plot(data_series=np.log(train[train["LowQualFinSF"]!=0]["LowQualFinSF"]), label="LowQualFinSF", title="LowQualFinSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["LowQualFinSF"]), target_series=np.log(train["SalePrice"]), label="LowQualFinSF with log scale", target_label="SalePrice with log scale", title="LowQualFinSF vs SapePrice")
# Data check

data_check(train["GrLivArea"])
# Data check

data_check(test["GrLivArea"])
# Raw data distribution

dist_plot(data_series=train["GrLivArea"], label="GrLivArea", title="GrLivArea")
# Raw data distribution

dist_plot(data_series=np.log(train["GrLivArea"]), label="GrLivArea", title="GrLivArea")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["GrLivArea"]), target_series=np.log(train["SalePrice"]), label="GrLivArea with log scale", target_label="SalePrice with log scale", title="GrLivArea vs SapePrice")
# Data check

data_check(train["BsmtFullBath"])
# Data check

data_check(test["BsmtFullBath"])
# Raw data distribution

dist_plot(data_series=train["BsmtFullBath"], label="BsmtFullBath", title="BsmtFullBath")
# Plot vs SalePrice

plot_corr(data_series=train["BsmtFullBath"], target_series=np.log(train["SalePrice"]), label="BsmtFullBath", target_label="SalePrice with log scale", title="BsmtFullBath vs SapePrice")
test["BsmtFullBath"].fillna(0, inplace=True)
# Data check

data_check(train["BsmtHalfBath"])
# Data check

data_check(test["BsmtHalfBath"])
# Raw data distribution

dist_plot(data_series=train["BsmtHalfBath"], label="BsmtHalfBath", title="BsmtHalfBath")
# Plot vs SalePrice

plot_corr(data_series=train["BsmtHalfBath"], target_series=np.log(train["SalePrice"]), label="BsmtHalfBath", target_label="SalePrice with log scale", title="BsmtHalfBath vs SapePrice")
test["BsmtHalfBath"].fillna(0, inplace=True)
# Data check

data_check(train["FullBath"])
# Data check

data_check(test["FullBath"])
# Raw data distribution

dist_plot(data_series=train["FullBath"], label="FullBath", title="FullBath")
# Plot vs SalePrice

plot_corr(data_series=train["FullBath"], target_series=np.log(train["SalePrice"]), label="FullBath", target_label="SalePrice with log scale", title="FullBath vs SapePrice")
# Data check

data_check(train["HalfBath"])
# Data check

data_check(test["HalfBath"])
# Raw data distribution

dist_plot(data_series=train["HalfBath"], label="HalfBath", title="HalfBath")
# Plot vs SalePrice

plot_corr(data_series=train["HalfBath"], target_series=np.log(train["SalePrice"]), label="HalfBath", target_label="SalePrice with log scale", title="HalfBath vs SapePrice")
# Data check

data_check(train["BedroomAbvGr"])
# Data check

data_check(test["BedroomAbvGr"])
# Raw data distribution

dist_plot(data_series=train["BedroomAbvGr"], label="BedroomAbvGr", title="BedroomAbvGr")
# Plot vs SalePrice

plot_corr(data_series=train["BedroomAbvGr"], target_series=np.log(train["SalePrice"]), label="BedroomAbvGr", target_label="SalePrice with log scale", title="BedroomAbvGr vs SapePrice")
# Data check

data_check(train["KitchenAbvGr"])
# Data check

data_check(test["KitchenAbvGr"])
# Raw data distribution

dist_plot(data_series=train["KitchenAbvGr"], label="KitchenAbvGr", title="KitchenAbvGr")
# Plot vs SalePrice

plot_corr(data_series=train["KitchenAbvGr"], target_series=np.log(train["SalePrice"]), label="KitchenAbvGr", target_label="SalePrice with log scale", title="KitchenAbvGr vs SapePrice")
# Data check

data_check(train["KitchenQual"])
# Data check

data_check(test["KitchenQual"])
# Plot vs SalePrice

bar_and_box_plot(train["KitchenQual"], np.log(train["SalePrice"]), label="KitchenQual", target_label="SalePrice with log scale", title="KitchenQual")
test["KitchenQual"].fillna("TA", inplace=True)
# Data check

data_check(train["TotRmsAbvGrd"])
# Data check

data_check(test["TotRmsAbvGrd"])
# Raw data distribution

dist_plot(data_series=train["TotRmsAbvGrd"], label="TotRmsAbvGrd", title="TotRmsAbvGrd")
# Plot vs SalePrice

plot_corr(data_series=train["TotRmsAbvGrd"], target_series=np.log(train["SalePrice"]), label="TotRmsAbvGrd", target_label="SalePrice with log scale", title="TotRmsAbvGrd vs SapePrice")
# Data check

data_check(train["Functional"])
# Data check

data_check(test["Functional"])
# Plot vs SalePrice

bar_and_box_plot(train["Functional"], np.log(train["SalePrice"]), label="Functional", target_label="SalePrice with log scale", title="Functional")
test["Functional"].fillna("Typ", inplace=True)
# Data check

data_check(train["Fireplaces"])
# Data check

data_check(test["Fireplaces"])
# Raw data distribution

dist_plot(data_series=train["Fireplaces"], label="Fireplaces", title="Fireplaces")
# Plot vs SalePrice

plot_corr(data_series=train["Fireplaces"], target_series=np.log(train["SalePrice"]), label="Fireplaces", target_label="SalePrice with log scale", title="Fireplaces vs SapePrice")
# Data check

data_check(train["FireplaceQu"])
# Data check

data_check(test["FireplaceQu"])
# Plot vs SalePrice

bar_and_box_plot(train["FireplaceQu"], np.log(train["SalePrice"]), label="FireplaceQu", target_label="SalePrice with log scale", title="FireplaceQu")
train["FireplaceQu"].fillna("None", inplace=True)

test["FireplaceQu"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["FireplaceQu"], np.log(train["SalePrice"]), label="FireplaceQu", target_label="SalePrice with log scale", title="FireplaceQu")
# Data check

data_check(train["GarageType"])
# Data check

data_check(test["GarageType"])
# Plot vs SalePrice

bar_and_box_plot(train["GarageType"], np.log(train["SalePrice"]), label="GarageType", target_label="SalePrice with log scale", title="GarageType")
train["GarageType"].fillna("None", inplace=True)

test["GarageType"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["GarageType"], np.log(train["SalePrice"]), label="GarageType", target_label="SalePrice with log scale", title="GarageType")
# Data check

data_check(train["GarageYrBlt"])
# Data check

data_check(test["GarageYrBlt"])
# Raw data distribution

dist_plot(data_series=train["GarageYrBlt"], label="GarageYrBlt", title="GarageYrBlt")
# Plot vs SalePrice

plot_corr(data_series=train["GarageYrBlt"], target_series=np.log(train["SalePrice"]), label="GarageYrBlt", target_label="SalePrice with log scale", title="GarageYrBlt vs SapePrice")
# Data check

data_check(train["GarageFinish"])
# Data check

data_check(test["GarageFinish"])
# Plot vs SalePrice

bar_and_box_plot(train["GarageFinish"], np.log(train["SalePrice"]), label="GarageFinish", target_label="SalePrice with log scale", title="GarageFinish")
train["GarageFinish"].fillna("None", inplace=True)

test["GarageFinish"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["GarageFinish"], np.log(train["SalePrice"]), label="GarageFinish", target_label="SalePrice with log scale", title="GarageFinish")
# Data check

data_check(train["GarageCars"])
# Data check

data_check(test["GarageCars"])
dist_plot(data_series=train["GarageCars"], label="GarageCars", title="GarageCars")
# Plot vs SalePrice

plot_corr(data_series=train["GarageCars"], target_series=np.log(train["SalePrice"]), label="GarageCars", target_label="SalePrice with log scale", title="GarageCars vs SapePrice")
test[(test["GarageCars"].isnull())][["GarageYrBlt", "GarageType","GarageCars"]]
test["GarageCars"].fillna(0, inplace=True)
# Data check

data_check(train["GarageArea"])
# Data check

data_check(test["GarageArea"])
dist_plot(data_series=train["GarageArea"], label="GarageArea", title="GarageArea")
# Plot vs SalePrice

plot_corr(data_series=train["GarageArea"], target_series=np.log(train["SalePrice"]), label="GarageArea", target_label="SalePrice with log scale", title="GarageArea vs SapePrice")
# Plot vs SalePrice

plot_corr(data_series=train["GarageArea"], target_series=train["GarageCars"], label="GarageArea", target_label="GarageCars", title="GarageArea vs GarageCars")
test["GarageArea"].fillna(0, inplace=True)
# Data check

data_check(train["GarageQual"])
# Data check

data_check(test["GarageQual"])
# Plot vs SalePrice

bar_and_box_plot(train["GarageQual"], np.log(train["SalePrice"]), label="GarageQuale", target_label="SalePrice with log scale", title="GarageQuale")
train["GarageQual"].fillna("None", inplace=True)

test["GarageQual"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["GarageQual"], np.log(train["SalePrice"]), label="GarageQuale", target_label="SalePrice with log scale", title="GarageQuale")
# Data check

data_check(train["GarageCond"])
# Data check

data_check(test["GarageCond"])
# Plot vs SalePrice

bar_and_box_plot(train["GarageCond"], np.log(train["SalePrice"]), label="GarageCond", target_label="SalePrice with log scale", title="GarageCond")
train["GarageCond"].fillna("None", inplace=True)

test["GarageCond"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["GarageCond"], np.log(train["SalePrice"]), label="GarageCond", target_label="SalePrice with log scale", title="GarageCond")
# Data check

data_check(train["PavedDrive"])
# Data check

data_check(test["PavedDrive"])
# Plot vs SalePrice

bar_and_box_plot(train["PavedDrive"], np.log(train["SalePrice"]), label="PavedDrive", target_label="SalePrice with log scale", title="PavedDrive")
# Data check

data_check(train["WoodDeckSF"])
# Data check

data_check(test["WoodDeckSF"])
dist_plot(data_series=train["WoodDeckSF"], label="WoodDeckSF", title="WoodDeckSF")
dist_plot(data_series=np.log(train[train["WoodDeckSF"]!=0]["WoodDeckSF"]), label="WoodDeckSF", title="WoodDeckSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["WoodDeckSF"]), target_series=np.log(train["SalePrice"]), label="WoodDeckSF", target_label="SalePrice with log scale", title="WoodDeckSF vs SapePrice")
# Data check

data_check(train["OpenPorchSF"])
# Data check

data_check(test["OpenPorchSF"])
dist_plot(data_series=train["OpenPorchSF"], label="OpenPorchSF", title="OpenPorchSF")
dist_plot(data_series=np.log(train[train["OpenPorchSF"]!=0]["OpenPorchSF"]), label="OpenPorchSF", title="OpenPorchSF")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["OpenPorchSF"]), target_series=np.log(train["SalePrice"]), label="OpenPorchSF", target_label="SalePrice with log scale", title="OpenPorchSF vs SapePrice")
# Data check

data_check(train["EnclosedPorch"])
# Data check

data_check(test["EnclosedPorch"])
dist_plot(data_series=train["EnclosedPorch"], label="EnclosedPorch", title="EnclosedPorch")
dist_plot(data_series=np.log(train[train["EnclosedPorch"]!=0]["EnclosedPorch"]), label="EnclosedPorch", title="EnclosedPorch")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["EnclosedPorch"]), target_series=np.log(train["SalePrice"]), label="EnclosedPorch", target_label="SalePrice with log scale", title="EnclosedPorch vs SapePrice")
# Data check

data_check(train["3SsnPorch"])
# Data check

data_check(test["3SsnPorch"])
dist_plot(data_series=train["3SsnPorch"], label="3SsnPorch", title="3SsnPorch")
dist_plot(data_series=np.log(train[train["3SsnPorch"]!=0]["3SsnPorch"]), label="3SsnPorch", title="3SsnPorch")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["3SsnPorch"]), target_series=np.log(train["SalePrice"]), label="3SsnPorch", target_label="SalePrice with log scale", title="3SsnPorch vs SapePrice")
# Data check

data_check(train["ScreenPorch"])
# Data check

data_check(test["ScreenPorch"])
dist_plot(data_series=train["ScreenPorch"], label="ScreenPorch", title="ScreenPorch")
dist_plot(data_series=np.log(train[train["ScreenPorch"]!=0]["ScreenPorch"]), label="ScreenPorch", title="ScreenPorch")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["ScreenPorch"]), target_series=np.log(train["SalePrice"]), label="ScreenPorch", target_label="SalePrice with log scale", title="ScreenPorchh vs SapePrice")
# Data check

data_check(train["PoolArea"])
# Data check

data_check(test["PoolArea"])
dist_plot(data_series=train["PoolArea"], label="PoolArea", title="PoolArea")
dist_plot(data_series=np.log(train[train["PoolArea"]!=0]["PoolArea"]), label="PoolArea", title="PoolArea")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["PoolArea"]), target_series=np.log(train["SalePrice"]), label="PoolArea", target_label="SalePrice with log scale", title="PoolArea vs SapePrice")
# Data check

data_check(train["PoolQC"])
# Data check

data_check(test["PoolQC"])
# Plot vs SalePrice

bar_and_box_plot(train["PoolQC"], np.log(train["SalePrice"]), label="PoolQC", target_label="SalePrice with log scale", title="PoolQC")
# Data check

data_check(train["Fence"])
# Data check

data_check(test["Fence"])
# Plot vs SalePrice

bar_and_box_plot(train["Fence"], np.log(train["SalePrice"]), label="Fence", target_label="SalePrice with log scale", title="Fence")
train["Fence"].fillna("None", inplace=True)

test["Fence"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["Fence"], np.log(train["SalePrice"]), label="Fence", target_label="SalePrice with log scale", title="Fence")
# Data check

data_check(train["MiscFeature"])
# Data check

data_check(test["MiscFeature"])
# Plot vs SalePrice

bar_and_box_plot(train["MiscFeature"], np.log(train["SalePrice"]), label="MiscFeature", target_label="SalePrice with log scale", title="MiscFeature")
train["MiscFeature"].fillna("None", inplace=True)

test["MiscFeature"].fillna("None", inplace=True)
# Plot vs SalePrice

bar_and_box_plot(train["MiscFeature"], np.log(train["SalePrice"]), label="MiscFeature", target_label="SalePrice with log scale", title="MiscFeature")
# Data check

data_check(train["MiscVal"])
# Data check

data_check(test["MiscVal"])
dist_plot(data_series=train["MiscVal"], label="MiscVal", title="MiscVal")
dist_plot(data_series=np.log(train[train["MiscVal"]!=0]["MiscVal"]), label="MiscValF", title="MiscVal")
# Plot vs SalePrice

plot_corr(data_series=np.log(train["MiscVal"]), target_series=np.log(train["SalePrice"]), label="MiscVal", target_label="SalePrice with log scale", title="MiscVal vs SapePrice")
# Data check

data_check(train["MoSold"])
# Data check

data_check(test["MoSold"])
dist_plot(data_series=train["MoSold"], label="MoSold", title="MoSold")
# Plot vs SalePrice

plot_corr(data_series=train["MoSold"], target_series=np.log(train["SalePrice"]), label="MoSold", target_label="SalePrice with log scale", title="MoSold vs SapePrice")
# Data check

data_check(train["YrSold"])
# Data check

data_check(test["YrSold"])
dist_plot(data_series=train["YrSold"], label="YrSold", title="YrSold")
# Plot vs SalePrice

plot_corr(data_series=train["YrSold"], target_series=np.log(train["SalePrice"]), label="YrSold", target_label="SalePrice with log scale", title="YrSold vs SapePrice")
# Data check

data_check(train["SaleType"])
# Data check

data_check(test["SaleType"])
# Plot vs SalePrice

bar_and_box_plot(train["SaleType"], np.log(train["SalePrice"]), label="SaleType", target_label="SalePrice with log scale", title="SaleType")
test["SaleType"].fillna("WD", inplace=True)
# Data check

data_check(train["SaleCondition"])
# Data check

data_check(test["SaleCondition"])
# Plot vs SalePrice

bar_and_box_plot(train["SaleCondition"], np.log(train["SalePrice"]), label="SaleCondition", target_label="SalePrice with log scale", title="SaleCondition")