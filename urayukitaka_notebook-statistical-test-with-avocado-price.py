# Basic library

import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Statistics library

from scipy.stats import norm

from scipy import stats

import scipy

import statsmodels.formula.api as smf

from statsmodels.formula.api import ols

import statsmodels.api as sm

import statsmodels.stats.anova as anova



# random value

from numpy.random import *



# Visualization

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv", header=0)
df.head()
# Null value

df.isnull().sum()
sns.pairplot(df.sample(200))
# data

data = df["AveragePrice"]



# calculation of skew and kurtosis

skew = scipy.stats.skew(data)

kurt = scipy.stats.kurtosis(data)



# basic check with 

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(data, fit=norm, ax=ax[0])

ax[0].set_ylabel("frequency")

ax[0].set_title("Distribution plot\n<skewness:%.2f>\n<kurtosis:%.2f>" % (skew,kurt))

stats.probplot(data, plot=ax[1])

ax[1].set_title("Probability plot")
# data

data = np.log(df["AveragePrice"])



# calculation of skew and kurtosis

skew = scipy.stats.skew(data)

kurt = scipy.stats.kurtosis(data)



# basic check with 

fig, ax = plt.subplots(1,2,figsize=(20,6))

sns.distplot(data, fit=norm, ax=ax[0])

ax[0].set_ylabel("frequency")

ax[0].set_title("Distribution plot\n<skewness:%.2f>\n<kurtosis:%.2f>" % (skew,kurt))

stats.probplot(data, plot=ax[1])

ax[1].set_title("Probability plot")
# data

data = np.log(df["AveragePrice"])



# with stats model

WS, p = stats.shapiro(data.sample(4999))
print("p value:{}".format(p))
# data

data = np.log(df["AveragePrice"])



# with stats model

KS, p = stats.kstest(data, "norm")
print("p value:{}".format(p))
# data

data = np.log(df["AveragePrice"])



statistic, critical_values, significance_level = scipy.stats.anderson(data, "norm")
print("critical values:{}".format(critical_values))

print("significant_level:{}".format(significance_level))
# create norm data, about Shapiro Wilk test, N<5000

norm_data = randn(4999)



# Visualization

sns.distplot(norm_data)
# with stats model

WS, p = stats.shapiro(norm_data)

print("Shapiro Wilk test p value:{}".format(p))



# with stats model

KS, p = stats.kstest(norm_data, "norm")

print("Kolmogorov–Smirnov test p value:{}".format(p))



# with stats model

statistic, critical_values, significance_level = scipy.stats.anderson(norm_data, "norm")

print("critical values:{}".format(critical_values))

print("significant_level:{}".format(significance_level))
plt.figure(figsize=(10,6))

sns.boxplot(x="year", y="AveragePrice", data=df)

print("2015 average price:{}".format(df.query("year==2015")["AveragePrice"].mean()))

print("2016 average price:{}".format(df.query("year==2016")["AveragePrice"].mean()))
plt.figure(figsize=(10,6))

sns.distplot(df.query("year==2015")["AveragePrice"])

sns.distplot(df.query("year==2016")["AveragePrice"])

print("2015 average price:{}".format(df.query("year==2015")["AveragePrice"].mean()))

print("2016 average price:{}".format(df.query("year==2016")["AveragePrice"].mean()))
# data

price_2015 = np.log(df.query("year==2015")["AveragePrice"].values)

price_2016 = np.log(df.query("year==2016")["AveragePrice"].values)



# with stats model, 

stats.ttest_ind(price_2015, price_2016)
# with stats model, equal_var=False

stats.ttest_ind(price_2015, price_2016, equal_var=False)
# with stats model

stats.mannwhitneyu(price_2015, price_2016)
# pivot table

pivot = pd.pivot_table(df, index="type", columns="year", values="AveragePrice", aggfunc="mean")

pivot.head()
# visualization

plt.figure(figsize=(10,6))

plt.plot(pivot.T.index, pivot.T["conventional"])

plt.plot(pivot.T.index, pivot.T["organic"])

plt.xlabel("year")

plt.xticks([2015,2016,2017,2018])

plt.ylabel("Average price")

plt.yticks([0.5,1,1.5,2])
# stats model

x2, p, dof, expected = scipy.stats.chi2_contingency(pivot)
# result

print("x2:{}".format(x2))

print("p:{}".format(p))

print("dof:{}".format(dof))

print("expectd:\n{}".format(expected))
# data

price_2015 = df.query("year==2015")["AveragePrice"].values

price_2016 = df.query("year==2016")["AveragePrice"].values



# stats model

scipy.stats.bartlett(price_2015, price_2016)
# Visualization check

plt.figure(figsize=(10,6))

sns.distplot(price_2015)

sns.distplot(price_2016)

plt.xlabel("variance")

plt.title("Distribution \n variance at 2015 %.2f \n variance at 2016 %.2f" % (price_2015.var(), price_2016.var()))
plt.figure(figsize=(10,6))

sns.boxplot(x="year", y="AveragePrice", data=df)

print("2015 average price:{}".format(df.query("year==2015")["AveragePrice"].mean()))

print("2016 average price:{}".format(df.query("year==2016")["AveragePrice"].mean()))

print("2017 average price:{}".format(df.query("year==2017")["AveragePrice"].mean()))

print("2018 average price:{}".format(df.query("year==2018")["AveragePrice"].mean()))
# data

price_2015 = df.query("year==2015")["AveragePrice"].values

price_2016 = df.query("year==2016")["AveragePrice"].values

price_2017 = df.query("year==2017")["AveragePrice"].values

price_2018 = df.query("year==2018")["AveragePrice"].values



print("Shapiro Wilk test")

print("price_2015 p-value:{}:".format(stats.shapiro(price_2015[:4999])[1]))

print("price_2016 p-value:{}:".format(stats.shapiro(price_2016[:4999])[1]))

print("price_2017 p-value:{}:".format(stats.shapiro(price_2017[:4999])[1]))

print("price_2018 p-value:{}:".format(stats.shapiro(price_2018[:4999])[1]))
# Levene test

stats.levene(price_2015, price_2016, price_2017, price_2018)
# Reference) Bartlett test

stats.bartlett(price_2015, price_2016, price_2017, price_2018)
# Reference) Parametric version, if can be eaual variances.

f, p = stats.f_oneway(price_2015, price_2016, price_2017, price_2018)



print("p-value:{}".format(p))
# Check data frame summary

df.groupby(["year", "type"])["AveragePrice"].mean()
# Create dataframe

sample_data = df[["AveragePrice", "type", "year"]]

sample_data.head()
# Statsmodel

formula = 'AveragePrice ~ C(type)+C(year) + C(type):C(year)'



model = ols(formula, sample_data).fit()



# Result

model.summary()
aov_table = sm.stats.anova_lm(model, typ=2)

print(aov_table)