# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib

matplotlib.use('Agg') # workaround, there may be a better way

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats.distributions as dist
#we'll import data now   

data = pd.read_csv('/kaggle/input/nhanes-2015-2016/NHANES.csv') #Data is available in my dataset

data.head()
data["SMQ020x"] = data.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan})

data["RIAGENDRx"] = data.RIAGENDR.replace({1: "Male", 2: "Female"})

data["RIAGENDRx"].head()
x = data.SMQ020x.dropna() == "Yes"
p = x.mean()
p
Standard_error=np.sqrt(.4 * (1 - .4)/ len(x))
Standard_error
test_statistic = (p - 0.4) / Standard_error

test_statistic
pvalue = 2 * dist.norm.cdf(-np.abs(test_statistic))

print(test_statistic, pvalue)
sm.stats.proportions_ztest(x.sum(), len(x), 0.4)



sm.stats.binom_test(x.sum(), len(x), 0.4)
dx = data[["SMQ020x", "RIDAGEYR", "RIAGENDRx"]].dropna()  # Drop missing values

dx = dx.loc[(dx.RIDAGEYR >= 20) & (dx.RIDAGEYR <= 25), :] # Restrict to people between 20 and 25 years old
# Summarize the data by caclculating the proportion of yes responses and the sample size

p = dx.groupby("RIAGENDRx")["SMQ020x"].agg([lambda z: np.mean(z=="Yes"), "size"])

p.columns = ["Smoke", "N"]

print(p)


# The pooled rate of yes responses, and the standard error of the estimated difference of proportions

p_comb = (dx.SMQ020x == "Yes").mean()

va = p_comb * (1 - p_comb)

se = np.sqrt(va * (1 / p.N.Female + 1 / p.N.Male))

se
# Calculate the test statistic and its p-value

test_stat = (p.Smoke.Female - p.Smoke.Male) / se

pvalue = 2*dist.norm.cdf(-np.abs(test_stat))

print(test_stat, pvalue)
dx_females = dx.loc[dx.RIAGENDRx=="Female", "SMQ020x"].replace({"Yes": 1, "No": 0})

dx_males = dx.loc[dx.RIAGENDRx=="Male", "SMQ020x"].replace({"Yes": 1, "No": 0})

sm.stats.ttest_ind(dx_females, dx_males) # prints test statistic, p-value, degrees of freedom
dx = data[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()

dx = dx.loc[(dx.RIDAGEYR >= 40) & (dx.RIDAGEYR <= 50) & (dx.RIAGENDRx == "Male"), :]

print(len(dx))

print(dx.BPXSY1.mean()) # prints mean blood pressure

sm.stats.ztest(dx.BPXSY1, value=120)  # prints test statistic, p-value
dx = data[["BPXSY1", "RIDAGEYR", "RIAGENDRx"]].dropna()

dx = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :]

bpx_female = dx.loc[dx.RIAGENDRx=="Female", "BPXSY1"]

bpx_male = dx.loc[dx.RIAGENDRx=="Male", "BPXSY1"]

print(bpx_female.mean(), bpx_male.mean()) # prints female mean, male mean

print(sm.stats.ztest(bpx_female, bpx_male)) # prints test statistic, p-value

print(sm.stats.ttest_ind(bpx_female, bpx_male)) # prints test statistic, p-value, degrees of freedom
dx = data[["BMXBMI", "RIDAGEYR", "RIAGENDRx"]].dropna()

data["agegrp"] = pd.cut(data.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])

data.groupby(["agegrp", "RIAGENDRx"])["BMXBMI"].agg(np.std).unstack()
for k, v in data.groupby("agegrp"):

    bmi_female = v.loc[v.RIAGENDRx=="Female", "BMXBMI"].dropna()

    bmi_female = sm.stats.DescrStatsW(bmi_female)

    bmi_male = v.loc[v.RIAGENDRx=="Male", "BMXBMI"].dropna()

    bmi_male = sm.stats.DescrStatsW(bmi_male)

    print(k)

    print("pooled: ", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='pooled'))

    print("unequal:", sm.stats.CompareMeans(bmi_female, bmi_male).ztest_ind(usevar='unequal'))

    print()