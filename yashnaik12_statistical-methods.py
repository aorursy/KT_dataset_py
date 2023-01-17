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
df=pd.read_csv("/kaggle/input/statistical-learning-dataset/cs1.csv")

df.head()
male=df[df["gender"]=="MALE"]

female=df[df["gender"]=="FEMALE"]
male.income.mean()
female.income.mean()
## 1) the sample is drawn highly randomised

## 2) normality of income
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(df["income"])
import scipy.stats as st

df.income.describe()
st.shapiro(df.income)
st.mannwhitneyu(male.income,female.income)
st.levene(male.income,female.income)
# since p value is greater than alpha(5%) accept H0; means variance of male income and female income are same.

st.ttest_ind(male.income,female.income)
Yes=df[df["married"]=="YES"]

No=df[df["married"]=="NO"]
st.mannwhitneyu(Yes.income,No.income)
st.levene(Yes.income,No.income)
st.ttest_ind(Yes.income,No.income)
ipl=df[df["pl"]=="YES"]



inpl=df[df["pl"]=="NO"]
st.mannwhitneyu(ipl.income,inpl.income)
st.levene(ipl.income,inpl.income)
st.ttest_ind(ipl.income,inpl.income)
## Assumptions : 1.Randomness 2.Normality 3.Variance equality

## 1. assumed randomness

## 2. Normality is already checked,assume for a while it follows noemal

## 3. Variance equality (need to check)
df.region.unique()
iic=df[df["region"]=="INNER_CITY"]["income"]

it=df[df["region"]=="TOWN"]["income"]

ir=df[df["region"]=="RURAL"]["income"]

isu=df[df["region"]=="SUBURBAN"]["income"]

st.levene(iic,it,ir,isu)
sns.boxplot(x="region",y="income",data=df)
st.f_oneway(iic,it,ir,isu)
## since income is not satisfied the normality, we should do kruskal wallis test
st.kruskal(iic,it,ir,isu)
c0=df[df["children"]==0]["income"]

c1=df[df["children"]==1]["income"]

c2=df[df["children"]==2]["income"]

c3=df[df["children"]==3]["income"]

st.levene(c0,c1,c2,c3)
st.f_oneway(c0,c1,c2,c3)
## Z proportion test

from statsmodels.stats.proportion import proportions_ztest
tab=pd.crosstab(df["pl"],df["gender"])

tab
proportions_ztest([86,62],[170,160])
tab.loc["YES"]
proportions_ztest(tab.loc["YES"],tab.sum(axis=0))
st.chi2_contingency(tab)
tab1=pd.crosstab(df["pl"],df["married"])

tab1
proportions_ztest([63,85],[116,214])
st.chi2_contingency(tab1)
from statsmodels.formula.api import ols
df = pd.get_dummies(df,columns=["pl","car"],drop_first=True)

lin_model=ols("income~pl_YES+car_YES",data=df).fit()

lin_model.summary()
from statsmodels.stats.anova import anova_lm
formula = 'income ~ pl_YES + car_YES'

model = ols(formula,df).fit()

aov_table = anova_lm(model, typ = 2)

aov_table
model1 = ols('income ~ pl_YES + car_YES',data = df).fit()

model1.summary()

from scipy.stats import f
f.sf(17.03,2,327)