import numpy as np

import matplotlib.pyplot as mpb

%matplotlib inline

import pandas as pd

import seaborn as sns

from scipy import stats

from scipy.stats import chi2

from scipy.stats import f
# Data Description:

# The data at hand contains medical costs of people

# characterized by certain attributes.

# Domain:

# Healthcare

# Context:

# Leveraging customer information is paramount for most

# businesses. In the case of an insurance company, attributes of

# customers like the ones mentioned below can be crucial in

# making business decisions. Hence, knowing to explore and

# generate value out of such data can be an invaluable skill to

# have.

# Attribute Information:

# age: age of primary beneficiary

# sex: insurance contractor gender, female, male

# bmi: Body mass index, providing an understanding of body,

# weights that are relatively high or low relative to height,

# objective index of body weight (kg / m ^ 2) using the ratio of

# height to weight, ideally 18.5 to 24.9

# children: Number of children covered by health insurance /

# Number of dependents

# smoker: Smoking

# region: the beneficiary's residential area in the US, northeast,

# southeast, southwest, northwest.

# charges: Individual medical costs billed by health insurance.

# Learning Outcomes:

# ● Exploratory Data Analysis

# ● Practicing statistics using Python

# ● Hypothesis testing

# Objective:

# We want to see if we can dive deep into this data to find some

# valuable insights.

# Steps and tasks:
myData = pd.read_csv('../input/insurance.csv')

myData.head()
dataShape = myData.shape

print(dataShape)
dataTypes = myData.dtypes

print(dataTypes)
print(myData.isnull().values.any())
print(myData.isnull().sum())
print(myData.isnull())
myData.describe(include=[np.number])
myData.hist(column=['bmi','age','charges'],figsize=(20,20))
sns.distplot(myData['age'])
sns.distplot(myData['bmi'])
sns.distplot(myData['charges'])
myData.skew()
print(myData[{'age','bmi','charges'}].skew())
for column in myData[['age','bmi','charges']]:

    val = column

    q1 = myData[val].quantile(0.25)

    q3 = myData[val].quantile(0.75)

    iqr = q3-q1

    fence_low  = q1-(1.5*iqr)

    fence_high = q3+(1.5*iqr)

    df_out = myData.loc[(myData[val] < fence_low) | (myData[val] > fence_high)]

    if df_out.empty:

        print('No Outliers in the ' + val + ' column of given dataset')

    else:

        print('There are Outliers in the ' + val + ' column of given dataset')



print(sns.boxplot(myData['age']))
print(sns.boxplot(myData['bmi']))
print(sns.boxplot(myData['charges']))
print(sns.catplot(x="region", y="charges", hue="children", kind = "boxen", col = "sex", row = "smoker", data=myData))
print(sns.catplot(x="region", y="bmi", hue="children", kind = "boxen", col = "sex", row = "smoker", data=myData))
print(sns.catplot(x="region", y="age", hue="children", kind = "boxen", col = "sex", row = "smoker", data=myData))
print(sns.pairplot(myData))
df_anova = myData[['charges','smoker']]

grps = pd.unique(df_anova.smoker.values)

d_data = {grp:df_anova['charges'][df_anova.smoker == grp] for grp in grps}

F, p = stats.f_oneway(d_data['yes'], d_data['no'])

print("p-value for significance is: ", p)

if p<0.05:

    print("The charges of people differ between smokers and non-smokers. Reject null Hypothesis")

else:

    print("The charges of people is the same irrespective of the smoking habits. Accept null hypothesis")
df_anova = myData[['bmi','sex']]

grps = pd.unique(df_anova.sex.values)

d_data = {grp:df_anova['bmi'][df_anova.sex == grp] for grp in grps}

F, p = stats.f_oneway(d_data['male'], d_data['female'])

print("p-value for significance is: ", p)

if p<0.05:

    print("The bmi of male is different from that of the female. Reject null Hypothesis")

else:

    print("The bmi of male and female are the same. Accept null hypothesis")
contingency_table=pd.crosstab(myData["sex"],myData["smoker"])



Observed_Values = contingency_table.values 



b=stats.chi2_contingency(contingency_table)

Expected_Values = b[3]



no_of_rows=len(contingency_table.iloc[0:2,0])

no_of_columns=len(contingency_table.iloc[0,0:2])

ddof=(no_of_rows-1)*(no_of_columns-1)

print("Degree of Freedom: ",ddof)



alpha = 0.05



chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])

chi_square_statistic=chi_square[0]+chi_square[1]



critical_value=chi2.ppf(q=1-alpha,df=ddof)



p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value: ',p_value)

print('Significance level: ',alpha)

print('Degree of Freedom: ',ddof)

print('chi-square statistic: ',chi_square_statistic)

print('critical_value: ',critical_value)



if chi_square_statistic>=critical_value:

    print("The proportion of smokers is different in different genders. Reject null Hypothesis")

else:

    print("The proportion of smokers is the same in different genders. Accept null Hypothesis")

    

if p_value<=alpha:

    print("The proportion of smokers is different in different genders. Reject null Hypothesis")

else:

    print("The proportion of smokers is the same in different genders. Accept null Hypothesis")
female_smokers = myData[myData['sex'] == 'female'].smoker.value_counts()[1]

male_smokers = myData[myData['sex'] == 'male'].smoker.value_counts()[1]

n_females = myData.sex.value_counts()[1]

n_males = myData.sex.value_counts()[0]
print([female_smokers, male_smokers] , [n_females, n_males])

print(f' Proportion of smokers in females, males = {round(115/662,2)}%, {round(159/676,2)}% respectively')
from statsmodels.stats.proportion import proportions_ztest



stat, pval = proportions_ztest([female_smokers, male_smokers] , [n_females, n_males])



if pval < 0.05:

    print(f'With a p-value of {round(pval,4)} the proportion of smokers is different in different genders. Reject null Hypothesis')

else:

    print(f'With a p-value of {round(pval,4)} the proportion of smokers is the same in different genders. Accept null Hypothesis')
myD = myData[myData.sex == 'female']

df_anova = myD[['bmi','children']]

grps = pd.unique(df_anova.children.values)

d_data = {grp:df_anova['bmi'][df_anova.children == grp] for grp in grps}

F, p = stats.f_oneway(d_data[0], d_data[1], d_data[2])

print("p-value for significance is: ", p)

if p<0.05:

    print("The distribution of bmi across women with no children, 1 child and 2 child are different. Reject null Hypothesis")

else:

    print("The distribution of bmi across women with no children, 1 child and 2 child are the same. Accept null Hypothesis")
da = myD[myD.children < 3]

print(sns.catplot(x="children", y="bmi", kind = "box", data=da))