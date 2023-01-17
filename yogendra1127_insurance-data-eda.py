# Importing neccessary libraries for the project



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats
# Importing/Reading given CSV file using Pandas



Insurance_df = pd.read_csv("../input/insurance/insurance.csv")
# Reading header of Dataframe

Insurance_df.head()
#Reading tail of Dataframe

Insurance_df.tail()
# Shape of given data

Insurance_df.shape
# Data Type of all atributes in data set

Insurance_df.info()
# Data Type of all atributes in data set

Insurance_df.dtypes
# Check of missing values in data frame



Insurance_df.isna()
# 5 point summary of all attributes in data set



Insurance_df.describe().round(2)
Insurance_df.describe().transpose().round(2)
#Distribution of 'bmi'

Insurance_df['bmi'].hist(figsize=(5,5))
#Distribution of Age

Insurance_df['age'].hist(figsize=(5,5))
#Distribution of 'Charges'

Insurance_df['charges'].hist(figsize=(5,5))
# Skewness of 'bmi'

sns.distplot(Insurance_df['bmi'], color='g')

plt.title('Distribution of BMI')

plt.show()
#Skewness of Age

sns.distplot(Insurance_df['age'], color='r', rug=True)

plt.title('Distribution of Age')

plt.show()
# Skewness of 'Charges'

sns.distplot(Insurance_df['charges'],axlabel='Distribution of Charges')

plt.title('Distribution of Charges')

plt.show()
# Checking the presence of outliers in BMI using boxplot

sns.boxplot(Insurance_df['bmi'], orient='v', color='b')
# Checking the presence of outliers in 'age' attribute using boxplot

sns.boxplot(Insurance_df['age'], palette='Set2')
# Checking the presence of outliers in 'charges' using boxplot

sns.boxplot(Insurance_df['charges'], palette='OrRd', orient='h')
Insurance_df.head()
# Distribution of Categorical columns in data set (we have 4 such attributes here sex,smoker,region & children)

sns.countplot(x='sex',hue='smoker',data=Insurance_df)
#Distribution of categorical data to understand the interest of people of particular region.

sns.countplot(Insurance_df['sex'],hue=Insurance_df['region'], color='b')
#Distribution of categorical variable 'Sex' & 'Children'.

sns.countplot(x='sex',hue='children',data=Insurance_df, palette='RdYlGn')

plt.title('Count of Children Over Gender Category')

plt.show()
#Distribution of categorical variable 'Smoker' & 'Children'.

sns.countplot(x='smoker',hue='children',data=Insurance_df, palette='Set1_r')

plt.title('Count of Children Over Smoker Category')

plt.show()
# Will check the distribution of data using bar plot

sns.barplot(Insurance_df['sex'],Insurance_df['charges'],hue=Insurance_df['smoker'], color='g')

plt.title("Charges Vs Gender")

plt.show()
# To check number of smokers & non smokers in data set

sns.countplot(Insurance_df['smoker'])

plt.title('Count of Smokers')

plt.show()
Insurance_df['smoker'].value_counts()
Insurance_df[Insurance_df['smoker']=='yes']['charges'].mean()
Insurance_df[Insurance_df['smoker']=='no']['charges'].mean()
# Distribution of charges over sex & children using boxplot plot

sns.boxplot(x='sex',y='charges',hue='children', data=Insurance_df, palette='Set2_r')
# Distribution of age over region & children using boxplot plot

sns.boxplot(x='sex',y='age',hue='region', data=Insurance_df,color='r')

plt.title('Box Plot distribution of Sex, Age & region')

plt.show()
#Analysing the Distribution of Data using factor plot.

sns.factorplot(x="sex", y="charges", hue="region", col="smoker", data=Insurance_df, kind="swarm")
sns.factorplot(x="sex", y="bmi", hue="smoker",col="children", data=Insurance_df, kind="swarm")
# Analyze the data using pair plot

sns.pairplot(Insurance_df,diag_kind='kde')
Insurance_df.head()
# Calculate the number of smokers & non smokers in the data set

smokers = Insurance_df['smoker'].value_counts()

print("Number of Smokers & Non-Smokers are: ", [smokers]) # There are 274 smokers and 1064 non-smokers
# Average Charges for Smokers & Non-Smokeres



Charges_smoker = Insurance_df[Insurance_df['smoker']=='yes']['charges'].mean()

Charges_Nonsmoker = Insurance_df[Insurance_df['smoker']=='no']['charges'].mean()

print("Avg. Charges for Smoker is", round(Charges_smoker,2))

print("Avg. Charges for Non-smoker is", round(Charges_Nonsmoker,2))

#Standard Deviation for Smokers & Non-smokers

Sigma_smoker = Insurance_df[Insurance_df['smoker']=='yes']['charges'].std()

Sigma_Nonsmoker = Insurance_df[Insurance_df['smoker']=='no']['charges'].std()

print("Std for Smoker is", round(Sigma_smoker,2))

print("Std Charges for Non-smoker is", round(Sigma_Nonsmoker,2))
#Visualize the data distribution using box plot...



sns.boxplot(Insurance_df['charges'],Insurance_df['smoker'],orient='h',palette='Set1_r')

plt.title("Distribution of Charges over Smokers")

plt.show()
Smoker_df = Insurance_df[['charges','smoker']]

Smoker_df.head()
Group1 = np.array(Smoker_df)

Group1
# Seperating the data into two groups

Group_smoker = Group1[:,1]=='yes'

Group_smoker = Group1[Group_smoker][:,0]

Group_Nonsmoker = Group1[:,1]=='no'

Group_Nonsmoker = Group1[Group_Nonsmoker][:,0]
# Now we will use two sample t-test on these groups assuming alpha =0.05

from scipy.stats import ttest_ind, shapiro,levene

from statsmodels.stats.power import ttest_power # importing neccessary libraries for test
t_stat, p_value = ttest_ind(Group_smoker,Group_Nonsmoker)



print(t_stat,p_value)
p_value.round(4)
shapiro(Group_smoker)
print ("two-sample t-test p-value=", p_value)
# Number of Males & Females in Data set



Gender_Count = Insurance_df['sex'].value_counts()

print ('Number of Males & Females ',Gender_Count)
# Now will calculate the mean BMI of both genders



bmi_male = Insurance_df[Insurance_df['sex']=='male']['bmi'].mean()

bmi_female = Insurance_df[Insurance_df['sex']=='female']['bmi'].mean()

print('Avg. BMI of Male is:- ', round(bmi_male,3))

print('Avg. BMI of Female is:- ', round(bmi_female,3))
#Visualize the data distribution using box plot...



sns.boxplot(Insurance_df['bmi'],Insurance_df['sex'], palette='Pastel2')

plt.title('Distribution of BMI')

plt.show()
# Create data frame for bmi & gender attributes

bmi_df = Insurance_df[['bmi','sex']]

bmi_df.head()
Group2 = np.array(bmi_df)

Group2
# Seperating the above array data in to two groups based on BMI and Gender

Group_male = Group2[:,1] == 'male'

Group_male = Group2[Group_male][:,0]

Group_female = Group2[:,1]=='female'

Group_female = Group2[Group_female][:,0]

print('Total Males:- ', len(Group_male))

print('Total Females:- ',len(Group_female))
# Now we will apply two sample t-test on these groups considering alpha value 0.05

# null hypothesis: the two groups have the same BMI

# this test assumes the two groups have the same variance..



t_stat_bmi, p_value_bmi = ttest_ind(Group_male,Group_female)

print ('t_stat for BMI is:- ', round(t_stat_bmi,4), "& p-Value for BMI is:- ", round(p_value_bmi,4))
shapiro(Group_male)
shapiro(Group_female)
levene(Group_male,Group_female)
print ("two-sample t-test p-value=",round(p_value_bmi,4))
female_smokers = Insurance_df[Insurance_df['sex']=='female']['smoker'].value_counts()[1]  # number of female smokers

male_smokers = Insurance_df[Insurance_df['sex']=='male']['smoker'].value_counts()[1]  # number of male smokers

n_females = Insurance_df['sex'].value_counts()[1] # Total number of females in data

n_males = Insurance_df['sex'].value_counts()[0]  #Total number of males in data



print([female_smokers, male_smokers] , [n_females, n_males])



print(f' Proportion of smokers in females, males = {round(115/662,2)}%, {round(159/676,2)}% respectively')
# Import neccessary libraries for proportion test



from statsmodels.stats.proportion import proportions_ztest



prop_stats, prop_p_value = proportions_ztest([female_smokers,male_smokers],[n_females,n_males])



if prop_p_value<0.05:

    print(f'With a p-value of {round(prop_p_value,4)} the difference is significant. Hence We reject the null Hypothesis')

else:

    print(f'With a p-value of {round(prop_p_value,4)} the difference is not significant. Hence We fail to reject the null Hypothesis')



print('Value of test proportion is', round(prop_stats,4))
# Read the CSV file to create Dataframe



Child_df = pd.read_csv("../input/insurance/insurance.csv")

Child_df.head()
# As we required only bmi, gender and children columns, we are dropping other columns from data frame

Child_df.drop(['smoker','region','charges','age'], axis=1, inplace=True)
Child_df.shape
Child_df['sex'].value_counts()
# Now we will select female entries having 0, 1 & 2 Children only



Final_df = Child_df[(Child_df['sex']=='female') & (Child_df['children']<3)]



print(Final_df)
Final_df.shape
Final_df['sex'].value_counts()
# Calculate the BMI mean, standard deviation & count of all three groups of female with (Zero, one & two child)

bmi_child0 = Final_df[Final_df['children']==0]['bmi']

bmi_child1 = Final_df[Final_df['children']==1]['bmi']

bmi_child2 = Final_df[Final_df['children']==2]['bmi']



print('Count, Mean and standard deviation of female having 0 children respectively: %3d, %3.2f and %3.2f' % (len(bmi_child0), bmi_child0.mean(),np.std(bmi_child0)))

print('Count, Mean and standard deviation of female having 1 children respectively: %3d, %3.2f and %3.2f' % (len(bmi_child1), bmi_child1.mean(),np.std(bmi_child1)))

print('Count, Mean and standard deviation of female having 2 children respectively: %3d, %3.2f and %3.2f' % (len(bmi_child2), bmi_child2.mean(),np.std(bmi_child2)))
# Let us explore the data by using boxplot

sns.boxplot(Final_df['children'], Final_df['bmi'])

plt.title('BMI of females having Children <=2')

plt.show()
# Import neccessary libraries & models for f-test

import statsmodels.api as sm

from statsmodels.formula.api import ols

mod = ols('bmi~children', data = Final_df).fit()

aov_table = sm.stats.anova_lm(mod,typ=2)

print(round(aov_table,3))
# import neccessary models to perform the test to identify the outlier



from statsmodels.stats.multicomp import pairwise_tukeyhsd



print(pairwise_tukeyhsd(Final_df['bmi'], Final_df['children']))