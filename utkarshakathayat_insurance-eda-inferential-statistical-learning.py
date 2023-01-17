##Define working directories

# Importing numpy library 

import numpy as np

# Importing pandas for data processing, file loading (e.g. pd.read_csv)

import pandas as pd
#Import the dataset

insurance = pd.read_csv("../input/insurance-dataset/insurance.csv")
#To find what all columns it contains, of what types and if they contain any value in it or not we use info() function.

insurance.info()
# Return the first five observation from the data set , to check what is there in the dataset

insurance.head()
# Now, We  Check the Shape of the data , it means how many rows, columns are present in the dataset

insurance.shape
#The dataset has 1338 rows and 7 columns , by dropping duplicate rows

#we will get the exact rows of our dataset

insurance.drop_duplicates(keep='first' , inplace= True)
insurance.shape
#Datatype of each attribute

insurance.dtypes
#Check for null values

insurance.isnull().sum()
#Importing matplotlib and Seaborn libraries

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline
#Check for the null values in the graph

sns.heatmap(insurance.isnull(),cbar=False,cmap='Reds')
#There is no null value in the dataset , Now we will check for the five point summary of NUMERICAL attributes 

insurance.describe()
#Check individually for each NUMERICAL attribute

#CHARGES

charges = sns.distplot(insurance['charges'], color="green", kde=True)
#AGE

age = sns.distplot(insurance['age'], color="blue", kde=True)

#CHILDREN

children  = sns.distplot(insurance['children'], color="slategray", kde=True)

#BMI

BMI  = sns.distplot(insurance['bmi'], color="magenta", kde=True)
#Boxplot : Used to detect the outliers in  a data set.

#BMI

bmi1 = sns.boxplot(insurance['bmi'], color="yellow")

bmi1.set_xlabel("Body mass index",fontsize=15)

#AGE

age1 = sns.boxplot(insurance['age'], color="pink")

age1.set_xlabel("AGE",fontsize=15)

#CHARGES

char1 = sns.boxplot(insurance['charges'], color="orange")

char1.set_xlabel("Charges",fontsize=15)
dataFrame = pd.DataFrame(insurance);

#Skewness is computed for each row or each column , here we will check for column

skewValue = dataFrame.skew(axis=0) # axis=0 for column

print("INSURANCE:")

print(insurance.head())

print("SKEW:")

print(skewValue)
#Let's visualize one by one all the categorical columns and after that we will go to the relation between the data

Smoker = sns.countplot(data = insurance, x = 'smoker')

Smoker.set_xlabel('Smoker', fontsize=15)
Gender = sns.countplot(data = insurance, x = 'sex')

Gender.set_xlabel('Sex', fontsize=15)

US_region = sns.countplot(data = insurance, x = 'region')

US_region.set_xlabel('region', fontsize=15)
Children = sns.countplot(data = insurance, x = 'children')

Children.set_xlabel('children', fontsize=15)
#Check pairplot for sex

sns.pairplot(insurance, hue='sex')
#Check pairplot for region

sns.pairplot(insurance, hue='region')
##Check pairplot for  children

sns.pairplot(insurance, hue='children')
#Check pairplot for the dataset 

#sns.pairplot(insurance)

sns.pairplot(insurance,hue = 'smoker',diag_kind = "kde",kind = "scatter",palette = "husl")
#Correlation of the dataset attributes

insurance.corr()
sns.heatmap(insurance.corr(),cmap='YlGnBu',annot=True , linecolor='white',linewidths=1)
sns.barplot(x='smoker',y='charges',data=insurance)
# To run an Independent Sample T-Test using python , let us first generate two samples 'a' and 'b'

a = np.array(insurance[insurance.smoker == 'yes'].charges) #People who smokes

b = np.array(insurance[insurance.smoker == 'no'].charges) #People who do not smoke

#Using the seaborn python library to generate a histogram of our 2 samples outputs the following.

sns.kdeplot(a, shade=False)

sns.kdeplot(b, shade=True)

plt.title("Independent Sample T-Test")
from scipy import stats

tStat, pValue = stats.ttest_ind(a, b, equal_var=False)

# ttest_ind function runs the independent sample T-Test and outputs a P-Value and the Test-Statistic.



print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic
# Male and Female counts who are non smokers

sns.countplot('smoker', hue='sex', data=insurance[insurance.smoker == 'no'])
# Male and Female counts who are non smokers

sns.countplot('smoker', hue='sex', data=insurance[insurance.smoker == 'yes'])
# We'll assign 'sex' and 'Smoker' to a new dataframe 'Smoker_Gender'.

import math

Smoker_Gender = insurance[['sex', 'smoker']]

Smoker_Gender.head()
Smoker_Gender['sex'].value_counts() #To find out the total numbers of males and females
Smoker_Gender['smoker'].value_counts() #To find out the total numbers of smokers and non smokers
contingency_table = pd.crosstab(

    Smoker_Gender['sex'],

    Smoker_Gender['smoker'],

    #margins = True

)

contingency_table 
stats.chi2_contingency(contingency_table)

# chi2_contingency() method conducts the Chi-square test on a contingency table 
#Construct a dataframe with only 'female' gender 

df= insurance[insurance.sex == 'female'] 

df.tail()
Dist_Female = df[['sex' , 'bmi' , 'children']]

Dist_Female
import scipy.stats as ss

ss.f_oneway(Dist_Female['bmi'][Dist_Female['children'] == 0], 

             Dist_Female['bmi'][Dist_Female['children'] == 1],

             Dist_Female['bmi'][Dist_Female['children'] == 2])

#Since we have to check distribution for women with no children, one child and two children 
sns.barplot(x='sex',y='bmi',data=insurance)
# To run an Independent Sample T-Test using python , let us first generate two samples 'a' and 'b'

a = np.array(insurance[insurance.sex == 'female'].bmi) # Females BMI

b = np.array(insurance[insurance.sex == 'male'].bmi) #Males BMI

#Using the seaborn python library to generate a histogram of our 2 samples outputs the following.

sns.kdeplot(a, shade=False)

sns.kdeplot(b, shade=True)

plt.title("Independent Sample T-Test")
from scipy import stats

tStat, pValue = stats.ttest_ind(a, b)

# ttest_ind function runs the independent sample T-Test and outputs a P-Value and the Test-Statistic.



print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic