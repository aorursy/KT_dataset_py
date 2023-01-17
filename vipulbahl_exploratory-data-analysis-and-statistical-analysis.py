import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import statsmodels.stats.proportion as stats_pro

%matplotlib inline

sns.set(color_codes=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

insurance=pd.read_csv('/kaggle/input/insurance/insurance.csv')

insurance.head()
shape_insurance=insurance.shape

print('The shape of the dataframe insurance is',shape_insurance,'which means there are',shape_insurance[0],'rows and',shape_insurance[1],'columns.')
#The data type be found through the info function

insurance.info()
#The data type also be found through the dtype function

print('The data type of attribute age is',insurance['age'].dtype)

print('The data type of attribute sex is',insurance['sex'].dtype)

print('The data type of attribute bmi is',insurance['bmi'].dtype)

print('The data type of attribute children is',insurance['children'].dtype)

print('The data type of attribute smoker is',insurance['smoker'].dtype)

print('The data type of attribute region is',insurance['region'].dtype)

print('The data type of attribute charges is',insurance['charges'].dtype)
print('The missing values in the dataframe Insurance are:','\n',insurance.isnull().sum(),'\n','which means there are no null values in the dataset')
# Another way to find the missing values

print('The missing values in the attribute age are',insurance['age'].isnull().sum())

print('The missing values in the attribute age are',insurance['sex'].isnull().sum())

print('The missing values in the attribute age are',insurance['bmi'].isnull().sum())

print('The missing values in the attribute age are',insurance['children'].isnull().sum())

print('The missing values in the attribute age are',insurance['smoker'].isnull().sum())

print('The missing values in the attribute age are',insurance['region'].isnull().sum())

print('The missing values in the attribute age are',insurance['charges'].isnull().sum())

insurance.describe().T
insurance_5pt=insurance.describe().loc[['min','25%','50%','75%','max'],['age','bmi','children','charges']].T

print('The 5 point summary of numerical attribute is:','\n',insurance_5pt)
plt.hist(insurance['bmi'])

plt.xlabel('bmi')

plt.ylabel('count')

plt.title('Distribution of BMI')

plt.show()
plt.hist(insurance['age'])

plt.xlabel('age')

plt.ylabel('count')

plt.title('Distribution of Age')

plt.show()
plt.hist(insurance['charges'])

plt.xlabel('charges')

plt.ylabel('count')

plt.title('Distribution of Charges')

plt.show()
skewness_bmi=round(stats.skew(insurance['bmi']),4)

skewness_age=round(stats.skew(insurance['age']),4)

skewness_charges=round(stats.skew(insurance['charges']),4)



print(' The skewness of bmi is', skewness_bmi,'\n','The skewness of age is',skewness_age,'\n','The skewness of charges is',skewness_charges)
bmi_boxplot=sns.boxplot(insurance['bmi']);

print(' As seen in the previous step, skewness is very less for BMI', '\n','checking if there are any outliars by ploting a box plot.','\n' ,' As seen in the chart below,There are outliars on the right.','\n')

plt.show()
age_boxplot=sns.boxplot(insurance['age']);

print(' As checked in the previous step, there is negligible skewness in age.', '\n','Checking if there are any outliars by ploting a box plot.','\n' ,'As seen in the chart below, there doesnt seem to be an outliar.')

plt.show()
charges_boxplot=sns.boxplot(insurance['charges']);

print(' As seen in the above step, charges have high skewness.','\n' ,'Checking if there are any outliars by ploting a box plot.',  '\n' ,'There are outliars on the right.')

plt.show()
#The categorical columns are sex,smoker,region,children

# Distribution of Sex

sns.countplot(insurance['sex'])

plt.xlabel('Gender')

plt.ylabel('count')

plt.title('Distribution of genders')

plt.show()
# Distribution of smoker

sns.countplot(insurance['smoker'])

plt.xlabel('smoker')

plt.ylabel('count')

plt.title('Distribution of smoker')

plt.show()
# Distribution of region

sns.countplot(insurance['region'])

plt.xlabel('region')

plt.ylabel('count')

plt.title('Distribution of region')

plt.show()
# Distribution of children

sns.countplot(insurance['children'])

plt.xlabel('children')

plt.ylabel('count')

plt.title('Distribution of children')

plt.show()
#Pair plot doesnt contain display non numeric values. Hence, we will have to convert non numeric columns into numbers. 

#The non-numeric columns are sex, smoke and BMI.

insurance_pp=insurance.copy()

insurance_pp['sex']=insurance_pp['sex'].astype('category').cat.codes

insurance_pp['smoker']=insurance_pp['smoker'].astype('category').cat.codes

insurance_pp['region']=insurance_pp['region'].astype('category').cat.codes
sns.pairplot(insurance_pp);
id_smo_charges=np.array(insurance[['charges','smoker']])

id_smo_charges
## separating the charges paid by smokers and non-smokers



# identify charges paid by smokers

smo_charges = id_smo_charges[:,1]=='yes'

smo_charges = id_smo_charges[smo_charges][:,0]



# identify charges paid by non-smoker

non_smo_charges = id_smo_charges[:,1]=='no'

non_smo_charges = id_smo_charges[non_smo_charges][:,0]
t_statistics, p_value = stats.ttest_ind(smo_charges,non_smo_charges)
print(t_statistics, p_value)
# p_value < 0.05. Hence, the null hypothesis is rejected.

# which means that the charges of people who smoke differ significantly from the people who don't smoke.

print(' Two sample t-test p-value',p_value, 'is significantly less than alpha (0.05).','\n' ,'Hence the null hypothesis is rejected.','\n','Therefore charges of people who smoke differ from charges of people who dont smoke')
bmi_sex=np.array(insurance[['bmi','sex']])

bmi_sex
bmi_male=bmi_sex[:,1]=='male'

bmi_male=bmi_sex[bmi_male][:,0]

bmi_female=bmi_sex[:,1]=='female'

bmi_female=bmi_sex[bmi_female][:,0]
t_statistics, p_value = stats.ttest_ind(bmi_male,bmi_female)

print(t_statistics,p_value)
# p-value is greater than alpha (0.05). Hence, we fail to reject the null hypothesis. 



print(' Two sample t-test p-value is',round(p_value,6),'which is more than alpha (0.05).','\n'' Hence, we fail to reject the null hypothesis; which means that gender has no effect on BMI.')
# computing the number of males and females

male_count=insurance['sex'].value_counts()[0]

female_count=insurance['sex'].value_counts()[1]

print(' The total number of males is',male_count,'\n','The total number of females is',female_count)



# computing the number of male and female smokers

male = insurance['sex']=='male'

male_smoker = insurance[male].smoker.value_counts()[1]

female = insurance['sex']=='female'

female_smoker = insurance[female].smoker.value_counts()[1]

print(' The male smoker count is',male_smoker,'\n','The female smoker count is',female_smoker)

print(' The proportion of male smoker is',round(male_smoker/male_count,4),'\n','The proportion of female smoker is',round(female_smoker/female_count,4))
test_statistics,p_value=stats_pro.proportions_ztest([male_smoker,female_smoker],[male_count,female_count])



test_statistics,p_value



print(' The p-value is',round(p_value,4),'which is significantly lower than alpha(0.05).','\n' ,'Hence, the null hypothesis is rejected.','\n','Therefore, the proportion of smokers differ significantly in genders.')

#plotting a bar graph to analyse the distribution of BMI

sns.boxplot(data=insurance,x="children",y="bmi",hue="sex");

plt.title('Distribution of BMI')

plt.show()
## in this step we will segregate BMI for all females by the number of children (0,1,2)

bmi_sex=np.array(insurance[['sex','bmi','children']])

#identify all females

bmi_female=bmi_sex[bmi_sex[:,0]=='female']

#bmi for females with 0 children

z_bmi_female=bmi_female[bmi_female[:,2]==0][:,1]

#bmifor females with 1 child

o_bmi_female=bmi_female[bmi_female[:,2]==1][:,1]

#bmi for females with 2 children

t_bmi_female=bmi_female[bmi_female[:,2]==2][:,1]
f_stat,p_value=stats.f_oneway(z_bmi_female,o_bmi_female,t_bmi_female)

print('The statistics computed is',round(f_stat,4),'and the p-value computed is',round(p_value,4))
print(' The p-value is',round(p_value,4),', which is significantly larger than alpha(0.05).','\n','Hence we fail to reject the null hypothesis.','\n', 'Therefore, There is no significant evidence to conclude that BMI for women having 0,1 or 2 children is different.')