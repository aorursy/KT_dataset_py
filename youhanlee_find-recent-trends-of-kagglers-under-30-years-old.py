# Import Python packages
import numpy as np 
import pandas as pd 
import os
import math

import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 5000)


plt.style.use('seaborn')
sns.set(font_scale=2)
## 2017
multipleChoice_2017 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv",  encoding="ISO-8859-1", low_memory=False)
freeForm_2017 = pd.read_csv("../input/kaggle-survey-2017/freeformResponses.csv", low_memory=False)
schema_2017 = pd.read_csv("../input/kaggle-survey-2017/schema.csv", index_col="Column")


## 2018
schema_2018 = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv')
multipleChoice_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
freeForm_2018 = pd.read_csv('../input/kaggle-survey-2018/freeFormResponses.csv')
questions = multipleChoice_2018.loc[0, :]
multipleChoice_2018 = multipleChoice_2018[1:]
print('Respondent : 2017 - {} respondents, 2018 - {} respondents'.format(multipleChoice_2017.shape[0], multipleChoice_2018.shape[0]))
multipleChoice_2017['CompensationAmount']=multipleChoice_2017['CompensationAmount'].str.replace(',','')
multipleChoice_2017['CompensationAmount']=multipleChoice_2017['CompensationAmount'].str.replace('-','')
rates = pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv')
salary=multipleChoice_2017[['CompensationAmount','CompensationCurrency','Country']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
def categorize_salary(x):
    if x < 10000:
        return '0-10,000'
    elif 10000 < x and x <= 20000:
        return '10-20,000'
    elif 20000 < x and x <= 30000:
        return '20-30,000'
    elif 30000 < x and x <= 40000:
        return '30-40,000'
    elif 40000 < x and x <= 50000:
        return '40-50,000'
    elif 50000 < x and x <= 60000:
        return '50-60,000'
    elif 60000 < x and x <= 70000:
        return '60-70,000'
    elif 70000 < x and x <= 80000:
        return '70-80,000'
    elif 80000 < x and x <= 90000:
        return '80-90,000'
    elif 90000 < x and x <= 100000:
        return '90-100,000'
    elif 100000 < x and x <= 125000:
        return '100-125,000'
    elif 125000 < x and x <= 150000:
        return '125-150,000'
    elif 150000 < x and x <= 200000:
        return '150-200,000'
    elif 200000 < x and x <= 250000:
        return '200-250,000'
    elif 250000 < x and x <= 300000:
        return '250-300,000'
    elif 300000 < x and x <= 400000:
        return '300-400,000'
    elif 400000 < x and x <= 5000000:
        return '400-500,000'
    else:
        return '500,000+'
salary['salary_cat'] = salary['Salary'].map(categorize_salary)

salary_2017 = salary['salary_cat'].value_counts().to_frame()
salary_2017['salary_cat'] = 100 * salary_2017['salary_cat'] / salary_2017['salary_cat'].sum()
salary_2018 = multipleChoice_2018['Q9'].value_counts()[1:].to_frame()
salary_2018['Q9'] = 100 * salary_2018['Q9'] / salary_2018['Q9'].sum()
sorting = ['0-10,000', '10-20,000', '20-30,000', '30-40,000', '40-50,000', '50-60,000', '60-70,000', '70-80,000', '80-90,000', '90-100,000',
          '100-125,000', '125-150,000', '150-200,000', '200-250,000', '250-300,000', '300-400,000','400-500,000', '500,000+']

total_salary = salary_2017.merge(salary_2018, left_index=True, right_index=True)

total_salary = total_salary.loc[sorting]
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
total_salary.plot.bar(ax=ax)
ax.legend(['2017 salary', '2018 salary'])
ax.set_ylabel('Normalized frequency (%)')
ax.set_xlabel('Salary category (USD)')
plt.show()
cnt_srs_2018 = multipleChoice_2018.loc[:, 'Q3'].value_counts()[:15]
cnt_srs_2018 = 100 * cnt_srs_2018 / cnt_srs_2018.sum()

cnt_srs_2017 = multipleChoice_2017['Country'].value_counts()[:15]
cnt_srs_2017 = 100 * cnt_srs_2017 / cnt_srs_2017.sum()
fig, ax = plt.subplots(2, 1, figsize=(8, 12))

sns.barplot(cnt_srs_2017.values, cnt_srs_2017.index, ax=ax[0])
ax[0].set_title('2017 Country')
# ax[0].set_xlabel('Frequency(%)')

sns.barplot(cnt_srs_2018.values, cnt_srs_2018.index, ax=ax[1])
ax[1].set_title('2018 Country')
ax[1].set_xlabel('Frequency(%)')

plt.show()
salary_mapping = {'0-10,000': 5000, '10-20,000':15000, '20-30,000':25000, '30-40,000':35000, '40-50,000':45000, '50-60,000':55000, '60-70,000':65000, 
                  '70-80,000':75000, '80-90,000':85000, '90-100,000':95000, '100-125,000':112500, '125-150,000':137500, '150-200,000':175000, 
                  '200-250,000':225000, '250-300,000':275000, '300-400,000':350000,'400-500,000':450000, '500,000+':700000}

temp_salary_2017 = salary.loc[~(salary['Salary'] > 700000)]
salary_country_2017 = temp_salary_2017.groupby('Country')['Salary'].median().sort_values(ascending=False)[:15].to_frame()

temp_salary_2018 = multipleChoice_2018[['Q3', 'Q9']]
temp_salary_2018['Q9'] = temp_salary_2018['Q9'].map(salary_mapping)

salary_country_2018 = temp_salary_2018.groupby('Q3')['Q9'].median().sort_values(ascending=False)[:15].to_frame()
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.barplot('Salary',salary_country_2017.index,data=salary_country_2017,palette='RdYlGn',ax=ax[0])
# ax[0].axvline(salary_country_2017['Salary'].median(),linestyle='dashed')
ax[0].set_title('Compensation of Top 15 Respondent Countries for 2017', fontsize=20)
ax[0].set_xlabel('')
ax[0].set_xlim([0, 120000])


sns.barplot('Q9',salary_country_2018.index, data=salary_country_2018,palette='RdYlGn',ax=ax[1])
# ax[1].axvline(salary_country_2018['Q9'].median(),linestyle='dashed')
ax[1].set_title('Compensation of Top 15 Respondent Countries for 2018', fontsize=20)
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_xlim([0, 120000])
plt.subplots_adjust(wspace=1)
plt.show()
temp_2017 = multipleChoice_2017.loc[multipleChoice_2017['Age'] > 18, :]
def categorize_age(x):
    if 18 <= x and x <= 21:
        return '18-21'
    elif 22 <= x and x <= 24:
        return '22-24'
    elif 25 <= x and x <= 29:
        return '25-29'
    elif 30 <= x and x <= 34:
        return '30-34'
    elif 35 <= x and x <= 39:
        return '35-39'
    elif 40 <= x and x <= 44:
        return '40-44'
    elif 45 <= x and x <= 49:
        return '45-49'
    elif 50 <= x and x <= 54:
        return '50-54'
    elif 55 <= x and x <= 59:
        return '55-59'
    elif 60 <= x and x <= 69:
        return '60-69'
    elif 70 <= x and x <= 79:
        return '70-79'
    elif x >= 80:
        return '80+'
    else:
        return 'None'
temp_2017['age_cat'] = temp_2017['Age'].apply(categorize_age)
age_2017 = temp_2017['age_cat'].value_counts().to_frame()
age_2017['age_cat'] = 100 * age_2017['age_cat'] / age_2017['age_cat'].sum()

age_2018 = multipleChoice_2018['Q2'].value_counts().to_frame()
age_2018['Q2'] = 100 * age_2018['Q2'] / age_2018['Q2'].sum()

sorting = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70-79', '80+']

total_age = age_2017.merge(age_2018, left_index=True, right_index=True)
total_age = total_age.loc[sorting]

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

total_salary.plot.bar(ax=ax[0])
ax[0].legend(['2017', '2018'])
ax[0].set_ylabel('Normalized frequency (%)')
ax[0].set_xlabel('Salary category (USD)')

total_age.plot.bar(ax=ax[1])
ax[1].legend(['2017', '2018'])
ax[1].set_ylabel('Normalized frequency (%)')
ax[1].set_xlabel('Age)')
plt.show()
new_2017 = multipleChoice_2017.loc[(18 <= multipleChoice_2017['Age']) & (multipleChoice_2017['Age'] < 30), :]
young_category = ['18-21', '22-24', '25-29']
new_2018 = multipleChoice_2018.loc[multipleChoice_2018['Q2'].isin(young_category), :]

print('Total number of young kaggler in 2017 was', new_2017.shape[0])
print('Total number of young kaggler in 2017 is', new_2018.shape[0])

print('The percent of young kaggler in 2017 was {:.2f}%'.format(100*(new_2017.shape[0] / multipleChoice_2017.shape[0])))
print('Total number of young kaggler in 2017 is {:.2f}%'.format(100*(new_2018.shape[0] / multipleChoice_2018.shape[0])))
language_2017 = new_2017['LanguageRecommendationSelect'].value_counts().to_frame()
language_2018 = new_2018['Q18'].value_counts().to_frame()

total_language = language_2017.merge(language_2018, how='outer', left_index=True, right_index=True).sort_values('LanguageRecommendationSelect', ascending=False)
total_language.columns = ['2017 language recommendation', '2018 language recommendation']
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
total_language.plot.bar(ax=ax)
plt.show()
source = []
counts = []
for i in range(11):
    col_name = 'Q33_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

dataset_2018 = pd.Series(index=source, data=counts)
data=new_2017['PublicDatasetsSelect'].str.split(',')
dataset=[]
for i in data.dropna():
    dataset.extend(i)
dataset_2017 = pd.Series(dataset).value_counts()
fig, ax = plt.subplots(2, 1, figsize=(10, 20))

dataset_2017.plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,
                      wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' }, ax=ax[0])

ax[0].set_title('2017 Dataset source')
dataset_2018.plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,
                      wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' }, ax=ax[1])
ax[1].set_title('2018 Dataset source')
employerIndustry_2017 = new_2017['EmployerIndustry'].value_counts().sort_values(ascending=True)
employerIndustry_2018 = new_2018['Q7'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots(2, 1, figsize=(10, 14))

employerIndustry_2018.plot.barh(ax=ax[0])
employerIndustry_2017.plot.barh(ax=ax[1])

ax[0].set_title('2017 industry')
ax[1].set_title('2018 industry')
source = []
counts = []
for i in range(6):
    col_name = 'Q11_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print('Select any activities that make up an important part of your role at work?')
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(14):
    col_name = 'Q13_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(10):
    col_name = 'Q14_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following hosted notebooks have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(6):
    col_name = 'Q15_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following cloud computing services have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(17):
    col_name = 'Q16_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("What programming languages do you use on a regular basis?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(18):
    col_name = 'Q19_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("What machine learning frameworks have you used in the past 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(12):
    col_name = 'Q21_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("What data visualization libraries or tools have you used in the past 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
print("Approximately what percent of your time at work or school is spent actively coding?")
plt.figure(figsize=(8, 8))
new_2018['Q23'].value_counts().sort_values(ascending=True).plot.barh()
plt.show()
print("How long have you been writing code to analyze data?")
plt.figure(figsize=(8, 8))
new_2018['Q24'].value_counts().sort_values(ascending=True).plot.barh()
plt.show()
print("Do you consider yourself to be a data scientist?")
fig, ax = plt.subplots(1, 2, figsize=(14, 8))

new_2017['DataScienceIdentitySelect'].value_counts().sort_values(ascending=True).plot.bar(ax=ax[0])
ax[0].set_title('2017 DataScienceIdentity')

new_2018['Q26'].value_counts().sort_values(ascending=True).plot.bar(ax=ax[1])
ax[1].set_title('2018 DataScienceIdentity')
plt.show()
source = []
counts = []
for i in range(19):
    col_name = 'Q27_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following cloud computing products have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(41):
    col_name = 'Q28_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following machine learning products have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True)[:20].plot.barh()
plt.show()
source = []
counts = []
for i in range(27):
    col_name = 'Q29_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following relational database products have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True)[:20].plot.barh()
plt.show()
source = []
counts = []
for i in range(24):
    col_name = 'Q30_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which of the following big data and analytics products have you used at work or school in the last 5 years?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True)[:20].plot.barh()
plt.show()
source = []
counts = []
for i in range(11):
    col_name = 'Q31_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print("Which types of data do you currently interact with most often at work or school?")
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True)[:20].plot.barh()
plt.show()
plt.figure(figsize=(8, 8))
print('How do young kaggler think about Fairness and bias in ML algorithms?')
new_2018['Q41_Part_1'].value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,
                      wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.ylabel(' ')
plt.show()
plt.figure(figsize=(8, 8))
print('How do young kaggler think about Being able to explain ML model outputs and/or predictions?')
new_2018['Q41_Part_2'].value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,
                      wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.ylabel(' ')
plt.show()
plt.figure(figsize=(8, 8))
print('How do young kaggler think about Reproducibility in data science?')
new_2018['Q41_Part_2'].value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,
                      wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.ylabel(' ')
plt.show()
source = []
counts = []
for i in range(16):
    col_name = 'Q47_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

interpretation_2018 = pd.Series(index=source, data=counts)

plt.figure(figsize=(8, 8))
print('What methods do you prefer for explaining and/or interpreting decisions that are made by ML models?')

interpretation_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(12):
    col_name = 'Q49_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print('What tools and methods do you use to make your work easy to reproduce?')
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(8):
    col_name = 'Q50_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print('What barriers prevent you from making your work even easier to reuse and reproduce?')
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()
source = []
counts = []
for i in range(18):
    col_name = 'Q38_Part_{}'.format(i+1)
    source.append(new_2018[col_name].value_counts().index.values[0])
    counts.append(new_2018[col_name].value_counts().values[0])

reproduce_2018 = pd.Series(index=source, data=counts)

print('Who/what are your favorite media sources that report on data science topics?')
plt.figure(figsize=(8, 8))
reproduce_2018.sort_values(ascending=True).plot.barh()
plt.show()