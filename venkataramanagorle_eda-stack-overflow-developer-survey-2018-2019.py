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

#https://medium.com/@venkataramanagorle/de-cluttering-the-software-developer-career-67ea76d5eff5?sk=d6a26effc298a55102d67d506141fd07
import pandas as pd

survey_results_public_2018 = pd.read_csv("../input/stackoverflowdevelopersurvey/survey_results_public_2018.csv")

survey_results_schema_2018 = pd.read_csv("../input/stackoverflowdevelopersurvey/survey_results_schema_2018.csv")

survey_results_public_2019 = pd.read_csv("../input/stackoverflowdevelopersurvey/survey_results_public_2019.csv")

survey_results_schema_2019 = pd.read_csv("../input/stackoverflowdevelopersurvey/survey_results_schema_2019.csv")
schema_features_2018 = survey_results_schema_2018['Column']

schema_features_2019 = survey_results_schema_2019['Column']
str(list(schema_features_2018))
str(list(schema_features_2019))
list(survey_results_schema_2018[survey_results_schema_2018.columns[1]])
list(survey_results_schema_2019[survey_results_schema_2019.columns[1]])
choosen_features_final = ['Country', 'EducationLevel', 'Major', 'OrganisationSize', 'DeveloperType', 'Experience', 'DatabaseExp',

                          'PlatformExp', 'LanguageExp', 'Age', 'Gender', 'Race', 'CareerSatisfaction', 

                          'JobSatisfaction', 'Salary']



choosen_features_2018 = ['Country', 'FormalEducation', 'UndergradMajor', 'CompanySize', 'DevType', 'YearsCoding', 'DatabaseWorkedWith', 

                         'PlatformWorkedWith', 'LanguageWorkedWith', 'Age', 'Gender',

                         'RaceEthnicity', 'CareerSatisfaction', 'JobSatisfaction', 'ConvertedSalary']



choosen_features_2019 = ['Country', 'EdLevel', 'UndergradMajor', 'OrgSize', 'DevType', 'YearsCode', 'DatabaseWorkedWith', 

                         'PlatformWorkedWith', 'LanguageWorkedWith', 'Age', 'Gender', 'Ethnicity', 'CareerSat', 

                         'JobSat', 'ConvertedComp']

combined_survey_df_main = pd.DataFrame()

for i in range(0, len(choosen_features_final)):

    combined_survey_df_main[choosen_features_final[i]] = pd.concat([survey_results_public_2018[choosen_features_2018[i]], survey_results_public_2019[choosen_features_2019[i]]], ignore_index=True)

combined_survey_df = combined_survey_df_main.copy()
combined_survey_df.head()
combined_survey_df.shape
combined_survey_df.Country.isna().sum()
combined_survey_df.Country.fillna('Others', inplace=True)
combined_survey_df.Country.value_counts().head(10)
countries_with_less_than_500_developers = list(combined_survey_df.Country.value_counts()[combined_survey_df.Country.value_counts().values < 500].index)

countries_with_less_than_500_developers
combined_survey_df.Country = combined_survey_df.Country.apply(lambda country: 'Others' if country in countries_with_less_than_500_developers else country)
combined_survey_df.Country.value_counts().head(10)
combined_survey_df.EducationLevel.isna().sum()
combined_survey_df.EducationLevel.value_counts()
combined_survey_df.EducationLevel.isna().sum()
combined_survey_df.EducationLevel.fillna('Unknown', inplace=True)
edu_level_dict = {'Bachelor’s degree (BA, BS, B.Eng., etc.)':'Bachelors',

                  'Master’s degree (MA, MS, M.Eng., MBA, etc.)':'Masters',

                  'Some college/university study without earning a degree':'Col. w/o Degree',

                  'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':'High School',

                  'Primary/elementary school':'Elem. School',

                  'Associate degree':'Degree',

                  'Other doctoral degree (Ph.D, Ed.D., etc.)':'Doctoral',

                  'Professional degree (JD, MD, etc.)':'Doctoral',

                  'I never completed any formal education':'Not Completed',

                  'Unknown':'Unknown'}

combined_survey_df.EducationLevel = combined_survey_df.EducationLevel.apply(lambda edu: edu_level_dict[edu])
combined_survey_df.EducationLevel.value_counts().head(10)
combined_survey_df.Major.isna().sum()
combined_survey_df.Major.fillna('Unknown', inplace=True)
combined_survey_df.Major.value_counts()
major_short_dict = {

    'Computer science, computer engineering, or software engineering':'Comp. Sci',

    'Unknown':'Unknown',

    'Another engineering discipline (ex. civil, electrical, mechanical)':'Engg. Others',

    'Information systems, information technology, or system administration':'Info. Tech',

    'A natural science (ex. biology, chemistry, physics)':'Basic. Sci',

    'Web development or web design':'Web. dsgn',

    'Mathematics or statistics':'Math Stats',

    'A business discipline (ex. accounting, finance, marketing)':'Buss. Adm',

    'A social science (ex. anthropology, psychology, political science)':'Social. Sci',

    'A humanities discipline (ex. literature, history, philosophy)':'Humanities',

    'Fine arts or performing arts (ex. graphic design, music, studio art)':'Arts',

    'I never declared a major':'Unknown',

    'A health science (ex. nursing, pharmacy, radiology)':'Health. Sci'

}

combined_survey_df.Major = combined_survey_df.Major.apply(lambda major: major_short_dict[major])
combined_survey_df.Major.value_counts()
combined_survey_df.OrganisationSize.isna().sum()
combined_survey_df.OrganisationSize.fillna('Unknown', inplace=True)
combined_survey_df.OrganisationSize.value_counts()
org_dict = {

    'Unknown':'Unknown',

    '20 to 99 employees':'20-99',

    '100 to 499 employees':'100-499',

    '10,000 or more employees':'>= 10000',

    '1,000 to 4,999 employees':'1000-4999',

    '10 to 19 employees':'10-19',

    '500 to 999 employees':'500-999',

    'Fewer than 10 employees':'< 10',

    '2-9 employees':'2-9',

    '5,000 to 9,999 employees':'5000-9999',

    'Just me - I am a freelancer, sole proprietor, etc.':'Freelancer'

}

combined_survey_df.OrganisationSize = combined_survey_df.OrganisationSize.apply(lambda org: org_dict[org])
combined_survey_df.OrganisationSize.value_counts()
combined_survey_df.DeveloperType.isna().sum()
combined_survey_df.DeveloperType.fillna('Unknown', inplace=True)

combined_survey_df.DeveloperType.value_counts()
combined_survey_df.DeveloperType.value_counts().shape
def extract_value_counts_from_feature(series, separator):

    value_counts = pd.Series()

    for index in series.values:

        values = str(index).split(separator)

        for value in values:

            value_counts[value] = value_counts[value] + 1 if value in value_counts.index else 1

    return value_counts.sort_values(ascending=False)
developer_value_counts = extract_value_counts_from_feature(combined_survey_df.DeveloperType, ';')
developer_value_counts.head(10)
combined_survey_df.CareerSatisfaction.isna().sum()
combined_survey_df.CareerSatisfaction.fillna('Unknown', inplace=True)
# Lets convert the grading to short notation

sat_dict = {

    'Slightly satisfied':'SS',

    'Very satisfied':'VS',

    'Moderately satisfied':'MS',

    'Slightly dissatisfied':'SD',

    'Neither satisfied nor dissatisfied':'NN',

    'Extremely satisfied':'ES',

    'Moderately dissatisfied':'MD',

    'Very dissatisfied':'VD',

    'Extremely dissatisfied':'ED',

    'Unknown':'Unknown'

}
combined_survey_df.CareerSatisfaction = combined_survey_df.CareerSatisfaction.apply(lambda sat: sat_dict[sat])
combined_survey_df.CareerSatisfaction.value_counts()
combined_survey_df.JobSatisfaction.isna().sum()
combined_survey_df.JobSatisfaction.fillna('Unknown', inplace=True)
combined_survey_df.JobSatisfaction = combined_survey_df.JobSatisfaction.apply(lambda sat: sat_dict[sat])
combined_survey_df.JobSatisfaction.value_counts()
combined_survey_df.Experience.isna().sum()
combined_survey_df.Experience.fillna(0)
def collect_non_int_str(array):

    non_int_arr = []

    for element in array:

        try:

            int(element)

        except ValueError:

            non_int_arr.append(element)

    return non_int_arr
collect_non_int_str(combined_survey_df.Experience.value_counts().index)
exp_dict = {

    '3-5 years':4,

    '6-8 years':7,

    '9-11 years':10,

    '0-2 years':1,

    '12-14 years':13,

    '15-17 years':16,

    '18-20 years':19,

    '30 or more years':35,

    '21-23 years':22,

    '24-26 years':25,

    'Less than 1 year':1,

    '27-29 years':28,

    'More than 50 years':55,

    'Unknown':0.0

}

combined_survey_df.Experience = combined_survey_df.Experience.apply(lambda exp: exp_dict[exp] if exp in exp_dict.keys() else float(exp))

exp_mean = combined_survey_df.Experience.mean()

combined_survey_df.Experience.replace(0, exp_mean, inplace=True)
combined_survey_df.Experience.value_counts().head(10)
combined_survey_df.Age.isna().sum()
combined_survey_df.Age.fillna(0, inplace=True)
collect_non_int_str(combined_survey_df.Age.value_counts().index)
age_dict = {

    '25 - 34 years old':30,

    '18 - 24 years old':21,

    '35 - 44 years old':40,

    '45 - 54 years old':50,

    'Under 18 years old':17,

    '55 - 64 years old':60,

    '65 years or older':65

}

combined_survey_df.Age = combined_survey_df.Age.apply(lambda age: age_dict[age] if age in age_dict.keys() else float(age))

age_mean = combined_survey_df.Age.mean()

combined_survey_df.Age = combined_survey_df.Age.apply(lambda age: age_mean if age == 0 else age )
combined_survey_df.Age.value_counts().head(10)
combined_survey_df.Salary.isna().sum()
combined_survey_df.Salary.fillna(0, inplace=True)

combined_survey_df = combined_survey_df.loc[combined_survey_df.Salary != 0]
collect_non_int_str(combined_survey_df.Salary.value_counts().index)
combined_survey_df.head()
import seaborn as sns

sns.set_style("darkgrid")

sns.set_context("paper")

import matplotlib.pyplot as plt



def plot_bar_top_n_percentage(dataframe, features, top_n):

    no_features = len(features)

    cols = 3

    rows = no_features // 3 if no_features % 3 == 0  else no_features // 3 + 1

    

    fig, ax = plt.subplots(ncols = cols, nrows=rows, figsize=(10*cols, 8*rows), squeeze=False)

    

    count = 0

    for i in range(rows):

        for j in range(cols):

            if(count >= no_features) : break

            data = dataframe[features[count]].value_counts(normalize=True).sort_values(ascending=False).head(top_n)*100

            sns.barplot(x=data.index.values, y=data.values, ax=ax[i][j])

            labels = [label if len(label) < 15 else label[:15] for label in data.index.values]

            ax[i][j].tick_params(axis='both', which='major', labelsize=20)

            ax[i][j].set_xticklabels(labels, rotation=45)

            ax[i][j].set_ylabel("Percentage", fontsize=20)

            ax[i][j].set_title(features[count], fontsize=25)

            count = count + 1

    plt.tight_layout()

    plt.show()

    
plot_bar_top_n_percentage(combined_survey_df, ['Country', 'EducationLevel', 'Major','OrganisationSize', 'Gender', 'Race'], 7)
fig = plt.figure(figsize=(12,6))

sns.distplot(combined_survey_df.Salary, kde=False)

plt.show()
combined_survey_df = combined_survey_df.loc[combined_survey_df.Salary <= 300000].reset_index(drop=True)
fig = plt.figure(figsize=(12,6))

sns.distplot(combined_survey_df.Salary, kde=False)

plt.title("Overall Compensation Distribution", fontsize=15)

plt.show()
mean = combined_survey_df.Salary.mean()

std  = combined_survey_df.Salary.std()



sd_one_minus = mean - std

sd_one_plus  = mean + std



print("mean  : ", mean)

print("std   : ", std)

print("1 SD -: ", sd_one_minus)

print("1 SD +: ", sd_one_plus)
high_earn_dev_group = combined_survey_df.loc[combined_survey_df.Salary > sd_one_plus]

plot_bar_top_n_percentage(high_earn_dev_group, ['Country', 'EducationLevel', 'Major','OrganisationSize', 'Gender', 'Race'], 7)
low_earn_dev_group = combined_survey_df.loc[combined_survey_df.Salary < sd_one_minus]

plot_bar_top_n_percentage(low_earn_dev_group, ['Country', 'EducationLevel', 'Major','OrganisationSize', 'Gender', 'Race'], 7)
developing_cntry_dev_group = combined_survey_df[(combined_survey_df.Country == 'India') | (combined_survey_df.Country == 'Others')]

developed_cntry_dev_group  = combined_survey_df[(combined_survey_df.Country == 'United States')]

fig = plt.figure(figsize=(14,8))

sns.distplot(developing_cntry_dev_group.Salary, label="Developing Countries", color='red', kde=False)

sns.distplot(developed_cntry_dev_group.Salary, label="Developed Countries", color='blue', kde=False)

plt.legend(fontsize=15)

plt.show()
print("Salary of Developers from Developing Countries: Mean ", developing_cntry_dev_group.Salary.mean(), ": Median ", developing_cntry_dev_group.Salary.median())

print("Salary of Developers from Deveploed Countries : Mean ", developed_cntry_dev_group.Salary.mean(),  ": Median ", developed_cntry_dev_group.Salary.median())
def plot_value_counts(value_counts, title):

    fig = plt.figure(figsize=(14,8))

    sns.barplot(y=value_counts.index.values, x=value_counts.values)

    ax = plt.gca()

    ax.yaxis.set_ticks_position('right')

    labels = [label if len(label) < 15 else label[0:14] for label in value_counts.index.values]

    ax.set_yticklabels(labels)

    plt.setp(ax.get_yticklabels(), fontsize=15)

    plt.title(title, fontsize=20)

    plt.show()
prog_lang_high_earners = extract_value_counts_from_feature(high_earn_dev_group.LanguageExp,";")

plot_value_counts(prog_lang_high_earners.head(20), "High Earning")
prog_lang_low_earners = extract_value_counts_from_feature(low_earn_dev_group.LanguageExp,";")

plot_value_counts(prog_lang_low_earners.head(20), "Low Earning")
dev_type_high_earners = extract_value_counts_from_feature(high_earn_dev_group.DeveloperType,";")

plot_value_counts(dev_type_high_earners.head(20), "High Earning")
dev_type_low_earners  = extract_value_counts_from_feature(low_earn_dev_group.DeveloperType,";")

plot_value_counts(dev_type_low_earners.head(20), "Low Earning")
plat_exp_high_earners = extract_value_counts_from_feature(high_earn_dev_group.PlatformExp,";")

plot_value_counts(plat_exp_high_earners.head(20), "High Earning")
plat_exp_low_earners = extract_value_counts_from_feature(low_earn_dev_group.PlatformExp,";")

plot_value_counts(plat_exp_low_earners.head(20), "Low Earning")
plot_bar_top_n_percentage(combined_survey_df, ['CareerSatisfaction', 'JobSatisfaction'], 10)
high_car_sat_dev_group = combined_survey_df.loc[(combined_survey_df.CareerSatisfaction == 'ES') | (combined_survey_df.CareerSatisfaction == 'VS')].reset_index(drop=True)

low_car_sat_dev_group  = combined_survey_df.loc[(combined_survey_df.CareerSatisfaction == 'ED') | (combined_survey_df.CareerSatisfaction == 'VD')].reset_index(drop=True)
plot_bar_top_n_percentage(high_car_sat_dev_group, ['Country', 'OrganisationSize'], 7)
plot_bar_top_n_percentage(low_car_sat_dev_group, ['Country', 'OrganisationSize'], 7)
high_job_sat_dev_group = combined_survey_df.loc[(combined_survey_df.JobSatisfaction == 'ES') | (combined_survey_df.JobSatisfaction == 'VS')].reset_index(drop=True)

low_job_sat_dev_group  = combined_survey_df.loc[(combined_survey_df.JobSatisfaction == 'ED') | (combined_survey_df.JobSatisfaction == 'VD')].reset_index(drop=True)
plot_bar_top_n_percentage(high_job_sat_dev_group, ['Country', 'OrganisationSize'], 7)
plot_bar_top_n_percentage(low_job_sat_dev_group, ['Country', 'OrganisationSize'], 7)
fig = plt.figure(figsize=(14,8))

sns.distplot(high_car_sat_dev_group.Salary, label='High Satisfaction', color='red', norm_hist=False)

sns.distplot(low_car_sat_dev_group.Salary, label='Low Satisfaction', color='blue', norm_hist=False)

plt.legend(fontsize=15)

plt.show()
print("Salary of Satisfied Developers   : Mean ", high_car_sat_dev_group.Salary.mean(), ": Median ", high_car_sat_dev_group.Salary.median())

print("Salary of Dissatisfied Developers: Mean ", low_car_sat_dev_group.Salary.mean(),  ": Median ", low_car_sat_dev_group.Salary.median())