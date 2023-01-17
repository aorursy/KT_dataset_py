import numpy as np

import pandas as pd

import re

import io

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA #, SparsePCA, LatentDirichletAllocation

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
text = open(r'/kaggle/input/simplyhired-job-data/home/sdf/marketing_sample_for_simplyhired_com-jobs__20190901_20190915__30k_data.xml','r')
string = text.read()
text.close()
string[:2000]
entries = string.split('record>')[1::2]

columns = ['employer', 'job-city', 'ad-title', 'salary-range', 'job-country', 'job-state', 'description'  ]

df_columns = ['employer', 'job_city', 'ad_title', 'salary_range', 'job_country', 'job_state', 'description'  ]

re_columns = ['<'+col+'>(.*)<\/'+col + '>' for col in columns]
rows = []

for entry in entries:

    row = []

    for col in re_columns:

        found = re.findall(col, entry)

        if len(found):

            row.append(re.findall(col, entry)[0])

        else:

            row.append(np.nan)

    rows.append(row)
data = pd.DataFrame(rows, columns=df_columns)

data = data.sample(frac=1).reset_index(drop=True)

data.head()
data.shape
test = data[-5000:].reset_index(drop=True)

data = data[:-5000]
def states_name(string):

    states_dict ={ 

              'California':'CA', 'Delaware':'DE', 'Utah': 'UT', 'Indiana':'IN', 'United States': np.nan,

              'Florida':'FL', 'Maryland': 'MD', 'Ohio': 'OH', 'Hawaii':'HI', 'Massachusetts': 'MA',

              'Washington State':'WA', 'Virginia':'VA', 'Tennessee':'TN', 'New Jersey':'NJ', 'Pennsylvania':'PA',

               'Texas':'TX', 'Nevada':'NV', 'Georgia':'GA', 'Rhode Island':'RI', 'Vermont':'VT',

               'Quad Cities':'MN', 'Oklahoma':'OK', 'New York State': 'NY', 'West Virginia':'WV',

               'Mississippi':'MS', 'Colorado':'CO', 'New Hampshire':'NH', 'Alaska':'AK', 'Illinois':'IL',

               'South Dakota':'SD', 'Kansas':'KS', 'Arizona':'AZ', 'Maine':'ME', 'Connecticut':'CT',

               'Iowa':'IA', 'Idaho':'ID', 'Wisconsin':'WI', 'Missouri':'MI', 'Minnesota':'MN' , 'Alabama':'AL',

               'Nebraska':'NE'}

    if string in states_dict.keys():

        return states_dict[string]

    else:

        return string
for index in data[data.job_state.isna()].index:

    data.loc[index, 'job_state'] = data.loc[index, 'job_city']

    data.loc[index, 'job_city'] = np.nan

    

for index in test[test.job_state.isna()].index:

    test.loc[index, 'job_state'] = test.loc[index, 'job_city']

    test.loc[index, 'job_city'] = np.nan
data.job_state = data.job_state.apply(lambda x: states_name(x))

data.job_state.fillna('not known', inplace=True)

test.job_state = test.job_state.apply(lambda x: states_name(x))

test.job_state.fillna('not known', inplace=True)
def type_salary(string):

    periods = ['hour', 'day', 'week', 'month', 'year']

    if isinstance(string, str)==False:

        return np.nan

    for period in periods:

        if re.search(period, string):

            return period

    return np.nan
def convert_int(string):

    return int(''.join(string.split(',')))
def extract_salary(string, lower):

    periods = ['hour', 'day', 'week', 'month', 'year']

    if isinstance(string, str)==False:

        return np.nan

    period = type_salary(string)

    if isinstance(period, str)==False:

        return np.nan

#    if period not in periods:

#        period = 'year'

    period_normal = {'year':1, 'month':12, 'week':52, 'day':260, 'hour':2080}

    sal_range = re.findall('[\d,]+', string) 

    if lower:

        if len(sal_range):

            return convert_int(sal_range[0])*period_normal[period]

        else:

            return np.nan

    else:

        if len(sal_range)==2:

            return convert_int(sal_range[1])*period_normal[period]

        else:

            return convert_int(sal_range[0])*period_normal[period]
data.columns
data['salary_low'] = data.salary_range.apply(lambda x: extract_salary(x,True))

data['salary_high'] = data.salary_range.apply(lambda x: extract_salary(x,False))

data['salary_type'] = data.salary_range.apply(lambda x: type_salary(x))



test['salary_low'] = test.salary_range.apply(lambda x: extract_salary(x,True))

test['salary_high'] = test.salary_range.apply(lambda x: extract_salary(x,False))

test['salary_type'] = test.salary_range.apply(lambda x: type_salary(x))



data.eval('salary = (salary_low + salary_high)/2', inplace=True)

test.eval('salary = (salary_low + salary_high)/2', inplace=True)
cat_columns = ['employer', 'job_city', 'ad_title', 'job_country', 'job_state', 'salary_type']

num_columns = ['salary_low', 'salary_high', 'salary']
data[cat_columns].describe()
pd.options.display.float_format = '{:,.2f}'.format
data[num_columns].describe(percentiles=[0.01, 0.25, 0.5, 0.9,0.99]).applymap("${:,.0f}".format)
data.drop(columns=[ 'salary_range', 'job_country'], inplace=True)

test.drop(columns=[ 'salary_range', 'job_country'], inplace=True)
data['descr_len'] = data.description.apply(lambda x: len(x))

test['descr_len'] = data.description.apply(lambda x: len(x))

plt.scatter(data[~data.salary.isna()].salary.values,data[~data.salary.isna()].descr_len.values)

plt.title('Length of description vs Salary')

plt.show()
plt.figure(figsize=(13,6))

sns.distplot(data.salary, hist=False, label='average')

sns.distplot(data.salary_low, hist=False, label='low')

sns.distplot(data.salary_high, hist=False, label='high')

plt.xlim(0,150000)

plt.yticks([])

plt.title('Salary distribution')

plt.show()
salary_range = data.salary_high.values - data.salary_low.values 

log_salary_range = np.log(data.salary_high.values) - np.log(data.salary_low.values)

fig, axes = plt.subplots(1,2, figsize=(20,7))

axes[0].scatter(salary_range, data.salary.values)

axes[0].set_title('Salary range depending on salary')

axes[0].set_ylabel('Width of salary range')

axes[0].set_xlabel('Average salary')



axes[1].scatter(log_salary_range, np.log(data.salary.values))

axes[1].set_title('Same graphs after log is taken')

axes[1].set_yticks([])

axes[1].set_xlabel('log_salary')

plt.show()
plt.figure(figsize=(18,7))

sns.boxplot(y='salary', x='job_state', data=data)

plt.ylim([10000,70000])

plt.title('Salary distribution by state')

plt.show()
plt.figure(figsize=(18,7))

sns.countplot(data.job_state)

plt.title('Job count by state')

plt.show()
plt.figure(figsize=(20,7))

for state in ['IN', 'TX', 'IL', 'MA', 'NY', 'OH', 'not known', 'Remote']:

    sns.distplot(data[data.job_state==state].salary.dropna(), hist=False, label=state)

plt.legend()

plt.xlim(20000,70000)

plt.title('Salary distribution by state')

plt.yticks([])

plt.show()
mean_salary_state = data.groupby('job_state').salary.mean().to_dict()
data['state_salary'] = data.job_state.map(mean_salary_state)

test['state_salary'] = data.job_state.map(mean_salary_state)
plt.figure(figsize=(10,7))

sns.boxplot(x='salary_type', y='salary', data=data)

plt.ylim([10000,100000])

plt.title('Salary distribution by type')

plt.show()
cities = data.job_city.unique()

states = data.job_state.unique()

most_common_city = []

for state in states:

    most_common_city.append((data[data.job_state == state].job_city.describe().top, state))
def check_common_city(city, state, common_list):

    return (city,state) in common_list
data['common_city'] = data.apply(lambda x: check_common_city(x.job_city, x.job_state, most_common_city), axis=1)

test['common_city'] = test.apply(lambda x: check_common_city(x.job_city, x.job_state, most_common_city), axis=1)
av_salary_common_city = pd.DataFrame(data[data.common_city==True].groupby(['job_state']).salary.mean())

av_salary_common_city['common_city'] = True

av_salary_other_city = pd.DataFrame(data[data.common_city==False].groupby(['job_state']).salary.mean())

av_salary_other_city['common_city'] = False

df = pd.concat([av_salary_common_city, av_salary_other_city ]).reset_index()



common_states = list(data.groupby('job_state').salary.count().sort_values()[-15:].index)



plt.figure(figsize=(15, 6))

sns.barplot(x='job_state', y='salary', hue='common_city',

                data=df[df.job_state.isin(common_states)])

plt.title('Mean salary for different states in most common city, and other city')

plt.legend()

plt.show()
vectorizer1 = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(2, 2), min_df=30)

vectorizer11 = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), min_df=100)



ad_title_tokens = vectorizer1.fit_transform(data.ad_title.values)

ad_title_tokens1 = vectorizer11.fit_transform(data.ad_title.values)



vectorizer2 = CountVectorizer(lowercase=True, stop_words='english', 

                                  ngram_range=(2, 2), min_df=300, max_df = 0.7)



vectorizer22 = CountVectorizer(lowercase=True, stop_words='english', 

                                  ngram_range=(1, 1), min_df=500, max_df = 0.5)



description_tokens = vectorizer2.fit_transform(data.description.values)

description_tokens1 = vectorizer22.fit_transform(data.description.values)
test_ad_title_tokens = vectorizer1.transform(test.ad_title.values)

test_ad_title_tokens1 = vectorizer11.transform(test.ad_title.values)

test_description_tokens = vectorizer2.transform(test.description.values)

test_description_tokens1 = vectorizer22.transform(test.description.values)
print('Number of words in vectorizers: {}, {}, {}, {}'.format(

    len(vectorizer1.get_feature_names()),len(vectorizer11.get_feature_names()), 

    len(vectorizer2.get_feature_names()),len(vectorizer22.get_feature_names()) ))
pca_ad = PCA(n_components=30)

pca_ad1 = PCA(n_components=25)

ad_title_matrix = pca_ad.fit_transform(ad_title_tokens.toarray())

ad_title_matrix1 = pca_ad1.fit_transform(ad_title_tokens1.toarray())



pca_desc = PCA(n_components=100)

pca_desc2 = PCA(n_components=125)

desc_matrix = pca_desc.fit_transform(description_tokens.toarray())

desc_matrix2 = pca_desc2.fit_transform(description_tokens.toarray())
test_ad_title_matrix = pca_ad.transform(test_ad_title_tokens.toarray())

test_ad_title_matrix1 = pca_ad1.transform(test_ad_title_tokens1.toarray())



test_desc_matrix = pca_desc.transform(test_description_tokens.toarray())

test_desc_matrix2 = pca_desc2.transform(test_description_tokens.toarray())
pca_data = pd.DataFrame(np.concatenate((ad_title_matrix, ad_title_matrix1 , desc_matrix, desc_matrix2), axis=1))



test_pca_data = pd.DataFrame(np.concatenate(

    (test_ad_title_matrix, test_ad_title_matrix1 , test_desc_matrix, test_desc_matrix2), axis=1))
data1 = pd.concat([data[['state_salary', 'salary_type','salary', 'common_city','descr_len', ]], 

                   pca_data], axis=1)



test1 = pd.concat([test[['state_salary', 'salary_type','salary', 'common_city','descr_len', ]], 

                   test_pca_data], axis=1)

data1['log_salary'] = data1.salary.apply(lambda x: np.log(x))

test1['log_salary'] = test1.salary.apply(lambda x: np.log(x))
# At this stage, the only na value are in salary column which we plan to predict, so it will make sense to get rid of them.

data1.dropna(inplace=True)

test1.dropna(inplace=True)
enc = OneHotEncoder(drop='first')



onehot_data = enc.fit_transform(data1[['salary_type','common_city']]).toarray()

data2 = pd.merge(data1.drop(columns=['salary_type','common_city']), pd.DataFrame(onehot_data), 

                 left_index=True, right_index=True)



test_onehot_data = enc.transform(test1[['salary_type','common_city']]).toarray()

test2 = pd.merge( test1.drop(columns=['salary_type','common_city']), pd.DataFrame(test_onehot_data), 

                left_index=True, right_index=True)
X_train, X_test, y_train, y_test = train_test_split(

    data2.drop(columns=['salary', 'log_salary']), data2.log_salary, test_size=0.2, random_state=42)
reg = GradientBoostingRegressor(random_state=0)

reg.fit(X_train, y_train)

print('Train score: {}, \nTest score: {}'.format(reg.score(X_train, y_train), reg.score(X_test,y_test)))
mult_error = np.exp(y_test) / np.exp(np.array(reg.predict(X_test)))

dev_from_mean = np.exp(y_test) / data[data.salary<80000].salary.mean()

plt.figure(figsize=(16,6))

sns.distplot(mult_error, label='our model', hist=False)

sns.distplot(dev_from_mean, label='naive model', hist=False)

plt.xlim(0.1,3)

plt.title('Distribution of ratios of real salary/predicted')

plt.yticks([])

plt.legend()

plt.show()
data2['top'] = data2.salary.apply(lambda x: int(x>84000))

data2['bottom'] = data2.salary.apply(lambda x: int(x<22501))
print('We consider top {}% and bottom {}% salaries'.format(round(100*data2.top.mean()), 

                                                           round(100*data2.bottom.mean())))
X_top_train, X_top_test, y_top_train, y_top_test = train_test_split(

    data2.drop(columns=['salary', 'log_salary', 'bottom', 'top']), data2.top, test_size=0.33, random_state=42)

X_bottom_train, X_bottom_test, y_bottom_train, y_bottom_test = train_test_split(

    data2.drop(columns=['salary', 'log_salary', 'bottom', 'top']), data2.bottom, test_size=0.33, random_state=42)



top_sample_weights = np.zeros(len(y_top_train))

top_sample_weights[y_top_train == 0] = 0.05

top_sample_weights[y_top_train == 1] = 0.95



bottom_sample_weights = np.zeros(len(y_bottom_train))

bottom_sample_weights[y_bottom_train == 0] = 0.09

bottom_sample_weights[y_bottom_train == 1] = 0.91
clf_top = GradientBoostingClassifier()

clf_bottom = GradientBoostingClassifier()

model1 = clf_top.fit(X_top_train, y_top_train, sample_weight = top_sample_weights)

model2 = clf_bottom.fit(X_bottom_train, y_bottom_train, sample_weight = bottom_sample_weights)
confusion_matrix(y_top_test, clf_top.predict(X_top_test))
print(classification_report(y_top_test, clf_top.predict(X_top_test)))
confusion_matrix(y_bottom_test, clf_bottom.predict(X_bottom_test))
print(classification_report(y_bottom_test, clf_bottom.predict(X_top_test)))
# Regression

X_train, y_train = data2.drop(columns=['salary', 'log_salary', 'top', 'bottom']), data2.log_salary

X_test, y_test = test2.drop(columns=['salary', 'log_salary']), test2.log_salary,

reg = GradientBoostingRegressor(random_state=0)

reg.fit(X_train, y_train)

print('Train score: {}, \nTest score: {}'.format(reg.score(X_train, y_train), reg.score(X_test,y_test)))
mult_error = np.exp(y_test) / np.exp(np.array(reg.predict(X_test)))

dev_from_mean = np.exp(y_test) / data[data.salary<80000].salary.mean()

plt.figure(figsize=(16,6))

sns.distplot(mult_error, label='our model', hist=False)

sns.distplot(dev_from_mean, label='naive model', hist=False)

plt.xlim(0.1,3)

plt.title('Distribution of ratios of real salary/predicted')

plt.yticks([])

plt.legend()

plt.show()
X_top_train, y_top_train = data2.drop(columns=['salary', 'log_salary', 'bottom', 'top']), data2.top

X_bottom_train, y_bottom_train = data2.drop(columns=['salary', 'log_salary', 'bottom', 'top']), data2.top



X_top_test = test2.drop(columns=['salary', 'log_salary'])

y_top_test = test2.salary>84000

y_top_test = y_top_test.astype(int)



X_bottom_test = X_top_test

y_bottom_test = test2.salary>22501

y_bottom_test = y_top_test.astype(int)



top_sample_weights = np.zeros(len(y_top_train))

top_sample_weights[y_top_train == 0] = 0.05

top_sample_weights[y_top_train == 1] = 0.95



bottom_sample_weights = np.zeros(len(y_bottom_train))

bottom_sample_weights[y_bottom_train == 0] = 0.09

bottom_sample_weights[y_bottom_train == 1] = 0.91
clf_top = GradientBoostingClassifier()

clf_bottom = GradientBoostingClassifier()

model1= clf_top.fit(X_top_train, y_top_train, sample_weight = top_sample_weights)

model2 = clf_bottom.fit(X_bottom_train, y_bottom_train, sample_weight = bottom_sample_weights)
print(classification_report(y_top_test, clf_top.predict(X_top_test)))
print(classification_report(y_bottom_test, clf_bottom.predict(X_top_test)))