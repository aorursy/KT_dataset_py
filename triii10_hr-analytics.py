import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib



matplotlib.rcParams['figure.dpi'] = 200

matplotlib.rcParams['figure.figsize'] = (15, 5)



sns.set_style("darkgrid")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_original = pd.read_csv('/kaggle/input/train_LZdllcl.csv')

test_original = pd.read_csv('/kaggle/input/test_2umaH9m.csv')
train_original.head()
train_original.shape
%%time

train_original.isna().sum()
%%time

null_vals = train_original['education'].value_counts(dropna = False)

null_vals[null_vals.index.isnull()]
%%time

train_original['education'].isna().sum()
train_original.info()
X_train, X_test, Y_train, Y_test = train_test_split(

    train_original.drop(['is_promoted'], axis = 1), 

    train_original['is_promoted'], 

    test_size = 0.3, random_state = 42)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
X_train.head()
X_train, X_val, Y_train, Y_val = train_test_split(

    X_train, 

    Y_train, 

    test_size = 0.3)
print(X_train.shape)

print(X_val.shape)

print(Y_train.shape)

print(Y_val.shape)
X_train.info()
print("\033[1mColumns with NULL values: \033[0m")

X_train.isnull().sum().to_frame().style.background_gradient('Oranges')
null_education = X_train[X_train['education'].isnull()]

null_education
def beauty_print(df, columns = None, cmap = "Blues"):

    if columns == None:

        columns = df.columns

    return df.style.background_gradient(cmap, subset = columns)
beauty_print(null_education['region'].value_counts().to_frame())
beauty_print(null_education['department'].value_counts().to_frame())
beauty_print(null_education['department'].value_counts().to_frame(), cmap = 'YlOrRd')
for group, item in X_train.groupby(['education']):

    print(group)

    print(item['department'].value_counts(dropna = False).to_frame())
beauty_print(X_train['education'].value_counts(dropna = False, normalize = True).to_frame(), cmap = 'plasma')
beauty_print(X_train['previous_year_rating'].value_counts(dropna = False, normalize = True).to_frame(), cmap = 'plasma')
X_train['education'].fillna(value = 'Unknown', inplace = True)

X_val['education'].fillna(value = 'Unknown', inplace = True)

X_test['education'].fillna(value = 'Unknown', inplace = True)

test_original['education'].fillna(value = 'Unknown', inplace = True)
X_train
Y_train
null_rating = X_train[X_train['previous_year_rating'].isna()]

beauty_print(null_rating['length_of_service'].value_counts(dropna = False, normalize = True).to_frame(), cmap = 'plasma')
null_rating.head()
X_train[['no_of_trainings', 'previous_year_rating', 'avg_training_score', 'KPIs_met >80%', 'awards_won?']].corr().style.background_gradient(cmap = 'Blues')
X_train.groupby('previous_year_rating')[['age', 'no_of_trainings', 'avg_training_score', 'KPIs_met >80%', 'awards_won?']].mean().style.bar()
X_train['previous_year_rating'].fillna(value = 0, inplace = True, axis = 0)

X_val['previous_year_rating'].fillna(value = 0, inplace = True, axis = 0)

X_test['previous_year_rating'].fillna(value = 0, inplace = True, axis = 0)

test_original['previous_year_rating'].fillna(value = 0, inplace = True, axis = 0)
X_train.isnull().sum()
X_train.duplicated().sum()
X_train.T
X_train.T.duplicated().sum()
X_train.drop('employee_id', axis = 1).var().to_frame()
X_train['awards_won?'].value_counts(normalize = True)
X_train.info()
X_train['gender'].value_counts().plot(kind='bar',title = "Gender distribution")
X_train['gender'].value_counts(normalize = True)
combine = X_train.merge(Y_train, left_index=True, right_index=True)

combine
promoted_gender = combine.groupby(['is_promoted','gender'])['is_promoted'].count()

c1 = (promoted_gender[1]/(promoted_gender[0] + promoted_gender[1])).plot.bar(color='green')

c2 = (promoted_gender[0]/(promoted_gender[0] + promoted_gender[1])).plot.bar(color='lightgreen', bottom=(promoted_gender[1]/(promoted_gender[0] + promoted_gender[1])))

handles, labels = c2.get_legend_handles_labels()

plt.axhline(y=(promoted_gender[1]/(promoted_gender[0] + promoted_gender[1])).max())

plt.legend(handles, ['promoted', 'not promoted'])
X_train['education'].value_counts().plot(kind = 'bar', color = "Red", title = "Education distribution")
(X_train['education'].value_counts()/len(X_train)).plot(kind = 'bar', color = "Red", title = "Education distribution percentage")
plt.title('Education distribution across genders')

sns.countplot(x = X_train['education'], hue = X_train['gender'])
plt.figure(figsize=(14, 7))

plt.title('Promoted employees across Education groups')

ax = combine.groupby('is_promoted')['education'].value_counts()

c1 = (ax[1]*100/(ax[0] + ax[1])).plot.bar(color='orange')

c2 = (ax[0]*100/(ax[0] + ax[1])).plot.bar(color='blue', bottom = (ax[1]*100/(ax[0] + ax[1])))

plt.axhline(y=(ax[1]*100/(ax[0] + ax[1])).max(), color='red', linewidth=0.7)

plt.axhline(y=(ax[1]*100/(ax[0] + ax[1])).min(), color='red', linewidth=0.5)

handles, labels = c2.get_legend_handles_labels()

plt.legend(handles, ['promoted', 'not promoted'])

plt.show()
plt.figure(figsize=(16, 8))

plt.title('Age distribution')

sns.distplot(X_train['age'], rug = True,

             color = 'green', hist_kws={

    'color' : "red",

})
plt.title('Count of ratings')

sns.countplot(x=X_train['previous_year_rating'])
sns.countplot(x = combine['previous_year_rating'], hue = combine['is_promoted'])
promoted_ratings = combine.groupby(['is_promoted', 'previous_year_rating'])['is_promoted'].count()

promoted_ratings.xs(1, level='is_promoted').plot(kind='pie', y='is_promoted', autopct='%1.1f%%', shadow=True, startangle=90, legend=False, title='Percentage of employees promoted', explode=(0, 0, 0, 0, 0, 0.2))
promoted_ratings[1]/(promoted_ratings[0]+promoted_ratings[1])
c1 = (promoted_ratings[1]/(promoted_ratings[0]+promoted_ratings[1])).plot.bar()

c2 = (promoted_ratings[0]/(promoted_ratings[0]+promoted_ratings[1])).plot.bar(color='red', bottom=(promoted_ratings[1]/(promoted_ratings[0]+promoted_ratings[1])))

handles, labels = c2.get_legend_handles_labels()

plt.legend(handles, ['promoted', 'not promoted'])

plt.axhline(y=(promoted_ratings[1]/(promoted_ratings[0]+promoted_ratings[1])).max(), color='blue')
plt.title('Award counts')

sns.countplot(x=X_train['awards_won?'])
plt.title('Rating across award receivers')

plt.ylabel('Rating')

X_train.groupby(by=['awards_won?'])['previous_year_rating'].mean().plot(kind = 'bar')
promoted_awards = combine.groupby(['is_promoted','awards_won?'])['is_promoted'].count()

promoted_awards
c1 = (promoted_awards[1]/(promoted_awards[0] + promoted_awards[1])).plot.bar()

c2 = (promoted_awards[0]/(promoted_awards[0] + promoted_awards[1])).plot.bar(color='yellow', bottom=(promoted_awards[1]/(promoted_awards[0] + promoted_awards[1])))

handles, labels = c2.get_legend_handles_labels()

plt.axhline(y=(promoted_awards[1]/(promoted_awards[0] + promoted_awards[1])).max())

plt.legend(handles, ['promoted', 'not promoted'])
plt.title('Training score distribution')

sns.distplot(X_train['avg_training_score'], bins=10, color = 'yellow', hist_kws={

    'color' : "blue"

})
X_train.info()
X_train.head(10)
sns.countplot(x=X_train['recruitment_channel'])
sns.countplot(x=combine['recruitment_channel'], hue=combine['is_promoted'])
promoted_recruitment = combine.groupby(['is_promoted','recruitment_channel'])['is_promoted'].count()

c1 = (promoted_recruitment[1]/(promoted_recruitment[0] + promoted_recruitment[1])).plot.bar()

c2 = (promoted_recruitment[0]/(promoted_recruitment[0] + promoted_recruitment[1])).plot.bar(color='pink', bottom=(promoted_recruitment[1]/(promoted_recruitment[0] + promoted_recruitment[1])))

handles, labels = c2.get_legend_handles_labels()

plt.axhline(y=(promoted_recruitment[1]/(promoted_recruitment[0] + promoted_recruitment[1])).max())

plt.legend(handles, ['promoted', 'not promoted'])
plt.figure(figsize=(14, 7))

chart = sns.countplot(x=X_train['region'], order = X_train['region'].value_counts().index)

chart = chart.set_xticklabels(labels=chart.get_xticklabels(), rotation=70)
plt.figure(figsize=(14, 7))

chart = sns.countplot(x=X_train['region'], order = X_train['region'].value_counts().index)

chart.set_yscale('log')

chart.set_xticklabels(labels=chart.get_xticklabels(), rotation=70)

chart.set_ylabel('log(count)')
sns.distplot(a=X_train['length_of_service'], bins=15)
sns.countplot(combine['no_of_trainings'])
sns.countplot(combine['no_of_trainings'], hue=combine['is_promoted'])
chart = sns.countplot(combine['no_of_trainings'], hue=combine['is_promoted'])

chart = chart.set_yscale('log')
ax = sns.countplot(combine['KPIs_met >80%'], hue=combine['is_promoted'])
promoted_KPI = combine.groupby(['is_promoted','KPIs_met >80%'])['is_promoted'].count()

c1 = (promoted_KPI[1]/(promoted_KPI[0] + promoted_KPI[1])).plot.bar(color='red')

c2 = (promoted_KPI[0]/(promoted_KPI[0] + promoted_KPI[1])).plot.bar(color='pink', bottom=(promoted_KPI[1]/(promoted_KPI[0] + promoted_KPI[1])))

handles, labels = c2.get_legend_handles_labels()

plt.axhline(y=(promoted_KPI[1]/(promoted_KPI[0] + promoted_KPI[1])).max())

plt.legend(handles, ['promoted', 'not promoted'])
sns.lmplot(x='age', y='avg_training_score', hue='is_promoted', 

           markers=['x', 'o'],

           fit_reg=False, data=combine, scatter_kws={"s": 50, "linewidth":2}, height=7,

           palette="Reds",

          )

plt.axvline(x=25)

plt.axvline(x=45)

plt.axhline(y=90)
plt.figure(figsize=(16, 8))

sns.heatmap(data=combine.corr(), cmap='coolwarm', annot=True)
sns.boxplot(X_train['age'])
sns.boxplot(X_train['length_of_service'], color='orange')
sns.boxplot(X_train['avg_training_score'], color='pink')
chart = sns.distplot(X_train['no_of_trainings'], color='blue', kde=False)

chart.set_yscale('log')
more_trainings = (X_train['no_of_trainings'] > 1).astype(np.int)
training_feature_table = combine.merge(more_trainings, left_index=True, right_index=True)

training_feature_table = training_feature_table.groupby(['is_promoted','no_of_trainings_y'])['is_promoted'].count()

c1 = (training_feature_table[1]/(training_feature_table[0] + training_feature_table[1])).plot.bar(color='blue')

c2 = (training_feature_table[0]/(training_feature_table[0] + training_feature_table[1])).plot.bar(color='green', bottom=(training_feature_table[1]/(training_feature_table[0] + training_feature_table[1])))

handles, labels = c2.get_legend_handles_labels()

plt.axhline(y=(training_feature_table[1]/(training_feature_table[0] + training_feature_table[1])).max())

plt.legend(handles, ['promoted', 'not promoted'])
print("Skew in training_feature_table: {}".format(training_feature_table.skew(axis=0)))

print("Kurtosis in training_feature_table: {}".format(training_feature_table.kurt(axis=0)))
X_train.skew(axis=0).to_frame()
# This is Fisher's Kutrosis, where Normal Distribution has a kurtosis of 0,

# whereas in Pearson's Kurtosis has a kurtosis of 3.

X_train.kurt(axis=0).to_frame().sort_values(by=0, ascending = False)
upper_boundary = X_train['length_of_service'].mean() + 3*X_train['length_of_service'].std()

lower_boundary = X_train['length_of_service'].mean() - 3*X_train['length_of_service'].std()

print(upper_boundary, lower_boundary)
outlier_removed = X_train[(X_train['length_of_service'] > lower_boundary) & (X_train['length_of_service'] < upper_boundary)]['length_of_service']

print("There are {} outliers.".format(X_train.shape[0] - len(outlier_removed)))

sns.distplot(a=outlier_removed, bins=15)
outliers = combine[(combine['length_of_service'] < lower_boundary) | (combine['length_of_service'] > upper_boundary)]
sns.countplot(outliers['length_of_service'], hue=outliers['is_promoted'])
combine.info()
from sklearn.utils import resample
combine = pd.concat([X_train, Y_train], axis = 1)
promoted = combine[combine['is_promoted'] == 1]

not_promoted = combine[combine['is_promoted'] != 1]

print(len(promoted), len(not_promoted))
promoted_upsampled = resample(promoted, replace = True, n_samples = len(not_promoted), random_state = 42)
upsampled = pd.concat([promoted_upsampled, not_promoted])

upsampled['is_promoted'].value_counts()
X_train_unsampled = X_train.copy()

Y_train_unsampled = Y_train.copy()



X_train = upsampled.drop('is_promoted', axis = 1)

Y_train = upsampled['is_promoted']
more_trainings = pd.DataFrame()

more_trainings['extra_trainings?'] = np.where(X_train['no_of_trainings'] > 1, 1, 0)
more_trainings['extra_trainings?'].value_counts()
combine.merge(more_trainings['extra_trainings?'], left_index=True, right_index=True).corr()
combine.corr()
combine.describe()
combine.info()
sns.catplot(x='KPIs_met >80%', y='is_promoted', hue='awards_won?', data=combine, kind='point')
awards_or_KPI = pd.DataFrame()

awards_or_KPI['good_performer?'] = combine['KPIs_met >80%'] | combine['awards_won?']

awards_or_KPI
awards_or_KPI.merge(Y_train, left_index=True, right_index=True).corr()
X_train['good_performer?'] = X_train['KPIs_met >80%'] | X_train['awards_won?']

X_test['good_performer?'] = X_test['KPIs_met >80%'] | X_test['awards_won?']

X_val['good_performer?'] = X_val['KPIs_met >80%'] | X_val['awards_won?']

combine['good_performer?'] = combine['KPIs_met >80%'] | combine['awards_won?']

test_original['good_performer?'] = test_original['KPIs_met >80%'] | test_original['awards_won?']



X_train_unsampled['good_performer?'] = X_train_unsampled['KPIs_met >80%'] | X_train_unsampled['awards_won?']
sns.distplot(X_train['avg_training_score'])
def grades(x):

    if x >= 90:

        return 'A'

    if x >= 80:

        return 'B'

    if x >= 70:

        return 'C'

    if x >= 60:

        return 'D'

    if x >= 50:

        return 'E'

    return 'F'

combine['training_score_grade'] = combine['avg_training_score'].apply(grades)
sns.catplot(x='training_score_grade', y='is_promoted', hue='good_performer?', data=combine, kind='point')
combine.info()
combine.head()
combine['training_score_grade'].value_counts()
X_train['training_score_grade'] = X_train['avg_training_score'].apply(grades)

X_test['training_score_grade'] = X_test['avg_training_score'].apply(grades)

X_val['training_score_grade'] = X_val['avg_training_score'].apply(grades)

test_original['training_score_grade'] = test_original['avg_training_score'].apply(grades)



X_train_unsampled['training_score_grade'] = X_train_unsampled['avg_training_score'].apply(grades)
combine['education'].value_counts()
def degrees(x):

    if x == 'Below Secondary':

        return 1

    if x == "Bachelor's":

        return 2

    if x == "Master's & above":

        return 3

    return 0

combine["degrees"] = combine['education'].apply(degrees)
sns.catplot(x='degrees', y='is_promoted', hue='good_performer?', data=combine, kind='point')
X_train['no_of_degrees'] = X_train['education'].apply(degrees)

X_test['no_of_degrees'] = X_test['education'].apply(degrees)

X_val['no_of_degrees'] = X_val['education'].apply(degrees)

test_original['no_of_degrees'] = test_original['education'].apply(degrees)



X_train_unsampled['no_of_degrees'] = X_train_unsampled['education'].apply(degrees)
import scipy.stats as ss

def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

cramers_v(combine['degrees'], combine['is_promoted'])
X_train
def hasBachelors(x):

    if (x == "Bachelor's") | (x == "Master's & above"):

        return 1

    return 0

def hasMasters(x):

    if (x == "Master's & above"):

        return 1

    return 0

def hasSecondary(x):

    if (x != "Unknown"):

        return 1

    return 0



combine['has_Bachelors'] = combine['education'].apply(hasBachelors)

combine['has_Master'] = combine['education'].apply(hasMasters)

combine['has_Secondary'] = combine['education'].apply(hasSecondary)
combine
combine.corr()
Y_train.to_frame().merge(pd.get_dummies(combine['education']), left_index = True, right_index = True).corr()
combine.drop(['has_Bachelors', 'has_Master', 'has_Secondary'], axis = 1, inplace = True)

combine.head()
out, age_bins = pd.qcut(x = combine['age'], q = 4, retbins=True)

out
[*age_bins]
X_train['age_bin'] = pd.cut(x=X_train['age'], bins = age_bins, include_lowest=True)

X_test['age_bin'] = pd.cut(x=X_test['age'], bins = age_bins, include_lowest=True)

X_val['age_bin'] = pd.cut(x=X_val['age'], bins = age_bins, include_lowest=True)

combine['age_bin'] = pd.cut(x=combine['age'], bins = age_bins, include_lowest=True)

test_original['age_bin'] = pd.cut(x=test_original['age'], bins = age_bins, include_lowest=True)



X_train_unsampled['age_bin'] = pd.cut(x=X_train_unsampled['age'], bins = age_bins, include_lowest=True)
X_train
cramers_v(combine['is_promoted'], combine['age_bin'])
sns.countplot(X_train['age_bin'])
sns.countplot(combine['length_of_service'])
out, service_bins = pd.cut(x = combine['length_of_service'], bins = [0, 3, 5, 7, 40], retbins=True)

print(*service_bins)

combine['service_bin'] = pd.cut(x = combine['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])
combine['service_bin'].value_counts().plot.bar()
combine
X_train['service_bin'] = pd.cut(x = X_train['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])

X_val['service_bin'] = pd.cut(x = X_val['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])

X_test['service_bin'] = pd.cut(x = X_test['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])

test_original['service_bin'] = pd.cut(x = test_original['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])



X_train_unsampled['service_bin'] = pd.cut(x = X_train_unsampled['length_of_service'], bins = service_bins, include_lowest=True, labels=['beginner', 'intermediate', 'expert', 'senior expert'])
sns.catplot(x='service_bin', y='is_promoted', data=combine, kind='point')
combine.merge(combine.groupby(['department', 'recruitment_channel', 'training_score_grade'])['avg_training_score'].agg('mean'), on=['department', 'recruitment_channel', 'training_score_grade'], right_index=True)
cramers_v(combine.merge(combine.groupby(['department','training_score_grade'])['avg_training_score'].agg('mean'), on=['department', 'training_score_grade'], right_index=True)['avg_training_score_y'], combine['is_promoted'])
combine.groupby(['department', 'recruitment_channel', 'training_score_grade'])['avg_training_score'].agg('mean')
cramers_v(combine['department'], combine['is_promoted'])
combine.groupby(['department','training_score_grade'])['age'].agg('count')
sns.countplot(x='department', data=combine)

plt.xticks(rotation=45)
combine_copy = combine.copy()



def group_dep(x):

    if (x == "HR") or (x == "Legal") or (x == "Finance") or (x == "R&D"):

        return "Others"

    return x

combine_copy['department'] = combine_copy['department'].apply(group_dep)
sns.catplot(x='department', y='is_promoted', hue='training_score_grade', data=combine, kind='point')

plt.xticks(rotation=90)
department_promotion = pd.crosstab(combine['department'], combine['is_promoted'])

department_promotion['%promoted'] = department_promotion[1]*100/(department_promotion[0] + department_promotion[1])

department_promotion
department_promotion = pd.crosstab(combine_copy['department'], combine_copy['is_promoted'])



department_promotion['%promoted'] = department_promotion[1]*100/(department_promotion[0] + department_promotion[1])

department_promotion
combine_copy['department'].value_counts().plot.bar()
sns.catplot(x='department', y='is_promoted', data=combine_copy, kind='point')
X_train['department'] = X_train['department'].apply(group_dep)

X_test['department'] = X_test['department'].apply(group_dep)

X_val['department'] = X_val['department'].apply(group_dep)

test_original['department'] = test_original['department'].apply(group_dep)

combine['department'] = combine['department'].apply(group_dep)



X_train_unsampled['department'] = X_train_unsampled['department'].apply(group_dep)
combine['department'].value_counts()
combine.info()
combine.groupby(by=['department', 'good_performer?'])['avg_training_score'].mean()
combine.groupby(by=['department', 'recruitment_channel'])['avg_training_score'].mean()
combine.groupby(by=['department', 'education'])['avg_training_score'].mean()
combine['service_bin'].value_counts(dropna=False)
combine['age_bin'].value_counts()
combine.groupby(['age_bin', 'training_score_grade'])['age'].count()
pd.crosstab(combine['age_bin'], combine['training_score_grade'], normalize=True).style.background_gradient(cmap='cool')
pd.crosstab(combine['service_bin'], combine['training_score_grade'], normalize=True).style.background_gradient(cmap='cool')
combine
print(cramers_v(combine['service_bin'], combine['is_promoted']), cramers_v(combine['length_of_service'], combine['is_promoted']))
combine['avg_training_score']/combine['length_of_service']
cramers_v((combine['avg_training_score']/combine['length_of_service']), combine['is_promoted'])
combine['age_bin'].value_counts()
import datetime

print(datetime.datetime.today().year)
combine['joining_year'] = 2020 - combine['length_of_service'] + 1
combine
X_train['joining_year'] = 2020 - X_train['length_of_service'] + 1

X_test['joining_year'] = 2020 - X_test['length_of_service'] + 1

X_val['joining_year'] = 2020 - X_val['length_of_service'] + 1

test_original['joining_year'] = 2020 - test_original['length_of_service'] + 1



X_train_unsampled['joining_year'] = 2020 - X_train_unsampled['length_of_service'] + 1
sns.distplot(combine['joining_year'])

print(combine['joining_year'].min(), combine['joining_year'].max())
sns.boxplot(combine['joining_year'])
upper_boundary = combine['joining_year'].mean() + 3*combine['joining_year'].std()

lower_boundary = combine['joining_year'].mean() - 3*combine['joining_year'].std()

print(upper_boundary, lower_boundary)
sns.distplot(combine[combine['joining_year'] > lower_boundary]['joining_year'])
sns.boxplot(combine[combine['joining_year'] > lower_boundary]['joining_year'])
combine[combine['joining_year'] < lower_boundary]['is_promoted'].value_counts()
X_train.info()
from sklearn.preprocessing import LabelEncoder

gender_encode = LabelEncoder()

gender_encode.fit(combine['gender'])
gender_dict = {key : value for (key, value) in zip(gender_encode.classes_, gender_encode.transform(gender_encode.classes_))}
gender_dict
combine['gender_encoded'] = gender_encode.transform(combine['gender'])

X_train['gender_encoded'] = gender_encode.transform(X_train['gender'])

X_test['gender_encoded'] = gender_encode.transform(X_test['gender'])

X_val['gender_encoded'] = gender_encode.transform(X_val['gender'])

test_original['gender_encoded'] = gender_encode.transform(test_original['gender'])



X_train_unsampled['gender_encoded'] = gender_encode.transform(X_train_unsampled['gender'])
X_train['gender_encoded']
department_encode = LabelEncoder()

department_encode.fit(X_train['department'])
department_encode.classes_
department_dict = {

    key : value for (key, value) in zip(department_encode.classes_, department_encode.transform(department_encode.classes_))

}

department_dict
combine['department_encode'] = department_encode.transform(combine['department'])

X_train['department_encode'] = department_encode.transform(X_train['department'])

X_test['department_encode'] = department_encode.transform(X_test['department'])

X_val['department_encode'] = department_encode.transform(X_val['department'])

test_original['department_encode'] = department_encode.transform(test_original['department'])



X_train_unsampled['department_encode'] = department_encode.transform(X_train_unsampled['department'])



X_train['department_encode']
region_encode = LabelEncoder()

region_encode.fit(X_train['region'])



region_dict = {

    key : value for (key, value) in zip(region_encode.classes_, region_encode.transform(region_encode.classes_))

}



combine['region_encode'] = region_encode.transform(combine['region'])

X_train['region_encode'] = region_encode.transform(X_train['region'])

X_test['region_encode'] = region_encode.transform(X_test['region'])

X_val['region_encode'] = region_encode.transform(X_val['region'])

test_original['region_encode'] = region_encode.transform(test_original['region'])



X_train_unsampled['region_encode'] = region_encode.transform(X_train_unsampled['region'])



X_train['region_encode']
test_original
education_encode = LabelEncoder()

education_encode.fit(X_train['education'])



education_dict = {

    key : value for (key, value) in zip(education_encode.classes_, education_encode.transform(education_encode.classes_))

}



combine['education_encode'] = education_encode.transform(combine['education'])

X_train['education_encode'] = education_encode.transform(X_train['education'])

X_test['education_encode'] = education_encode.transform(X_test['education'])

X_val['education_encode'] = education_encode.transform(X_val['education'])

test_original['education_encode'] = education_encode.transform(test_original['education'])



X_train_unsampled['education_encode'] = education_encode.transform(X_train_unsampled['education'])



X_train['education_encode']
recruitment_channel_encode = LabelEncoder()

recruitment_channel_encode.fit(X_train['recruitment_channel'])



recruitment_dict = {

    key : value for (key, value) in zip(recruitment_channel_encode.classes_, recruitment_channel_encode.transform(recruitment_channel_encode.classes_))

}



combine['recruitment_channel_encode'] = recruitment_channel_encode.transform(combine['recruitment_channel'])

X_train['recruitment_channel_encode'] = recruitment_channel_encode.transform(X_train['recruitment_channel'])

X_test['recruitment_channel_encode'] = recruitment_channel_encode.transform(X_test['recruitment_channel'])

X_val['recruitment_channel_encode'] = recruitment_channel_encode.transform(X_val['recruitment_channel'])

test_original['recruitment_channel_encode'] = recruitment_channel_encode.transform(test_original['recruitment_channel'])



X_train_unsampled['recruitment_channel_encode'] = recruitment_channel_encode.transform(X_train_unsampled['recruitment_channel'])



X_train['recruitment_channel_encode']
grade_encode = LabelEncoder()

grade_encode.fit(X_train['training_score_grade'])



grage_table = {

    key : value for (key, value) in zip(grade_encode.classes_, grade_encode.transform(grade_encode.classes_))

}



combine['grade_encode'] = grade_encode.transform(combine['training_score_grade'])

X_train['grade_encode'] = grade_encode.transform(X_train['training_score_grade'])

X_test['grade_encode'] = grade_encode.transform(X_test['training_score_grade'])

X_val['grade_encode'] = grade_encode.transform(X_val['training_score_grade'])

test_original['grade_encode'] = grade_encode.transform(test_original['training_score_grade'])



X_train_unsampled['grade_encode'] = grade_encode.transform(X_train_unsampled['training_score_grade'])



X_train['grade_encode']
X_val[X_val['service_bin'].isnull()]
service_bins
service_bin_encode = LabelEncoder()

service_bin_encode.fit(X_train['service_bin'])



service_bin_dict = {

    key : value for (key, value) in zip(service_bin_encode.classes_, service_bin_encode.transform(service_bin_encode.classes_))

}



combine['service_bin_encode'] = service_bin_encode.transform(combine['service_bin'])

X_train['service_bin_encode'] = service_bin_encode.transform(X_train['service_bin'])

X_test['service_bin_encode'] = service_bin_encode.transform(X_test['service_bin'])

X_val['service_bin_encode'] = service_bin_encode.transform(X_val['service_bin'])

test_original['service_bin_encode'] = service_bin_encode.transform(test_original['service_bin'])



X_train_unsampled['service_bin_encode'] = service_bin_encode.transform(X_train_unsampled['service_bin'])



X_train['service_bin_encode']
age_bin_encode = LabelEncoder()

age_bin_encode.fit(X_train['age_bin'])



age_bin_dict = {

    key : value for (key, value) in zip(age_bin_encode.classes_, age_bin_encode.transform(age_bin_encode.classes_))

}



combine['age_bin_encode'] = age_bin_encode.transform(combine['age_bin'])

X_train['age_bin_encode'] = age_bin_encode.transform(X_train['age_bin'])

X_test['age_bin_encode'] = age_bin_encode.transform(X_test['age_bin'])

X_val['age_bin_encode'] = age_bin_encode.transform(X_val['age_bin'])

test_original['age_bin_encode'] = age_bin_encode.transform(test_original['age_bin'])



X_train_unsampled['age_bin_encode'] = age_bin_encode.transform(X_train_unsampled['age_bin'])



X_train['age_bin_encode']
object_types = combine.select_dtypes(exclude=np.number).dtypes.index

object_types
X_train.drop(object_types, inplace = True, axis = 1)

X_test.drop(object_types, inplace = True, axis = 1)

X_val.drop(object_types, inplace = True, axis = 1)

combine.drop(object_types, inplace = True, axis = 1)

test_original.drop(object_types, inplace = True, axis = 1)



X_train_unsampled.drop(object_types, inplace = True, axis = 1)
sns.heatmap(X_train.corr(method = 'spearman'))
corr_features = set()



corr_matrix = X_train.corr()



sns.heatmap(corr_matrix)



for i in range(len(corr_matrix .columns)):

    for j in range(i):

        if abs(corr_matrix.iloc[i, j]) > 0.85:

            colname = corr_matrix.columns[i]

            corr_features.add(colname)

print(corr_features)
X_train.drop(corr_features, axis = 1, inplace = True)

X_test.drop(corr_features, axis = 1, inplace = True)

X_val.drop(corr_features, axis = 1, inplace = True)

test_original.drop(corr_features, axis = 1, inplace = True)



X_train_unsampled.drop(corr_features, axis = 1, inplace = True)
X_train.shape
X_train.drop('employee_id', axis = 1, inplace = True)

X_test.drop('employee_id', axis = 1, inplace = True)

X_val.drop('employee_id', axis = 1, inplace = True)

#test_original.drop('employee_id', axis = 1, inplace = True)



X_train_unsampled.drop('employee_id', axis = 1, inplace = True)
combine.drop([*corr_features, 'employee_id'], axis = 1, inplace = True)
X_train
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini',n_estimators = 100, verbose=1)

selector = RFE(classifier, n_features_to_select=6, step=1, verbose=2)
selector.fit(X_train, Y_train)
X_train.columns[selector.get_support()]
X_train.columns
selector.ranking_
pd.DataFrame((key, value) for (key, value) in zip(X_train.columns, selector.ranking_)).sort_values(1)
X_train.education_encode.value_counts()
X_train.var()
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
model = sfs(classifier,k_features=6,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='f1')
model.fit(X_train, Y_train)
model.k_feature_names_
model.k_score_
features = [*model.k_feature_names_]
features
classifier.fit(X_train[features], Y_train)
Y_val_pred = classifier.predict(X_val[features])
from sklearn.metrics import confusion_matrix, f1_score, classification_report
confusion_matrix(Y_val, Y_val_pred)
f1_score(y_true=Y_val, y_pred=Y_val_pred)
pd.DataFrame((key,value) for (key, value) in zip(features, classifier.feature_importances_)).sort_values(1, ascending = False).style.background_gradient(cmap='Reds')
print(classification_report(Y_val, Y_val_pred))
X_train_unsampled
Y_train_unsampled
from sklearn.utils.class_weight import compute_class_weight 



class_weights = compute_class_weight('balanced', [0, 1], Y_train_unsampled)
classifier_with_cw = RandomForestClassifier(criterion='gini',n_estimators = 100, verbose=1, class_weight={key: value for (key, value) in enumerate(class_weights)})

selector_with_cw = RFE(classifier_with_cw, n_features_to_select=6, step=1, verbose=2)
selector_with_cw.fit(X_train_unsampled, Y_train_unsampled)
feature_list = [*X_train_unsampled.columns[selector_with_cw.get_support()]]

feature_list
selector_with_cw.ranking_
classifier_with_cw.fit(X_train_unsampled[feature_list], Y_train_unsampled)
Y_val_unsampled_pred = classifier_with_cw.predict(X_val[feature_list])
confusion_matrix(Y_val, Y_val_unsampled_pred)
f1_score(y_true=Y_val, y_pred=Y_val_unsampled_pred)
pd.DataFrame((key,value) for (key, value) in zip(feature_list, classifier_with_cw.feature_importances_)).sort_values(1, ascending = False).style.background_gradient(cmap='Reds')
{key: value for (key, value) in enumerate(class_weights)}
print(classification_report(Y_val, Y_val_unsampled_pred))
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state = 42)

X_train_smote, Y_train_smote = sm.fit_sample(X_train_unsampled, Y_train_unsampled)
print(X_train_smote.shape, Y_train_smote.shape)
Y_train_smote.value_counts()
classifier.fit(X_train_smote[feature_list], Y_train_smote)
classifier.predict(X_val[feature_list])
print(classification_report(Y_val, classifier.predict(X_val[feature_list])))
from sklearn import *
tree_classifier_cw = tree.DecisionTreeClassifier(criterion='gini', 

                                                 splitter='best', 

                                                 min_samples_split = 0.1, 

                                                 class_weight={key:value for (key, value) in enumerate(class_weights)}, 

                                                 random_state = 42)
tree_classifier_cw.fit(X_train_unsampled, Y_train_unsampled)
tree_pred = tree_classifier_cw.predict(X_val)
tree_classifier_cw.predict_proba(X_val)
plt.figure(figsize = (30, 10), dpi=200)

tree.plot_tree(tree_classifier_cw)
tree_classifier_cw.score(X_val, Y_val)
print(classification_report(Y_val, tree_classifier_cw.predict(X_val)))
classifier_with_cw
classifier_with_cw = RandomForestClassifier(criterion='gini',  

                                             n_estimators = 1000,

                                             #min_samples_split = 0.1,

                                             class_weight={key:value for (key, value) in enumerate(class_weights)}, 

                                             n_jobs = -1,

                                             random_state = 42,

                                             verbose=1)
classifier_with_cw.fit(X_train_unsampled[feature_list], Y_train_unsampled)
print(classification_report(Y_val, classifier_with_cw.predict(X_val[feature_list])))
feature_list
search_params = {

    'criterion' : ['gini', 'entropy'],

    'max_depth' : list(np.linspace(10, 500, 5, dtype = int, endpoint = True)) + [None],

    'max_features' : ['auto', 'sqrt', 'log2', None],

    'min_samples_leaf' : list(np.linspace(1, 500, 5, dtype = int, endpoint = True)),

    'min_samples_split' : list(np.linspace(0.1, 1, 5, dtype = float, endpoint = True)),

    'n_estimators' : list(np.linspace(100, 1000, 10, dtype = int, endpoint = True)),

}
hyper_classifier_RF = RandomForestClassifier(class_weight = {key: value for (key, value) in enumerate(class_weights)})
hyper_model = model_selection.RandomizedSearchCV(estimator = hyper_classifier_RF, 

                                                param_distributions = search_params,

                                                n_iter = 30,

                                                cv = 5,

                                                verbose = 5,

                                                random_state = 42,

                                                n_jobs = -1,

                                                scoring = 'f1')
hyper_model.fit(X_train_unsampled[feature_list], Y_train_unsampled)
hyper_model.cv_results_.keys()
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model.cv_results_),

              values = 'mean_test_score', index = 'param_max_features', columns = 'param_criterion'), annot = True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model.cv_results_),

              values = 'mean_test_score', index = 'param_min_samples_split', columns = 'param_criterion'), annot=True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model.cv_results_),

              values = 'mean_test_score', index = 'param_min_samples_leaf', columns = 'param_criterion'), annot=True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model.cv_results_),

              values = 'mean_test_score', index = 'param_n_estimators', columns = 'param_criterion'), annot = True)
hyper_model.best_estimator_
hyper_pred_val = hyper_model.best_estimator_.predict(X_val[feature_list])

print(confusion_matrix(Y_val,hyper_pred_val))

print(classification_report(Y_val,hyper_pred_val))
search_params = {

    'criterion' : ['gini', 'entropy'],

    'max_depth' : list(np.linspace(10, 1000, 4, dtype = int, endpoint = True)) + [None],

    'max_features' : ['auto', 'sqrt', 'log2'],

    'min_samples_leaf' : list(np.linspace(1, 500, 5, dtype = int, endpoint = True)),

    'min_samples_split' : list(np.linspace(0.1, 1, 4, dtype = float, endpoint = True)),

    'n_estimators' : list(np.linspace(100, 1000, 5, dtype = int, endpoint = True)),

}
# hyper_classifier_RF = RandomForestClassifier(class_weight = {key: value for (key, value) in enumerate(class_weights)})



# hyper_model_grid = model_selection.GridSearchCV(estimator = hyper_classifier_RF, 

#                                                 param_grid = search_params,

#                                                 cv = 3,

#                                                 verbose = 5,

#                                                 n_jobs = -1,

#                                                 scoring = 'f1')
# hyper_model_grid.fit(X_train_unsampled[feature_list], Y_train_unsampled)
import pickle



with open('/kaggle/input/GridSearchRandomforest.pkl', 'rb') as handle:

    hyper_model_grid = pickle.load(handle)



print(hyper_model_grid)



print(hyper_model_grid.cv_results_.keys())
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model_grid.cv_results_),

              values = 'mean_test_score', index = 'param_max_features', columns = 'param_criterion'), annot = True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model_grid.cv_results_),

              values = 'mean_test_score', index = 'param_min_samples_split', columns = 'param_criterion'), annot=True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model_grid.cv_results_),

              values = 'mean_test_score', index = 'param_min_samples_leaf', columns = 'param_criterion'), annot=True)
sns.heatmap(pd.pivot_table(pd.DataFrame(hyper_model_grid.cv_results_),

              values = 'mean_test_score', index = 'param_n_estimators', columns = 'param_criterion'), annot = True)
hyper_model_grid.best_estimator_
hyper_pred_val_grid = hyper_model_grid.best_estimator_.predict(X_val[feature_list])

print(confusion_matrix(Y_val,hyper_pred_val_grid))

print(classification_report(Y_val,hyper_pred_val_grid))
# import pickle



# with open('GridSearchRandomforest.pkl', 'wb') as handle:

#     pickle.dump(hyper_model_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('GridSearchRandomforest.pkl', 'rb') as handle:

#     b = pickle.load(handle)



# print(b)
print(classification_report(Y_val, hyper_model_grid.best_estimator_.predict(X_val[feature_list])))
print(classification_report(Y_val, classifier_with_cw.predict(X_val[feature_list])))
param_dict = hyper_model_grid.best_estimator_.get_params()

param_dict['class_weight'] = None;

param_dict['verbose'] = 1

param_dict['n_estimators'] = 1000

param_dict
classifier_hyper_smote = RandomForestClassifier(**param_dict)
classifier_hyper_smote.fit(X_train_smote[feature_list], Y_train_smote)
print(classification_report(Y_val, classifier_hyper_smote.predict(X_val[feature_list])))
import lightgbm as lgbm
lgbm_train = lgbm.Dataset(X_train_unsampled[feature_list], label = Y_train_unsampled, free_raw_data=False)
lgbm_model = lgbm.train({

    'learning_rate': 0.001,

    'metric' : ['f1', 'recall'],

    'objective' : 'binary',

    'criterion' : 'entropy',

    'boosting_type' : 'gbdt', 

    'n_estimators' : 5000,

    'class_weight' : 'balanced'

    }, 

    lgbm_train, 

    1000, 

    feature_name=feature_list, 

    verbose_eval=True)
lgbm_model
y_pred = lgbm_model.predict(X_val[feature_list])

y_pred
for i, v in enumerate(y_pred):

    if(v >= 0.4):

        y_pred[i] = 1

    else:

        y_pred[i] = 0
print(classification_report(Y_val, y_pred))
y_test_pred = lgbm_model.predict(X_test[feature_list])



for i, v in enumerate(y_test_pred):

    if(v >= 0.4):

        y_test_pred[i] = 1

    else:

        y_test_pred[i] = 0

print(classification_report(Y_test, y_test_pred))
original_train_pred = lgbm_model.predict(test_original[feature_list])



for i, v in enumerate(original_train_pred):

    if(v >= 0.4):

        original_train_pred[i] = 1

    else:

        original_train_pred[i] = 0

original_train_pred
test_original_copy = test_original.copy()



test_original_copy['is_promoted'] = original_train_pred

test_original_copy['is_promoted'] = test_original_copy['is_promoted'].astype(int)



test_original_copy.to_csv('submission.csv', columns = ['employee_id', 'is_promoted'], index = False)

train_original[:]
y_pred_original_rf = classifier_with_cw.predict(test_original[feature_list])
test_original_copy['is_promoted'] = y_pred_original_rf
test_original_copy
test_original_copy.to_csv('submission_rf.csv', columns = ['employee_id', 'is_promoted'], index = False)
y_test_pred = lgbm_model.predict(X_test[feature_list])



for i, v in enumerate(y_test_pred):

    if(v >= 0.4):

        y_test_pred[i] = 1

    else:

        y_test_pred[i] = 0

print(classification_report(Y_test, y_test_pred))