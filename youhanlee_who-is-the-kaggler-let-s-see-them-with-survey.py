import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import re

import seaborn as sns

import wordcloud



# from mpl_toolkits.basemap import Basemap

from os import path

from PIL import Image

import missingno as msno

%matplotlib inline
df_multipleChoice = pd.read_csv("../input/multipleChoiceResponses.csv",  encoding="ISO-8859-1", low_memory=False)

df_freeform = pd.read_csv("../input/freeformResponses.csv", low_memory=False)

df_schema = pd.read_csv("../input/schema.csv", index_col="Column")



multiple_choice_columns = df_multipleChoice.columns

freeform_columns = df_freeform.columns
def check_NaN_percentage(df, df_columns):

    print("--------------------NaN value percentage--------------------")

    for col in df_columns:

        print("column: {:>20}\t Percent of NaN value: {:.2f}% (Total not NaN response: {})".format(col, 100 * df[col].isnull().sum() / len(df[col]), len(df[col]) - df[col].isnull().sum()))
check_NaN_percentage(df_multipleChoice, multiple_choice_columns)
msno.matrix(df=df_multipleChoice.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))  
check_NaN_percentage(df_freeform, freeform_columns)
msno.matrix(df=df_freeform.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
len(multiple_choice_columns) + len(freeform_columns) == df_schema.shape[0]
df_schema.head()
df_schema['Asked'].value_counts().sum()
all_features = df_schema.index
def make_meta(all_features):

    data = []

    for feature in all_features:

        # which form this feature included

        if feature in multiple_choice_columns:

            WhichForm = "Multiple_choice"

            Response_rate = 100 * df_multipleChoice[feature].isnull().sum() / len(df_multipleChoice[feature])

            dtype = str(df_multipleChoice[feature].dtype)

        else:

            WhichForm = "FreeForm"

            Response_rate = 100 * df_freeform[feature].isnull().sum() / len(df_freeform[feature])

            dtype = str(df_freeform[feature].dtype)

        # target

        target = df_schema.loc[feature, 'Asked']

        Question = df_schema.loc[feature, 'Question']

        temp_dict = {

            "feature": feature,

            "WhichForm": WhichForm,

            "target": target,

            "Question": Question,

            "Response_rate": 100 - np.round(Response_rate, 1),

            "dtype": dtype

        }

        data.append(temp_dict)

    return data

data = make_meta(all_features)

meta = pd.DataFrame(data, columns=['feature', 'WhichForm', 'target', 'Question', 'Response_rate', 'dtype'])

meta.set_index('feature', inplace=True)
meta
feature_all_float = meta.loc[(meta['target'] == 'All') & (meta['WhichForm'] == 'Multiple_choice') & (meta['dtype'] == 'float64')]
feature_all_float
fig = plt.figure(figsize=(10, 10))

sns.set(font_scale=2)

sns.distplot(df_multipleChoice.loc[~df_multipleChoice['Age'].isnull()]['Age'])

plt.title("{}\nResponse Rate: {}%".format(meta.loc['Age', 'Question'], meta.loc['Age', 'Response_rate']))
for feature in feature_all_float.index[1:]:

    fig = plt.figure(figsize=(10, 10))

    sns.set(font_scale=2)

    sns.distplot(df_multipleChoice.loc[~df_multipleChoice[feature].isnull()][feature])

    plt.title("{}\n{}\nResponse rate: {}%".format(meta.loc[feature, 'Question'][:int(len(meta.loc[feature, 'Question'])/2)], 

                              meta.loc[feature, 'Question'][int(len(meta.loc[feature, 'Question'])/2):],

                                                meta.loc[feature, 'Response_rate']))
def CategorizeAge(df):

    df.loc[(0.0 <= df['Age']) & (df['Age'] < 18.0), 'CategorizedAge'] = '0~18' # before university

    df.loc[(19 <= df['Age']) & (df['Age'] < 26), 'CategorizedAge'] = '19~25' # during university

    df.loc[(26 <= df['Age']) & (df['Age'] < 41), 'CategorizedAge'] = '26~40' # Hard worker

    df.loc[(41 <= df['Age']) & (df['Age'] < 61), 'CategorizedAge'] = '41~60' # more experienced

    df.loc[(61 <= df['Age']), 'CategorizedAge'] = '61~' # Master :)

#     df.loc[(0 <= df['Age']) & (df['Age'] < 10), 'CategorizedAge'] = '0~10'

#     df.loc[(10 <= df['Age']) & (df['Age'] < 20), 'CategorizedAge'] = '10~20'

#     df.loc[(20 <= df['Age']) & (df['Age'] < 30), 'CategorizedAge'] = '20~30'

#     df.loc[(30 <= df['Age']) & (df['Age'] < 40), 'CategorizedAge'] = '30~40'

#     df.loc[(40 <= df['Age']) & (df['Age'] < 50), 'CategorizedAge'] = '40~50'

#     df.loc[(50 <= df['Age']) & (df['Age'] < 60), 'CategorizedAge'] = '50~60'

#     df.loc[(60 <= df['Age']) & (df['Age'] < 70), 'CategorizedAge'] = '60~70'

#     df.loc[(70 <= df['Age']) & (df['Age'] < 80), 'CategorizedAge'] = '70~80'

#     df.loc[(80 <= df['Age']), 'CategorizedAge'] = '80~'

    return df
df_multipleChoice = CategorizeAge(df_multipleChoice.loc[df_multipleChoice['Age'].notnull()])
Categorized_Age = df_multipleChoice.groupby(['CategorizedAge'])['Age'].count().reset_index().set_index('CategorizedAge')

Categorized_Age['Percent'] = 100* np.round(Categorized_Age['Age'] / Categorized_Age['Age'].sum(), 3)
Categorized_Age
for count in range(6):

    if count == 0:

        continue

    fig, ax = plt.subplots(figsize=(5, 5))



    target_feature = feature_all_float.index[count]

    sns.distplot(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == '0~18') & (df_multipleChoice[target_feature].notnull())][target_feature], 

                 hist=False, label='0~18', ax=ax)

    sns.distplot(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == '19~25') & (df_multipleChoice[target_feature].notnull())][target_feature], 

                 hist=False, label='19~25', ax=ax)

    sns.distplot(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == '26~40') & (df_multipleChoice[target_feature].notnull())][target_feature], 

                 hist=False, label='26~40', ax=ax)

    sns.distplot(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == '41~60') & (df_multipleChoice[target_feature].notnull())][target_feature], 

                 hist=False, label='41~60', ax=ax)

    sns.distplot(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == '60~') & (df_multipleChoice[target_feature].notnull())][target_feature], 

                 hist=False, label='60~', ax=ax)
feature_all_object = meta.loc[(meta['target'] == 'All') & (meta['WhichForm'] == 'Multiple_choice') & (meta['dtype'] != 'float64')]
feature_all_object
feature_all_object.shape
count = 0

fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:100]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
for category in ['0~18', '19~25', '26~40', '41~60', '61~']:

    fig = plt.figure(figsize=(10, 10))

    temp_results = pd.DataFrame(df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == category) \

                                       & (df_multipleChoice[target_feature].notnull())][target_feature].value_counts()[:10]\

                                        )

    temp_results['Percent'] = np.round(100 * temp_results[target_feature] / df_multipleChoice.loc[(df_multipleChoice['CategorizedAge'] == category) \

                                       & (df_multipleChoice[target_feature].notnull())][target_feature].value_counts().sum())

    ax = sns.barplot(x=temp_results['MLToolNextYearSelect'].values, y=temp_results.index)

    plt.title("{}\nAge category: {}".format(meta.loc[target_feature, 'Question'], 

                                              category))
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(20, 20))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:6]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

                ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(12, 12))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(15, 15))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\



fig = plt.figure(figsize=(15, 15))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:7]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1
fig = plt.figure(figsize=(15, 15))

target_feature = feature_all_object.index[count]

temp_data = df_multipleChoice.loc[(df_multipleChoice[target_feature].isin(df_multipleChoice[target_feature].value_counts().index[:10]))]

ncount = temp_data[target_feature].value_counts().sum()

ax = sns.countplot(y=target_feature, data=temp_data, 

              order=temp_data[target_feature].value_counts().index)

plt.title("{}\nResponse rate: {:.1f}%".format(meta.loc[target_feature, 'Question'], 

                                          meta.loc[target_feature, 'Response_rate']))

for p in ax.patches:

    x=p.get_bbox().get_points()[1,0]

    y=p.get_bbox().get_points()[:,1]

    ax.annotate('{:.1f}%'.format(100.*x/ncount), (x, y.mean()+0.1), 

            ha='center', va='bottom') # set the alignment of the text\

    

count = count + 1