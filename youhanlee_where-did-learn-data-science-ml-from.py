import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

import seaborn as sns



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
df_multipleChoice = pd.read_csv("../input/multipleChoiceResponses.csv",  encoding="ISO-8859-1", low_memory=False)

df_freeform = pd.read_csv("../input/freeformResponses.csv", low_memory=False)

df_schema = pd.read_csv("../input/schema.csv", index_col="Column")

df_conversion = pd.read_csv('../input/conversionRates.csv')



multiple_choice_columns = df_multipleChoice.columns

freeform_columns = df_freeform.columns
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
def plot_worldmap(title, bar_title):

    data = [ dict(

            type = 'choropleth',

            locations = results['nation'],

            z = results[target_feature],

            text = results['nation'],

            colorscale='jet',

            autocolorscale = False,

            locationmode = 'country names',

            marker = dict(

                line = dict (

                    color = 'rgb(180,180,180)',

                    width = 0.5

                ) ),

            colorbar = dict(

                autotick = False,

                tickprefix = '',

                title = bar_title),

          ) ]



    layout = dict(

        title = title,

        geo = dict(

            showocean = True,

            oceancolor = 'rgb(28,107,160)',

            showframe = False,

            showcoastlines = False,

            projection = dict(

                type = 'Mercator'

            )

        )

    )



    fig = dict( data=data, layout=layout )

    py.iplot( fig, validate=False, filename='d3-world-map' )
meta[(meta['WhichForm'] == 'Multiple_choice') & (meta['target'] == 'All') & (meta['dtype'] == 'float64')]
LearningCategory = ['LearningCategorySelftTaught', 'LearningCategoryOnlineCourses',

                         'LearningCategoryWork', 'LearningCategoryUniversity', 'LearningCategoryKaggle',

                         'LearningCategoryOther']
temp = df_multipleChoice[['LearningCategorySelftTaught', 'LearningCategoryOnlineCourses',

                         'LearningCategoryWork', 'LearningCategoryUniversity', 'LearningCategoryKaggle',

                         'LearningCategoryOther', 'Country']].dropna()



temp['LearningCategoryOnline'] = temp['LearningCategoryKaggle'] + temp['LearningCategoryOnlineCourses']

temp['LearningCategoryOffline'] = temp['LearningCategoryUniversity'] + temp['LearningCategorySelftTaught'] + temp['LearningCategoryWork'] + temp['LearningCategoryOther']



temp['LeaningCategoryOffline'] = temp['LearningCategoryUniversity'] + temp['LearningCategorySelftTaught'] + temp['LearningCategoryWork'] + temp['LearningCategoryOther']

temp['Ratio_Online_Offline'] = temp['LearningCategoryOnline'] / (temp['LearningCategoryOffline'] + 1)
target_feature = 'Ratio_Online_Offline'



nations = []

target_numbers = []

for nation in temp['Country'].unique():

    if nation == "Other":

        continue

    nations.append(nation)

    target_numbers.append(np.round(temp.loc[temp['Country'] == nation][target_feature].mean(), 2))

    

results = pd.DataFrame({'nation': nations, target_feature: target_numbers})



results.loc[results['nation'] == "People 's Republic of China", target_feature] = (float(results.loc[results['nation'] == "People 's Republic of China", target_feature]) 

                                                                                      + float(results.loc[results['nation'] == "Republic of China", target_feature]))/2

results = results.drop(50).reset_index(drop=True)
title = "Where did you learn from - Online or Offline?"

bar_title = "Ratio(Online/Offline)"

plot_worldmap(title, bar_title)
results.sort_values(target_feature, ascending=False)[:10]
top_lists = results.sort_values(target_feature, ascending=False)[:3]['nation'].values

for nation in top_lists:

    fig = plt.figure(figsize=(8, 5))

    means = []

    for category in LearningCategory:

        means.append(temp.loc[temp['Country'] == nation][category].mean())

    temp_results = pd.Series(LearningCategory, means)

    plt.title(nation, fontsize=20)

    sns.barplot(temp_results.index, temp_results.values)
results.sort_values(target_feature, ascending=True)[:10]
bottom_lists = results.sort_values(target_feature, ascending=True)[:3]['nation'].values

for nation in bottom_lists:

    fig = plt.figure(figsize=(8, 5))

    means = []

    for category in LearningCategory:

        means.append(temp.loc[temp['Country'] == nation][category].mean())

    temp_results = pd.Series(LearningCategory, means)

    plt.title(nation, fontsize=20)

    sns.barplot(temp_results.index, temp_results.values)
temp['Ratio_Online_University'] = temp['LearningCategoryOnlineCourses'] / (temp['LearningCategoryUniversity'] + 1)



target_feature = 'Ratio_Online_University'



nations = []

target_numbers = []

for nation in temp['Country'].unique():

    if nation == "Other":

        continue

    nations.append(nation)

    target_numbers.append(np.round(temp.loc[temp['Country'] == nation][target_feature].mean(), 2))

    

results = pd.DataFrame({'nation': nations, target_feature: target_numbers})



results.loc[results['nation'] == "People 's Republic of China", target_feature] = (float(results.loc[results['nation'] == "People 's Republic of China", target_feature]) 

                                                                                      + float(results.loc[results['nation'] == "Republic of China", target_feature]))/2

results = results.drop(50).reset_index(drop=True)
title = "Where did you learn from - OnlineCourse or University?"

bar_title = "Ratio(OnlineCourse/University)"

plot_worldmap(title, bar_title)
results.sort_values(target_feature, ascending=False)[:10]
top_lists = results.sort_values(target_feature, ascending=False)[:3]['nation'].values

for nation in top_lists:

    fig = plt.figure(figsize=(8, 5))

    means = []

    for category in LearningCategory:

        means.append(temp.loc[temp['Country'] == nation][category].mean())

    temp_results = pd.Series(LearningCategory, means)

    plt.title(nation, fontsize=20)

    sns.barplot(temp_results.index, temp_results.values)
results.sort_values(target_feature, ascending=True)[:10]
bottom_lists = results.sort_values(target_feature, ascending=True)[:3]['nation'].values

for nation in bottom_lists:

    fig = plt.figure(figsize=(8, 5))

    means = []

    for category in LearningCategory:

        means.append(temp.loc[temp['Country'] == nation][category].mean())

    temp_results = pd.Series(LearningCategory, means)

    plt.title(nation, fontsize=20)

    sns.barplot(temp_results.index, temp_results.values)