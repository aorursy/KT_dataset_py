import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

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
target_feature = 'JobSatisfaction'

temp = df_multipleChoice[[target_feature, 'Country']].dropna().reset_index(drop=True)

temp = temp.loc[~(temp[target_feature] == 'I prefer not to share')]
respondents = pd.DataFrame(temp['Country'].value_counts()).reset_index()

respondents.columns = ['Country', 'number']

# Sum the numbers of "People 's Republic of China" and "Republic of China"

respondents = respondents.drop(49)

respondents.loc[12, 'number'] = respondents.loc[12,'number'] + 16



data = [ dict(

        type = 'choropleth',

        locations = respondents['Country'],

        z = respondents['number'],

        text = respondents['Country'],

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

            title = 'Number'),

      ) ]



layout = dict(

    title = "The number respondents of each nations on Q.\n'How satisfied are you with your current job?'",

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
temp[target_feature] = temp[target_feature].apply(lambda x: 10 if x == '10 - Highly Satisfied' else x)

temp[target_feature] = temp[target_feature].apply(lambda x: 10 if x == '1 - Highly Dissatisfied' else x).astype(int)



nations = []

satisfaction = []

for nation in temp['Country'].unique():

    if nation == "Other":

        continue

    nations.append(nation)

    satisfaction.append(np.round(temp.loc[temp['Country'] == nation][target_feature].mean(), 2))

    

results = pd.DataFrame({'nation': nations, target_feature: satisfaction})

results.loc[results['nation'] == "People 's Republic of China", target_feature] = (float(results.loc[results['nation'] == "People 's Republic of China", target_feature]) 

                                                                                      + float(results.loc[results['nation'] == "Republic of China", target_feature]))/2

results = results.drop(48).reset_index(drop=True)



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

            title = 'JobSatisfaction'),

      ) ]



layout = dict(

    title = "How satisfied are you with your current job? (1 - highly disatisfied, 10 - higly satisfied)",

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