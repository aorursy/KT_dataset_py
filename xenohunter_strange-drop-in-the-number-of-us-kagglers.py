import numpy as np

import pandas as pd



multiple_choice_2017 = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', low_memory=False, encoding='ISO-8859-1')

multiple_choice_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

multiple_choice_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory=False)



names_to_2019 = {

    'Hong Kong': 'Hong Kong (S.A.R.)',

    'Iran': 'Iran, Islamic Republic of...',

    'Republic of China': 'China',

    'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',

    'United States': 'United States of America',

    'Vietnam': 'Viet Nam'

}



names_to_simple = {

    'Hong Kong (S.A.R.)': 'Hong Kong',

    'Iran, Islamic Republic of...': 'Iran',

    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',

    'United States of America': 'United States',

    'Viet Nam': 'Vietnam'

}



countries_2017 = multiple_choice_2017.Country.dropna().value_counts().drop(labels=['Other'])

countries_2018 = multiple_choice_2018.Q3[1:].value_counts().drop(labels=['Other', 'I do not wish to disclose my location'])

countries_2019 = multiple_choice_2019.Q3[1:].value_counts().drop(labels=['Other'])



countries_2017['Republic of China'] += countries_2017['People \'s Republic of China']

countries_2017.drop(labels='People \'s Republic of China', inplace=True)



countries_2018['South Korea'] += countries_2018['Republic of Korea']

countries_2018.drop(labels='Republic of Korea', inplace=True)



countries_2019['South Korea'] += countries_2019['Republic of Korea']

countries_2019.drop(labels='Republic of Korea', inplace=True)



countries_2017.rename(names_to_2019, inplace=True)



top_by_total_number = pd.DataFrame({

    '2017': countries_2017,

    '2018': countries_2018,

    '2019': countries_2019,

    '(2017 + 2018 + 2019)': countries_2017 + countries_2018 + countries_2019

}).rename(names_to_simple).dropna().astype(int).sort_values('(2017 + 2018 + 2019)', ascending=False).head(5)



top_by_total_number
top_by_total_number.drop(columns=['(2017 + 2018 + 2019)']).T.plot();
df = pd.DataFrame({

    '2017': countries_2017 / np.sum(countries_2017) * 1000,

    '2018': countries_2018 / np.sum(countries_2018) * 1000,

    '2019': countries_2019 / np.sum(countries_2019) * 1000

}).rename(names_to_simple).dropna().astype(int)



normalized_by_yearly_totals = df.copy()

normalized_by_yearly_totals['(2017 + 2018 + 2019)'] = df['2017'] + df['2018'] + df['2019']

normalized_by_yearly_totals = normalized_by_yearly_totals.sort_values('(2017 + 2018 + 2019)', ascending=False).head(5)



normalized_by_yearly_totals
normalized_by_yearly_totals.drop(columns=['(2017 + 2018 + 2019)']).T.plot();
df = pd.DataFrame({

    '2017': countries_2017 / np.sum(countries_2017) * 1000,

    '2018': countries_2018 / np.sum(countries_2018) * 1000,

    '2019': countries_2019 / np.sum(countries_2019) * 1000

}).rename(names_to_simple).dropna().astype(int)



df['2017-to-2018'] = (df['2018'] - df['2017']) / df['2017']

df['2018-to-2019'] = (df['2019'] - df['2018']) / df['2018']



df['2017-to-2019'] = (df['2017-to-2018'] + df['2018-to-2019']) / 2



df.sort_values('2017-to-2019')