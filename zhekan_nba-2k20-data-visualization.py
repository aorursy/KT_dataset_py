# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date, datetime
#visualization libraries
import plotly.graph_objs as go
import plotly.express as px
from plotly.figure_factory import create_annotated_heatmap

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/nba2k20-player-dataset/nba2k20-full.csv')
data.head(5)
data.info()
def prepare_data(data: pd.DataFrame):
    '''
        Preprocesses data
    '''
    def calculateAge(birthDate: str):
        '''
        calculates age of person, on given birth day
        '''
        datetime_object = datetime.strptime(birthDate, '%m/%d/%y')
        today = date.today() 
        age = today.year - datetime_object.year -  \
              ((today.month, today.day) < (datetime_object.month, datetime_object.day)) 
        return age 
    
    data['jersey'] = data['jersey'].apply(lambda x: int(x[1:]))
    data['age'] = data['b_day'].apply(calculateAge)
    data['height'] = data['height'].apply(lambda x: float(x.split('/')[1]))
    data['weight'] = data['weight'].apply(lambda x: float(x.split('/')[1].split(' ')[1]))
    data['salary'] = data['salary'].apply(lambda x: float(x[1:]))
    data['draft_round'].replace('Undrafted', 0, inplace = True)
    data['draft_round'] = data['draft_round'].apply(int)
    data['team'] = data['team'].fillna('No team')
    data['college'] = data['college'].fillna('No education')
    data.drop(['b_day', 'draft_peak'], axis = 1, inplace = True)
prepare_data(data)
data.info()
fig = px.scatter(data, x="rating", y="salary", trendline="ols", 
                 title = "Relationship between rating and salary", 
                 marginal_x = 'box', marginal_y = 'box')
fig.show()

fig = px.scatter(data, x="rating", y="salary", trendline="ols", 
                 color="draft_year", facet_col="draft_round", 
                 title = "Relationship between rating and salary by draft round with draft year distribution")
fig.show()
fig = px.histogram(data, x="salary", marginal = "box", title = "distribution of salary")
fig.show()
df = data['age'].value_counts().sort_index().reset_index()
df.columns = ['age', 'count']
fig = px.bar(data_frame = df, x='age', y='count', 
             color = 'age', title = 'distribution of age')
fig.show()
df = data.groupby('age')['salary'].mean().reset_index()
fig = px.bar(data_frame = df, x='age', y='salary', 
             color = 'age', title = 'Mean salary by age')
fig.show()
fig = px.scatter(data, x="age", y="salary", trendline="ols", 
                 title = "Relationship between age and salary")
fig.show()

val = [[round(elem, 2) for elem in row] for row in data.corr().values] 
fig = create_annotated_heatmap(z=val, 
                               x = data.corr().columns.tolist(), 
                               y = data.corr().columns.tolist(),  
                               colorscale = px.colors.diverging.balance, 
                               showscale = True)
layout = {'title' : 'Correlation map'}
fig.update_layout(layout)
fig.show()

step = 5
intervals = list(range(min(data['draft_year']), max(data['draft_year'] + step), step))
data['draft_year_intervals'] = pd.cut(data['draft_year'], intervals).apply(str)
fig = px.box(data, x = 'draft_year_intervals', y = 'salary', 
             title = 'Distribution of salary on draft year intervals')
fig.show()
most_common_country = data['country'].value_counts()[:5]
df = data[data['country'].isin(most_common_country.index.tolist())]
count_by_countries = df['country'].value_counts().reset_index()
count_by_countries.columns = ['country', 'count']
fig = px.bar(count_by_countries, x = 'country', y = 'count', 
             title = 'Count of players NBA on countries(top 5)')
fig.show()
salary_per_country = data[['country', 'salary']]
salary_per_country.loc[salary_per_country['country'] != 'USA', 'country'] = 'not USA'
fig = px.box(salary_per_country, x = 'country', y = 'salary', 
             title = 'Distribution of salary based on countries')
fig.show()
salary = data[['salary', 'position']]
salary = salary.groupby('position').describe()['salary'][['mean', 'count']]\
         .reset_index().sort_values(by='mean', ascending = False)
fig = px.bar(salary, x = 'position', y = 'mean', color = 'count', 
             title = 'Mean salary of players NBA per position with gradation on count')
fig.show()
rating_age_pos = data[['position', 'salary', 'age']]
rating_age_pos.loc[data['position'] == 'C-F', 'position'] = 'F-C'
rating_age_pos.loc[data['position'] == 'F-G', 'position'] = 'F'
rating_age_pos.loc[data['position'] == 'G-F', 'position'] = 'F'

fig = px.scatter(rating_age_pos, x="age", y="salary", trendline="ols", 
                 color = 'position', facet_col="position", facet_col_wrap=2,
                 title = "Relationship between age and salary with hue on position")

fig.show()
college = data['college'].value_counts()>10
college = college.loc[college]
df = data[data['college'].isin(college.index.tolist())]
salary_per_col = df[['salary', 'college']]
salary_per_col = salary_per_col.groupby('college').describe()['salary'][['mean', 'count']]\
                 .reset_index().sort_values(by="mean", ascending = False)
fig = px.bar(salary_per_col, x = 'college', y = 'mean', color = 'count', 
             title = 'Mean salary of players NBA per college(>10 players from college with No education)')
fig.show()
rating = df[['rating', 'college']]
rating = rating.groupby('college').mean().reset_index().sort_values(by="rating", ascending = False)
fig = px.bar(rating, x = 'college', y = 'rating', color = 'rating', 
             title = 'Mean rating of players NBA per college(>10 players from college  with No education)')
fig.show()
salary = data[['salary', 'team']]
salary = salary.groupby('team').describe()['salary'][['mean', 'count']]\
         .reset_index().sort_values(by='mean', ascending = False)
fig = px.bar(salary, x = 'team', y = 'mean', color = 'count')
fig.show()