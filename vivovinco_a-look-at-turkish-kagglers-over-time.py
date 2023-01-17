!pip install -U vega_datasets notebook vega
!pip install pandas --upgrade
# importing libraries
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
import colorlover as cl
# loading data from different years
DIR = '/kaggle/input/kaggle-survey-2018/'
df_free_18 = pd.read_csv(DIR + 'freeFormResponses.csv', low_memory=False, header=[0,1])
df_choice_18 = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
# Format Dataframes
df_free_18.columns = ['_'.join(col) for col in df_free_18.columns]
df_choice_18.columns = ['_'.join(col) for col in df_choice_18.columns]

DIR = '/kaggle/input/kaggle-survey-2019/'
df_free_19 = pd.read_csv(DIR + 'other_text_responses.csv', low_memory=False)
df_choice_19 = pd.read_csv(DIR + 'multiple_choice_responses.csv', low_memory=False, encoding='latin-1', header=[0,1])
df_choice_19.columns = ['_'.join(col) for col in df_choice_19.columns]

DIR = '/kaggle/input/kaggle-survey-2017/'
df_free_17 = pd.read_csv(DIR + 'freeformResponses.csv', low_memory=False)
df_choice_17 = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, encoding='latin-1')
# processing data for visualizations

top_count = df_choice_19['Q3_In which country do you currently reside?'].value_counts().head(20).reset_index().rename(columns={'Q3_In which country do you currently reside?': 'count', 'index': 'Country'})
# taking only russian responders
df_choice_17 = df_choice_17.loc[df_choice_17['Country'] == 'Turkey']
df_choice_18 = df_choice_18.loc[df_choice_18['Q3_In which country do you currently reside?'] == 'Turkey']
df_choice_19 = df_choice_19.loc[df_choice_19['Q3_In which country do you currently reside?'] == 'Turkey']

def get_age(x: int):
    """
    Convert numerical age to categories.
    """
    if 18 <= x <= 21:
        return '18-21'
    elif 22 <= x <= 24:
        return '22-24'
    elif 25 <= x <= 29:
        return '25-29'
    elif 30 <= x <= 34:
        return '30-34'
    elif 35 <= x <= 39:
        return '35-39'
    elif 40 <= x <= 44:
        return '40-44'
    elif 45 <= x <= 49:
        return '45-49'
    elif 50 <= x <= 54:
        return '50-54'
    elif 55 <= x <= 59:
        return '55-59'
    elif 60 <= x <= 69:
        return '60-69'
    elif x >= 70:
        return '70+'
    
# create a new age column with the same name and unique values in all datasets
df_choice_17['Age_'] = df_choice_17['Age'].apply(lambda x: get_age(x))
df_choice_18['Age_'] = df_choice_18['Q2_What is your age (# years)?']
df_choice_18.loc[df_choice_18['Age_'].isin(['70-79', '80+']), 'Age_'] = '70+'
df_choice_19['Age_'] = df_choice_19['Q1_What is your age (# years)?']

# renaming columns so that it would be easier to work with them
df_choice_17 = df_choice_17.rename(columns={'GenderSelect': 'Gender', 'FormalEducation': 'Degree'})
df_choice_18 = df_choice_18.rename(columns={'Q1_What is your gender? - Selected Choice': 'Gender', 'Q9_What is your current yearly compensation (approximate $USD)?': 'Salary',
                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree'})
df_choice_19 = df_choice_19.rename(columns={'Q2_What is your gender? - Selected Choice': 'Gender', 'Q10_What is your current yearly compensation (approximate $USD)?': 'Salary',
                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree'})
df_choice_19['Degree'] = df_choice_19['Degree'].replace({'Masterâs degree': 'Master’s degree', 'Bachelorâs degree': 'Bachelor’s degree',
                                                         'Some college/university study without earning a bachelorâs degree': 'Some college/university study without earning a bachelor’s degree'})
df_choice_17['Degree'] = df_choice_17['Degree'].replace({"Master's degree": 'Master’s degree', "Bachelor's degree": 'Bachelor’s degree',
                                                         "Some college/university study without earning a bachelor's degree": 'Some college/university study without earning a bachelor’s degree',
                                                         "I did not complete any formal education past high school": "No formal education past high school"})
# Functions

# some of the code is taken from my old kernel: https://www.kaggle.com/artgor/russia-usa-india-and-other-countries

def plot_gender_vars(var1: str = '', title_name: str = ''):
    """
    Make separate count plots for genders over years.
    """
    colors = cl.scales['3']['qual']['Paired']
    names = {0: '2017', 1: '2018', 2: '2019'}
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), print_grid=False)
    # there are too little responders, who don't identify as Male/Female, I have decided that I can use the most common genders.
    for j, c in enumerate(['Male', 'Female']):
        data = []
        for i, df in enumerate([df_choice_17, df_choice_18, df_choice_19]):
            grouped = df.loc[(df['Gender'] == c), var1].value_counts().sort_index().reset_index()
            grouped['Age_'] = grouped['Age_'] / np.sum(grouped['Age_'])
            trace = go.Bar(
                x=grouped['index'],
                y=grouped.Age_,
                name=names[i],
                marker=dict(color=colors[i]),
                showlegend=True if j == 0 else False,
                legendgroup=i
            )
            fig.append_trace(trace, 1, j + 1)    

    fig['layout'].update(height=400, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Rate of Turkish Kagglers by {title_name} and Gender');
    return fig


def plot_var(var1: str = '', title_name: str = ''):
    """
    Plot one variable over years.
    """
    colors = cl.scales['3']['qual']['Paired']
    names = {0: '2017', 1: '2018', 2: '2019'}
    
    data = []
    for i, df in enumerate([df_choice_17, df_choice_18, df_choice_19]):
        grouped = df[var1].value_counts().sort_index().reset_index()
        grouped[var1] = grouped[var1] / np.sum(grouped[var1])
        trace = go.Bar(
            x=grouped['index'],
            y=grouped[var1],
            name=names[i],
            marker=dict(color=colors[i]),
            legendgroup=i
        )
        data.append(trace)
    layout = dict(height=400, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Turkish Kagglers by {title_name}');  
    fig = dict(data=data, layout=layout)
    return fig


def plot_choice_var(var: str = '', title_name: str = ''):
    """
    Plot a variable, in which responders could select several answers.
    """
    col_names = [col for col in df_choice_19.columns if f'{var}_Part' in col]
    data = []
    small_df = df_choice_19[col_names]
    text_values = [col.split('- ')[2] for col in col_names]
    counts = []
    for m, n in zip(col_names, text_values):
        if small_df[m].nunique() == 0:
            counts.append(0)
        else:
            counts.append(sum(small_df[m] == n))
            
    trace = go.Bar(
        x=text_values,
        y=counts,
        name='c',
        marker=dict(color='blue'),
        showlegend=False
    )
    data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=600, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Popular {title_name}');
    return fig
df_count = pd.DataFrame({'Year': [2017, 2018, 2019], 'Count': [df_choice_17.shape[0], df_choice_18.shape[0], df_choice_19.shape[0]]})
top_count = top_count.sort_values('count')

fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Bar(y=top_count['Country'], x=top_count['count'], orientation='h', name='Number of respondents by country in 2019'), row=1, col=1)
fig.add_trace(go.Bar(x=df_count['Year'], y=df_count['Count'], name='Number of Turkish responders by year'), row=1, col=2)

fig['layout'].update(height=600, width=1000,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)');
iplot(fig);
fig = plot_gender_vars(var1='Age_', title_name='Age')
iplot(fig);
fig = plot_var(var1='Degree', title_name='Degree')
iplot(fig);
fig = plot_choice_var(var='Q12', title_name='Resources')
iplot(fig);
fig = plot_choice_var(var='Q18', title_name='Languages')
iplot(fig);
fig = plot_choice_var(var='Q9', title_name='Additional Activities')
iplot(fig);
fig = plot_choice_var(var='Q27', title_name='NLP Tools')
iplot(fig);
fig = plot_choice_var(var='Q28', title_name='Libraries')
iplot(fig);
fig = plot_choice_var(var='Q25', title_name='ML Tools')
iplot(fig);