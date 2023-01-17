# importing libraries

import numpy as np

import random

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

from wordcloud import WordCloud

import pycountry



#from IPython.display import HTML

#HTML(cl.to_html( cl.scales ))
# credits to @artgor kernel: https://www.kaggle.com/artgor/a-look-at-russian-kagglers-over-time



# read data 

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





# renaming columns so that it would be easier to work with them

df_choice_17 = df_choice_17.rename(columns={'GenderSelect': 'Gender', 'FormalEducation': 'Degree', 'CurrentJobTitleSelect': 'Job'})

df_choice_18 = df_choice_18.rename(columns={'Q1_What is your gender? - Selected Choice': 'Gender', 'Q9_What is your current yearly compensation (approximate $USD)?': 'Salary',

                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree', 

                                            'Q3_In which country do you currently reside?': 'Country',

                                            'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'Job'})

df_choice_19 = df_choice_19.rename(columns={'Q2_What is your gender? - Selected Choice': 'Gender', 'Q10_What is your current yearly compensation (approximate $USD)?': 'Salary',

                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree',

                                            'Q3_In which country do you currently reside?': 'Country',

                                            'Q5_Select the title most similar to your current role (or most recent title if retired): - Selected Choice': 'Job'})

df_choice_19['Degree'] = df_choice_19['Degree'].replace({'Masterâs degree': 'Master’s degree', 'Bachelorâs degree': 'Bachelor’s degree',

                                                         'Some college/university study without earning a bachelorâs degree': 'Some college study (no bachelor’s degree)'})

df_choice_18['Degree'] = df_choice_18['Degree'].replace({"Some college/university study without earning a bachelor’s degree": 'Some college study (no bachelor’s degree)',

                                                         }) # Added to reduce place at graphs

df_choice_17['Degree'] = df_choice_17['Degree'].replace({"Master's degree": 'Master’s degree', "Bachelor's degree": 'Bachelor’s degree',

                                                         "Some college/university study without earning a bachelor's degree": 'Some college study (no bachelor’s degree)',

                                                         "I did not complete any formal education past high school": "No formal education past high school"})



df_choice_17['Job'] = df_choice_17['Job'].replace({"Software Developer/Software Engineer": "Software Engineer",

                                                   "Researcher": "Research Scientist", "Scientist/Researcher": "Research Scientist"})



df_free_19 = df_free_19.rename(columns={'Q13_OTHER_TEXT': 'Edu_Source'})

df_free_18 = df_free_18.rename(columns={'Q36_OTHER_TEXT_On which online platforms have you begun or completed data science courses? (Select all that apply) - Other - Text': 'Edu_Source'})



df_free_19['Edu_Source'] = df_free_19['Edu_Source'].replace({"YouTube": "youtube", "Youtube": "youtube", "stepik.org":"Stepik"})

df_free_18['Edu_Source'] = df_free_18['Edu_Source'].replace({"Stepic": "Stepik", "YouTube": "youtube", "Youtube": "youtube", "Lynda.com":"Lynda", "CodeAcademy":"Codecademy"})





# create a new age column with the same name and unique values in all datasets

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



# changing salary values to custom bins

#0-20

df_choice_19.loc[df_choice_19['Salary'].isin(['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', '5,000-7,499', '7,500-9,999', '10,000-14,999', '15,000-19,999']), 'Salary'] = '0-20,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['0-10,000', '10-20,000']), 'Salary'] = '0-20,000'

#20-40

df_choice_19.loc[df_choice_19['Salary'].isin(['20,000-24,999', '25,000-29,999', '30,000-39,999']), 'Salary'] = '20-40,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['20-30,000', '30-40,000']), 'Salary'] = '20-40,000'

#40-70

df_choice_19.loc[df_choice_19['Salary'].isin(['40,000-49,999', '50,000-59,999', '60,000-69,999']), 'Salary'] = '40-70,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['40-50,000', '50-60,000', '60-70,000']), 'Salary'] = '40-70,000'

#70-100

df_choice_19.loc[df_choice_19['Salary'].isin(['70,000-79,999', '80,000-89,999', '90,000-99,999']), 'Salary'] = '70-100,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['70-80,000', '80-90,000', '90-100,000']), 'Salary'] = '70-100,000'

#100-150

df_choice_19.loc[df_choice_19['Salary'].isin(['100,000-124,999', '125,000-149,999']), 'Salary'] = '100-150,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['100-125,000', '125-150,000']), 'Salary'] = '100-150,000'

#150 - 300

df_choice_19.loc[df_choice_19['Salary'].isin(['150,000-199,999', '200,000-249,999', '250,000-299,999']), 'Salary'] = '150-300,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['150-200,000', '200-250,000', '250-300,000']), 'Salary'] = '150-300,000'

#>300

df_choice_19.loc[df_choice_19['Salary'].isin(['300,000-500,000', '> $500,000']), 'Salary'] = '> $300,000'

df_choice_18.loc[df_choice_18['Salary'].isin(['400-500,000', '300-400,000', '500,000+']), 'Salary'] = '> $300,000'







# My own prepocessing functions



# functions to assign numerical value for degree

def set_value(row_number, assigned_value): 

    return assigned_value[row_number] 





def create_degree_level(df, col):

    data = df.copy()

    data[[col]] =  data[[col]].fillna('NaN')

    degree_dictionary ={'NaN': 0, 

                        'I prefer not to answer': 1,

                        'No formal education past high school' : 2,

                        'Some college study (no bachelor’s degree)' : 3,

                        'Professional degree': 4,

                        'Bachelor’s degree' : 5,

                        'Master’s degree': 6, 

                        'Doctoral degree': 7

                       } 

    data['Degree_level'] = data[col].apply(set_value, args=(degree_dictionary, ))

    data = data.sort_values(by=['Degree_level']).reset_index()

    return data



# filter people with no academic or professional degree

def filter_no_degree(data): 

    df = data.copy()

    df = create_degree_level(df, "Degree")

    df = df.query("Degree_level>3")

    return df



def filter_students(data): 

    df = data.copy()

    return df.query("Job!='Student' & Job!='Not employed'")



def filter_no_salary(df_18):

    return df_18.query("Salary!='I do not wish to disclose my approximate yearly compensation'")



# visualisations funcs

def plot_var(df_list, var1: str = '', title_name: str = '', degree_sort: bool = False):

    """

    Plot one variable over years.

    """

    #colors = cl.scales['3']['qual']['Accent'] #Paired

    colors = cl.scales['3']['seq']['GnBu']

    names = {0: '2017', 1: '2018', 2: '2019'}

    

    data = []

    for i, df in enumerate(df_list):

        grouped = df[var1].value_counts().sort_index().reset_index()

        if degree_sort:

            grouped = create_degree_level(grouped, "index")

        grouped[var1] = round(grouped[var1] / np.sum(grouped[var1]), 3)

        trace = go.Bar(

            x=grouped['index'],

            y=grouped[var1],

            name=names[i],

            marker=dict(color=colors[i]),

            legendgroup=i

        )

        data.append(trace)

    layout = dict(height=400, width=800, title=f'Kagglers {title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(showticklabels=False, title="% of respondents"));  

    fig = dict(data=data, layout=layout)

    return fig



def plot_var_h(df_list, var1: str = '', title_name: str = '', degree_sort: bool = False):

    """

    Plot one variable over years.

    """

    #colors = cl.scales['3']['qual']['Accent'] #Paired

    colors = cl.scales['3']['seq']['GnBu']

    names = {0: '2017', 1: '2018', 2: '2019'}

    

    data = []

    for i, df in enumerate(df_list):

        grouped = df[var1].value_counts().sort_index().reset_index()

        if degree_sort:

            grouped = create_degree_level(grouped, "index")

        grouped[var1] = round(grouped[var1] / np.sum(grouped[var1]), 3)

        trace = go.Bar(

            orientation='h',

            y=grouped['index'],

            x=grouped[var1],

            name=names[i],

            marker=dict(color=colors[i]),

            legendgroup=i

        )

        data.append(trace)

    layout = dict(height=800, width=800, title=f'Kagglers {title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', xaxis=dict(title="% of respondents", showticklabels=False));  

    fig = dict(data=data, layout=layout)

    return fig





def plot_subplots_of_degree_by_gender(df_list, var1: str = '', title_name: str = '', degree_sort: bool = False):

    #colors = cl.scales['3']['qual']['Accent']

    colors = cl.scales['3']['seq']['GnBu']

    names = {0: '2017', 1: '2018', 2: '2019'}

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), print_grid=False)

    # there are too little responders, who don't identify as Male/Female, I have decided that I can use the most common genders.

    for j, c in enumerate(['Male', 'Female']):

        data = []

        for i, df in enumerate(df_list):

            grouped = df.loc[(df['Gender'] == c), var1].value_counts().sort_index().reset_index()

            if degree_sort:

                grouped = create_degree_level(grouped, "index")

            grouped[var1] = round(grouped[var1] / np.sum(grouped[var1]), 3)

            trace = go.Bar(

                x=grouped['index'],

                y=grouped.Degree,

                name=names[i],

                marker=dict(color=colors[i]),

                showlegend=True if j == 0 else False,

                legendgroup=i

            )

            fig.append_trace(trace, 1, j + 1)

    



    fig['layout'].update(height=400, width=800, title=f'Rate of kagglers by {title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(title="% of respondents"));

    for c in range(1,3): 

        fig.update_yaxes(showticklabels=False, row=1, col=c)

    return fig





def plot_subplots_of_age_by_degree(df_list, var1: str = '', title_name: str = '', degree_sort: bool = False):

    #colors = cl.scales['3']['qual']['Accent']

    colors = cl.scales['3']['seq']['GnBu']

    names = {0: '2017', 1: '2018', 2: '2019'}

    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Professional degree', 'Bachelor"s degree', 'Masters"s degree', "Doctoral degree"), print_grid=False)

    # there are too little responders, who don't identify as Male/Female, I have decided that I can use the most common genders.

    for j, c in enumerate([4, 5, 6, 7]):

        data = []

        for i, df in enumerate(df_list):

            df = create_degree_level(df, "Degree")

            grouped = df.loc[(df['Degree_level'] == c), var1].value_counts().sort_index().reset_index()

            grouped[var1] = round(grouped[var1] / np.sum(grouped[var1]), 3)

            trace = go.Bar(

                x=grouped['index'],

                y=grouped.Age_,

                name=names[i],

                marker=dict(color=colors[i]),

                showlegend=True if j == 0 else False,

                legendgroup=i

            )

            if j<2:

                fig.append_trace(trace, 1, j + 1)   

            else:

                fig.append_trace(trace, 2, j - 1)

        

    fig['layout'].update(height=800, width=1000, title=f'Rate of kagglers by {title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(title="% of respondents"));

    for r in range(1,3):

        for c in range(1,3): 

            if c==1: 

                fig.update_yaxes(showticklabels=False, row=r, col=c, title="% of respondents")

            else: 

                fig.update_yaxes(showticklabels=False, row=r, col=c)

                

    return fig



def plot_subplots_of_sal_by_degree(df_list, var1: str = '', title_name: str = '', degree_sort: bool = False):

    colors = cl.scales['3']['seq']['GnBu'][1:]

    names = {0: '2018', 1: '2019'}

    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Professional degree', 'Bachelor"s degree', 'Masters"s degree', "Doctoral degree"), print_grid=False)

    # there are too little responders, who don't identify as Male/Female, I have decided that I can use the most common genders.

    for j, c in enumerate([4, 5, 6, 7]):

        data = []

        for i, df in enumerate(df_list):

            df = create_degree_level(df, "Degree")

            grouped = df.loc[(df['Degree_level'] == c), var1].value_counts().sort_index().reset_index()

            grouped[var1] = round(grouped[var1] / np.sum(grouped[var1]), 3)

            map_dict = {'0-20,000': 0,

                        '20-40,000': 1,

                        '40-70,000': 2,

                        '70-100,000' : 3,

                        '100-150,000': 4,

                        '150-300,000': 5,

                        '> $300,000': 6

                       }

            grouped['sorting'] = grouped['index'].apply(lambda x: map_dict[x])

            grouped = grouped.sort_values('sorting', ascending=True)

            trace = go.Bar(

                x=grouped['index'],

                y=grouped[var1],

                name=names[i],

                marker=dict(color=colors[i]),

                showlegend=True if j == 0 else False,

                legendgroup=i

            )

            if j<2:

                fig.append_trace(trace, 1, j + 1)   

            else:

                fig.append_trace(trace, 2, j - 1)

        

    fig['layout'].update(height=800, width=1000, title=f'Rate of kagglers by {title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(title="% of respondents"));

    for r in range(1,3):

        for c in range(1,3): 

            if c==1: 

                fig.update_yaxes(showticklabels=False, row=r, col=c, title="% of respondents")

            else: 

                fig.update_yaxes(showticklabels=False, row=r, col=c)

                

    return fig
dataframes = [df_choice_17, df_choice_18, df_choice_19]

fig = plot_var(dataframes, 'Degree', 'Education degree over years', True)

iplot(fig);
dataframes = [filter_no_degree(df_choice_17), filter_no_degree(df_choice_18), filter_no_degree(df_choice_19)]

fig = plot_subplots_of_degree_by_gender(dataframes, 'Degree', 'Education degree by Gender', True)

iplot(fig);
dataframes = [filter_no_degree(df_choice_17), filter_no_degree(df_choice_18), filter_no_degree(df_choice_19)]

fig = plot_subplots_of_age_by_degree(dataframes, 'Age_', 'Education degree by Age', True)

iplot(fig);
def grey_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)





def plot_wordcloudline(df_list, max_words_n: int = 20, year: str = ''):

     

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[26, 8])

    wordcloud1 = WordCloud( background_color='black',

                            width=600,

                            height=400,

                            max_words=max_words_n).generate(" ".join(df_list[0]['Job'].dropna()))

    

    ax1.imshow(wordcloud1.recolor(color_func=grey_color_func, random_state=3))

    ax1.axis('off')

    ax1.set_title(f'Top-5 Bachelor degree roles in {year}',fontsize=20);



    wordcloud2 = WordCloud( background_color='black',

                            width=600,

                            height=400,

                            max_words=max_words_n).generate(" ".join(df_list[1]['Job'].dropna()))

    ax2.imshow(wordcloud2.recolor(color_func=grey_color_func, random_state=3))

    ax2.axis('off')

    ax2.set_title(f'Top-5 Master degree roles in {year}',fontsize=20);



    wordcloud3 = WordCloud( background_color='black',

                            width=600,

                            height=400,

                            max_words=max_words_n).generate(" ".join(df_list[2]['Job'].dropna()))

    ax3.imshow(wordcloud3.recolor(color_func=grey_color_func, random_state=3))

    ax3.axis('off')

    ax3.set_title(f'Top-5 Doctoral degree roles in {year}',fontsize=20);

    

    

def choose_df_by_year(year: str = ''): 

    if year=='2017':

        df = df_choice_17

    elif year=='2018':

        df = df_choice_18

    elif year=='2019': 

        df = df_choice_19

    else: 

        print("ERROR: incorrect year")

        

    return df

        

    

for n, y in enumerate(['2017', '2018', '2019']):

    df = filter_students(filter_no_degree(choose_df_by_year(y)))

    dataframes = [df.query('Degree_level==5'), df.query('Degree_level==6'), df.query('Degree_level==7')]

    plot_wordcloudline(dataframes, 5, y)
dataframes = [df_choice_17, df_choice_18, df_choice_19]

dataframes = [filter_students(i) for i in dataframes]

fig = plot_var_h(dataframes, 'Job', 'Job roles by Education degree over years', False)

iplot(fig);
dataframes = [filter_no_degree(filter_no_salary(df_choice_18)), filter_no_degree(df_choice_19)]

fig = plot_subplots_of_sal_by_degree(dataframes, 'Salary', 'Salary level by degree for last two years', True)

iplot(fig);
# credit to @Parul Pandey and her great kernel: https://www.kaggle.com/parulpandey/geek-girls-rising-myth-or-reality/notebook#2.-Country



def get_name(code):

    '''

    Translate code to name of the country

    '''

    try:

        name = pycountry.countries.get(alpha_3=code).name

    except:

        name=code

    return name





am = df_choice_19['Country'].value_counts()



def find_n_by_country(country_str: str=''):

    return am[country_str]





df_choice_19['n_of_people_in_country'] = df_choice_19['Country'].apply(lambda x: find_n_by_country(x))





def plot_world_map(df,degree: str='', title: str = ''): 

    """Function return fig, which should be plotted by iplot(fig);

    Arguments:

    df - pd Data Frame by which we'll plot the graph

    degree - `doctoral`, `master` or `bachelor` degree (string)

    title - title of the figure (string)

    """

    if degree=='doctoral': 

        query_t = "Degree_level==7"

    elif degree=='master': 

        query_t = "Degree_level==6"

    elif degree=='bachelor': 

        query_t = "Degree_level==5"

    else :

        print("ERROR! There is no support for that degree. ")    

    

    country_number = pd.DataFrame(filter_no_degree(df).query(query_t).groupby('Country').count()['Degree']/df_choice_19.groupby('Country').count()['n_of_people_in_country'])

    country_number['country'] = country_number.index

    country_number.columns = ['number', 'country']

    country_number['country'] = country_number['country'].apply(lambda c: get_name(c))

    worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',

                 z = country_number['number'], colorscale = "Viridis", autocolorscale=True,reversescale = False, 

                 marker = dict(line = dict( width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of respondents'))]

    layout = dict(title = title, geo = dict(showframe = False, showcoastlines = True, 

                                                              projection = dict(type = 'Mercator')))  

    

    fig = dict(data = worldmap, layout = layout)

    return fig



fig = plot_world_map(df_choice_19, 'doctoral', 'The Percentage of Doctoral Degrees by Country in 2019')

iplot(fig, validate=False); 
fig = plot_world_map(df_choice_19, 'master', 'The Percentage of Master Degrees by Country in 2019')

iplot(fig, validate=False); 
fig = plot_world_map(df_choice_19, 'bachelor', 'The Percentage of Bachelor Degrees by Country in 2019')

iplot(fig, validate=False); 
# visualisations funcs

def plot_single_var(df_list, var1: str = '', title_name: str = ''):

    """

    Plot one variable over years.

    """

    colors = cl.scales['3']['seq']['YlOrRd'][1:]

    names = {0: '2019'}

    

    data = []

    for i, df in enumerate(df_list):

        grouped = df[var1].dropna().value_counts()[:20].sort_index().reset_index()



        grouped[var1] = grouped[var1]

        trace = go.Bar(

            x=grouped['index'],

            y=grouped[var1],

            name=names[i],

            marker=dict(color=colors[i]),

            legendgroup=i

        )

        data.append(trace)

    layout = dict(height=400, width=800, title=f'{title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(showticklabels=False, title="# of respondents"));  

    fig = dict(data=data, layout=layout)

    return fig





df_list = [df_free_19]

fig = plot_single_var(df_list, 'Edu_Source','Top-20 Others Educational Platforms in 2019')

iplot(fig);
# visualisations funcs

def plot_single_var_over_years(df_list, var1: str = '', title_name: str = ''):

    """

    Plot one variable over years.

    """

    colors = cl.scales['3']['seq']['YlOrRd']

    names = {0: '2018', 1:'2019'}

    

    data = []

    for i, df in enumerate(df_list):

        grouped = df[var1].dropna().value_counts()[:10].sort_index().reset_index()

        grouped[var1] = grouped[var1]

        trace = go.Bar(

            x=grouped['index'],

            y=grouped[var1],

            name=names[i],

            marker=dict(color=colors[i]),

            legendgroup=i

        )

        data.append(trace)

    layout = dict(height=400, width=800, title=f'{title_name}', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', yaxis=dict(showticklabels=False, title="# of respondents"));  

    fig = dict(data=data, layout=layout)

    return fig



df_list = [df_free_18, df_free_19]

fig = plot_single_var_over_years(df_list, 'Edu_Source','Top-10 Others Educational Platforms over years')

iplot(fig);