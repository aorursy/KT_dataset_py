# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as offline

import plotly.graph_objs as go



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



MAX_SEASON = 18



def read_south_park_data(file_name):

    return pd.read_csv(file_name)



def word_count_by_season_and_episode(df, word, character="", seasons=[], episodes=[]):

    '''

    '''

    count = 0

    character = character.lower()

    word = word.lower()



    for idx, row in df.iterrows():



        # If seasons and episodes are both unspecified,

        # calculate word frequency overall seasons and episodes.

        if seasons == [] and episodes == []:

            if character == "":

                if word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1



        # If seasons are specified but episodes are not,

        # calculate word frequency by season.

        elif seasons != [] and episodes == []:

            if character == "":

                if any(season == row['Season'] for season in seasons) and word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if any(season == row['Season'] for season in seasons) and word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1



        # If seasons and episode are specified,

        # calculate word frequency over specific

        # seasons and episodes.

        else:

            if character == "":

                if any(season == row['Season'] for season in seasons) and any(episode == row['Episode'] for episode in episodes) and word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if any(season == row['Season'] for season in seasons) and any(episode == row['Episode'] for episode in episodes) and word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

    return count



df = read_south_park_data('../input/All-seasons.csv')
import plotly.offline as offline

import plotly.graph_objs as go





def plot_swear_count_season_frequency():



    ass = [152, 207, 269, 189, 147, 164, 136, 123, 99, 125, 122, 78, 96, 61, 117, 110, 48, 62]

    damn = [100, 115, 104, 79, 77, 79, 64, 54, 53, 50, 38, 45, 37, 32, 28, 26, 17, 23] # damn or dammit

    fuck = [19, 48, 29, 14, 15, 29, 29, 43, 15, 18, 32, 53, 106, 162, 133, 100, 71, 85]

    shit = [5, 21, 10, 7, 168, 25, 16, 7, 7, 4, 35, 10, 18, 21, 116, 30, 30, 44]



    trace1 = go.Scatter(

        x=['S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-10', 'S-11', 'S-12', 'S-13', 'S-14', 'S-15', 'S-16', 'S-17', 'S-18'],

        y=ass,

        mode='lines+markers',

        name="'Ass'",

        hoverinfo='name',

        line=dict(

            shape='linear'

        )

    )

    trace2 = go.Scatter(

        x=['S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-10', 'S-11', 'S-12', 'S-13', 'S-14', 'S-15', 'S-16', 'S-17', 'S-18'],

        y=damn,

        mode='lines+markers',

        name="'Damn/Dammit'",

        hoverinfo='name',

        line=dict(

            shape='linear'

        )

    )

    trace3 = go.Scatter(

        x=['S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-10', 'S-11', 'S-12', 'S-13', 'S-14', 'S-15', 'S-16', 'S-17', 'S-18'],

        y=fuck,

        mode='lines+markers',

        name="'Fuck'",

        hoverinfo='name',

        line=dict(

            shape='linear'

        )

    )

    trace4 = go.Scatter(

        x=['S-1', 'S-2', 'S-3', 'S-4', 'S-5', 'S-6', 'S-7', 'S-8', 'S-9', 'S-10', 'S-11', 'S-12', 'S-13', 'S-14', 'S-15', 'S-16', 'S-17', 'S-18'],

        y=shit,

        mode='lines+markers',

        name="'Shit'",

        hoverinfo='name',

        line=dict(

            shape='linear'

        )

    )

    data = [trace1, trace2, trace3, trace4]

    layout = dict(

        title="Occurrence of Swear Word by Season",

        xaxis = dict(title = 'South Park Season'),

        yaxis = dict(title = 'Swear Word Frequency'),        

        legend=dict(

            x=1,

            y=0.5,

            font=dict(

                size=16

            )

        ),

    )    

    fig = go.Figure(data=data, layout=layout)

    offline.plot(fig)

 

# Uncomment to run

#plot_swear_count_season_frequency()
import pandas as pd



MAX_SEASONS = 18



def word_count_by_season_and_episode(df, word, character="", seasons=[], episodes=[]):

    '''

    '''

    count = 0

    character = character.lower()

    word = word.lower()



    for idx, row in df.iterrows():



        # If seasons and episodes are both unspecified,

        # calculate word frequency overall seasons and episodes.

        if seasons == [] and episodes == []:

            if character == "":

                if word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1



        # If seasons are specified but episodes are not,

        # calculate word frequency by season.

        elif seasons != [] and episodes == []:

            if character == "":

                if any(season == row['Season'] for season in seasons) and word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if any(season == row['Season'] for season in seasons) and word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1



        # If seasons and episode are specified,

        # calculate word frequency over specific

        # seasons and episodes.

        else:

            if character == "":

                if any(season == row['Season'] for season in seasons) and any(episode == row['Episode'] for episode in episodes) and word in row['Line'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

            else:

                if any(season == row['Season'] for season in seasons) and any(episode == row['Episode'] for episode in episodes) and word in row['Line'].lower() and character in row['Character'].lower():

                    split_line = row['Line'].split()

                    for w in split_line:

                        if word in w.lower():

                            count += 1

    return count



def read_south_park_data(file_name):

    return pd.read_csv(file_name)



df = read_south_park_data('../input/All-seasons.csv')



# Uncomment to run:

#swear_word = "ass"

#season_swear_count = []

#for season in range(1,MAX_SEASONS+1):

#    for idx, row in df.iterrows():

#        season_swear_count.append(word_count_by_season_and_episode(df, word = swear_word, seasons=[str(season)]))

#print(season_swear_count)
import plotly.offline as offline

import plotly.graph_objs as go





def plot_total_lines_of_dialog():

    x = ['Cartman', 'Stan', 'Kyle', 'Kenny']

    y = [9911, 7900, 7419, 923]



    trace0 = go.Bar(

                x=["Cartman"],y=[9911],

                name="Cartman",

                marker=dict(

                    color='rgb(90,192,214)',

                    line=dict(

                        color='rgb(8,48,107)',

                        width=1.5),

                ),

                opacity=0.8

            )



    trace1 = go.Bar(

                x=["Stan"],y=[7900],

                name="Stan",

                marker=dict(

                    color='rgb(81,97,172)',

                    line=dict(

                        color='rgb(8,48,107)',

                        width=1.5),

                ),

                opacity=0.8

            )



    trace2 = go.Bar(

                x=["Kyle"],y=[7419],

                name="Kyle",

                marker=dict(

                    color='rgb(108,192,106)',

                    line=dict(

                        color='rgb(8,48,107)',

                        width=1.5),

                ),

                opacity=0.8

            )



    trace3 = go.Bar(

                x=["Kenny"],y=[923],

                name="Kenny",

                marker=dict(

                    color='rgb(244,115,32)',

                    line=dict(

                        color='rgb(8,48,107)',

                        width=1.5),

                ),

                opacity=0.8

            )



    layout = go.Layout(

        title='Total Lines of Dialog Per Character',

        xaxis=dict(

            title="South Park Character",

        ),

        yaxis=dict(

            title='Total Lines of Dialog',

        ),    

        annotations=[

            dict(x=xi,y=yi,

                 text=str(yi),

                 xanchor='center',

                 yanchor='bottom',

                 showarrow=False,

            ) for xi, yi in zip(x, y)]

    )



    data = [trace0, trace1, trace2, trace3]



    fig = go.Figure(data=data, layout=layout)

    offline.plot(fig)    

 

# Uncomment to run

#plot_total_lines_of_dialog()