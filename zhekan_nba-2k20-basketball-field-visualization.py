# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date, datetime

#visualization libraries

import plotly.graph_objs as go

import plotly.express as px



%matplotlib inline

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

        age = today.year - datetime_object.year -  ((today.month, today.day) < (datetime_object.month, datetime_object.day)) 

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
def draw_plotly_court(fig, fig_width=600, margins=10):

    '''

    Plots basketball field

    '''

    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, 

                    end_angle=2 * np.pi, N=200, closed=False):

        t = np.linspace(start_angle, end_angle, N)

        x = x_center + a * np.cos(t)

        y = y_center + b * np.sin(t)

        path = f'M {x[0]}, {y[0]}'

        for k in range(1, len(t)):

            path += f'L{x[k]}, {y[k]}'

        if closed:

            path += ' Z'

        return path



    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)

    fig.update_layout(width=fig_width, height=fig_height)



    # Set axes ranges

    fig.update_xaxes(range=[-250 - margins, 250 + margins])

    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])



    threept_break_y = 89.47765084

    three_line_col = "#000000"

    main_line_col = "#000000"

    three_second_zone = "#555b6e"



    fig.update_layout(

        # Line Horizontal

        margin=dict(l=20, r=20, t=20, b=20),

        paper_bgcolor="white",

        plot_bgcolor="white",

        yaxis=dict(

            scaleanchor="x",

            scaleratio=1,

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False,

            fixedrange=True,

        ),

        xaxis=dict(

            showgrid=False,

            zeroline=False,

            showline=False,

            ticks='',

            showticklabels=False,

            fixedrange=True,

        ),

        shapes=[

            

            dict(

                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,

                line=dict(color=main_line_col, width=1),

                fillcolor='#9FC490',

                layer='below'

            ),

            dict(

                type="line", x0=-85, y0=-52.5, x1=-60, y1=137.5,

                line=dict(color=main_line_col, width=1),

                layer='below'

            ),

            dict(

                type="line", x0=85, y0=-52.5, x1=60, y1=137.5,

                line=dict(color=main_line_col, width=1),

                layer='below'

            ),

            dict(

                type="path",

                path=" M -60,137.5 L60,137.5 L85,-52 L-85,-52, L-60, 137.5",

                fillcolor=three_second_zone,

                line_color=main_line_col,

                opacity = 0.8,

                layer='below'

            ),

            dict(type="path",

                 path=ellipse_arc(y_center=137.5, a=60, b=60, start_angle=0, end_angle=-np.pi),

                 line=dict(color='#FF934F', width=1, dash='dot') ),

            dict(type="path",

                 path=ellipse_arc(y_center=137.5, a=60, b=60, start_angle=0, end_angle=np.pi),

                 line=dict(color=main_line_col, width=1), layer='below', fillcolor='#dddddd',),

            dict(

                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,

                line=dict(color=main_line_col, width=1),

                layer='below'

            ),



            dict(

                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,

                line=dict(color="#ec7607", width=1),

                fillcolor='#ec7607',

            ),

            dict(

                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",

                line=dict(color="#ec7607", width=1),

            ),

            dict(

                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,

                line=dict(color="#ec7607", width=1),

            ),



            dict(type="path",

                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),

                 line=dict(color=main_line_col, width=1)),

            dict(type="path",

                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, 

                                  end_angle=np.pi - 0.386283101),

                 line=dict(color=main_line_col, width=1), layer='below'),

            dict(

                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,

                line=dict(color=three_line_col, width=1), layer='below'

            ),



            dict(

                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=250, y0=227.5, x1=220, y1=227.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-86, y0=19.5, x1=-76, y1=17.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-85, y0=29.5, x1=-75, y1=27.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-80, y0=59.5, x1=-70, y1=57.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=-77, y0=89.5, x1=-67, y1=87.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=86, y0=19.5, x1=76, y1=17.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=85, y0=29.5, x1=75, y1=27.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=80, y0=59.5, x1=70, y1=57.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),

            dict(

                type="line", x0=77, y0=89.5, x1=67, y1=87.5,

                line=dict(color=main_line_col, width=1), layer='below'

            ),



            dict(type="path",

                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),

                 line=dict(color=main_line_col, width=1), layer='below'),

            



        ]

    )

    return True
def plot_players(data: pd.DataFrame, fig: go.Figure):

    '''

    Plots players on basketball field

    '''

    if len(data) == 5:

        data = data.set_index('position').T.to_dict()

        x = [-110, 40, 0, -160, 160]

        y = [70, 45, 260, 200, 150]

        positions = ['C-F', 'C', 'G', 'G-F', 'F']

        font_color = "#000"

        marker_color = "#DC602E"

        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                mode='markers+text',

                marker=dict(size=[30, 30, 30, 30, 30], color=marker_color),

                text=positions,

                hoverinfo = 'text',

                hovertext = [data[positions[i]]['full_name'] + ', ' 

                             + str(data[positions[i]]['rating']) 

                             for i in range(len(data))]

            )

        )



        for i in range(len(data)):

            fig.add_annotation(x=x[i], y=y[i]+25,

                              text=data[positions[i]]['full_name'] + ', ' 

                               + str(data[positions[i]]['rating']))

           

        fig.update_annotations(dict(

                    xref="x",

                    yref="y",

                    showarrow=False,

                    font=dict(

                        family="sans serif",

                        size=16,

                        color=font_color

                    )

        ))



    fig.update_layout(showlegend=False)

    return fig



data.loc[data['position'] == 'F-G', 'position'] = 'G-F'

data.loc[data['position'] == 'F-C', 'position'] = 'C-F'
df = data[['rating', 'team', 'position', 'country', 'full_name']]

df = df[df['country'] == 'USA']

if len(df.groupby('position').rating.agg('idxmax')) < 5:

    print('Not enough players')

else:

    df = df.loc[df.groupby('position').rating.agg('idxmax')]

    fig = go.Figure()

    draw_plotly_court(fig)

    fig = plot_players(df, fig)

    layout = {'title' : 'top USA team'}

    fig.update_layout(layout)

    fig.show()

df = data[['rating', 'team', 'position', 'country', 'full_name']]

df = df[df['country'] == 'Canada']

if len(df.groupby('position').rating.agg('idxmax')) < 5:

    print('Not enough players')

else:

    df = df.loc[df.groupby('position').rating.agg('idxmax')]

    fig = go.Figure()

    draw_plotly_court(fig)

    fig = plot_players(df, fig)

    layout = {'title' : 'top Canada team'}

    fig.update_layout(layout)

    fig.show()
df = data[['rating', 'team', 'position', 'country', 'full_name', 'salary']]

if len(df.groupby('position').salary.agg('idxmax')) < 5:

    print('Not enough players')

else:

    df = df.loc[df.groupby('position').salary.agg('idxmax')]

    fig = go.Figure()

    draw_plotly_court(fig)

    fig = plot_players(df, fig)

    layout = {'title' : 'Most paid team'}

    fig.update_layout(layout)

    fig.show()
df = data[['rating', 'team', 'position', 'country', 'full_name', 'height']]

if len(df.groupby('position').height.agg('idxmax')) < 5:

    print('Not enough players')

else:

    df = df.loc[df.groupby('position').height.agg('idxmax')]

    fig = go.Figure()

    draw_plotly_court(fig)

    fig = plot_players(df, fig)

    layout = {'title' : 'Most high team'}

    fig.update_layout(layout)

    fig.show()
df = data[['rating', 'team', 'position', 'country', 'full_name', 'age']]

if len(df.groupby('position').age.agg('idxmax')) < 5:

    print('Not enough players')

else:

    df = df.loc[df.groupby('position').age.agg('idxmax')]

    fig = go.Figure()

    draw_plotly_court(fig)

    fig = plot_players(df, fig)

    layout = {'title' : 'Most oldest team'}

    fig.update_layout(layout)

    fig.show()