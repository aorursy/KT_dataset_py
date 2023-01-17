import plotly as py

from plotly import graph_objs as go

from plotly.offline import iplot, init_notebook_mode

from plotly import tools



import numpy as np

import pandas as pd
init_notebook_mode(connected=True)
# Load the data

df = pd.read_csv('../input/Pokemon.csv')



df.head()
#Let's normalize the data first (fill the empty value in 'Type 2' column)

df['Type 2'].fillna('', inplace=True)



# Next, list all types of pokemon

types = df['Type 1'].unique()

types2 = np.insert(types, 18, '')



# Now we can count all Pokemon based on their types (type 1 & type 2)

types_count = []

for t in types:

    criteria_1 = df['Type 1'] == t

    criteria_2 = df['Type 2'] == t

    

    types_count.append(len(df[criteria_1 | criteria_2]))

    

# We do the same thing with Type 1 and Type 2 only

type1_count = []

for t in types:

    criteria = df['Type 1'] == t

    type1_count.append(len(df[criteria]))



type2_count = []

for t in types2:

    criteria = df['Type 2'] == t

    type2_count.append(len(df[criteria]))
# Color list based on actual types color in game

colors = ['#78C850', '#F08030', '#6890F0', '#A8B820', '#A8A878', '#A040A0', '#F8D030', '#E0C068',

         '#EE99AC', '#C03028', '#F85888', '#B8A038', '#705898', '#98D8D8', '#7038F8', '#705848', '#B8B8D0',

         '#A890F0', '#353535']
# Make a bar chart

trace_bar = go.Bar(x = types,

                   y = types_count,                   

                   marker=dict(color=colors)

                  )



layout = go.Layout(title='<b>Total Number of Pokemon Based on Type</b>',

                   height=650,

                   margin=go.layout.Margin(                   

                   pad=5)

                  )



data = [trace_bar]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# Now lets we set for each generation



types_count = []

types_count_1 = []

types_count_2 = []

types_count_3 = []

types_count_4 = []

types_count_5 = []

types_count_6 = []



for t in types:

    criteria_1 = df['Type 1'] == t

    criteria_2 = df['Type 2'] == t

    criteria_all = criteria_1 | criteria_2

    

    types_count.append(len(df[criteria_all]))

    types_count_1.append(len(df[criteria_all & (df['Generation'] == 1)]))

    types_count_2.append(len(df[criteria_all & (df['Generation'] == 2)]))

    types_count_3.append(len(df[criteria_all & (df['Generation'] == 3)]))

    types_count_4.append(len(df[criteria_all & (df['Generation'] == 4)]))

    types_count_5.append(len(df[criteria_all & (df['Generation'] == 5)]))

    types_count_6.append(len(df[criteria_all & (df['Generation'] == 6)]))  
trace_1 = go.Bar(x = types,

                 y = types_count_1,

                 marker=dict(color=colors))



trace_2 = go.Bar(x = types,

                 y = types_count_2,

                 marker=dict(color=colors))



trace_3 = go.Bar(x = types,

                 y = types_count_3,

                 marker=dict(color=colors))



trace_4 = go.Bar(x = types,

                 y = types_count_4,

                 marker=dict(color=colors))



trace_5 = go.Bar(x = types,

                 y = types_count_5,

                 marker=dict(color=colors))



trace_6 = go.Bar(x = types,

                 y = types_count_6,

                 marker=dict(color=colors))



# data = [trace_1, trace_2, trace_3, trace_4, trace_5, trace_6]





fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('Generation 1', 'Generation 2',

                                                          'Generation 3', 'Generation 4',

                                                          'Generation 5', 'Generation 6'),

                          horizontal_spacing = 0.1,

                          vertical_spacing = 0.22,

                          print_grid=False

                         )





fig.append_trace(trace_1, 1, 1)

fig.append_trace(trace_2, 1, 2)

fig.append_trace(trace_3, 2, 1)

fig.append_trace(trace_4, 2, 2)

fig.append_trace(trace_5, 3, 1)

fig.append_trace(trace_6, 3, 2)



fig['layout'].update(title='<b>Number of Pokemon for Each Generation</b>',

                   height=670,

                   margin=go.layout.Margin(pad=5),

                   showlegend=False,

                   

                    )



iplot(fig)
# Make bar chart to compare between pokes who have 2 type and who don't

labels = ['Single Type', 'Double Type']

notype2 = [len(df[df['Type 2'] != '']), len(df[df['Type 2'] == '']) ]



trace = go.Pie(labels=labels,

               values=notype2,

               textfont=dict(size=19, color='#FFFFFF'),

               marker=dict(

                   colors=['#DB0415', '#2424FF'] 

               )

              )



layout = go.Layout(title = '<b>Single Type vs Double Type</b>')

data = [trace]

fig = go.Figure(data=data, layout=layout)



iplot(fig)
z1 = np.array([])

z2 = []

z3 = np.array([])

for t2 in types2: 

    for t1 in types:

        criteria1 = df['Type 1'] == t1

        criteria2 = df['Type 2'] == t2

        z2_val = len(df[criteria1 & criteria2])

        z2.append(z2_val)

    z1 = np.append(z1,[z2])

    z2.clear()





z3 = np.reshape(z1, (19, 18))



# Make heatmap graph

trace = go.Heatmap(z=z3,

                   x=types,

                   y=types2,

                   colorscale='Hot',

                   reversescale=True

                  )



layout = go.Layout(

    title='<b>Type 1 - Type 2 Relation</b>',

    xaxis = dict( title='<b>Type 1</b>'),

    yaxis = dict(title='<b>Type 2</b>' ),

    

)



data=[trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# Now Let's Move to Stats Section

df_gen_1 = df[df['Generation'] == 1]

df_gen_2 = df[df['Generation'] == 2]

df_gen_3 = df[df['Generation'] == 3]

df_gen_4 = df[df['Generation'] == 4]

df_gen_5 = df[df['Generation'] == 5]

df_gen_6 = df[df['Generation'] == 6]



total_list = [df_gen_1['Total'].mean(), df_gen_2['Total'].mean(), df_gen_3['Total'].mean(),

           df_gen_4['Total'].mean(), df_gen_5['Total'].mean(), df_gen_6['Total'].mean()]

total_list = [int(i) for i in total_list]

# print(results)



trace = go.Scatter(x=df['Generation'].unique(),

                   y=total_list,

                   name='Mean of Total Stats'

                  )

layout = go.Layout(title='<b>Average Total Status in Each Generation</b>',

                  showlegend=True,

                  yaxis=dict(

                    range=[400, 480],

                    dtick=10)

                  )



fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
status_type = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



all_stats_1 = [df_gen_1['HP'].mean(), df_gen_1['Attack'].mean(), df_gen_1['Defense'].mean(),

               df_gen_1['Sp. Atk'].mean(), df_gen_1['Sp. Def'].mean(), df_gen_1['Speed'].mean()]



all_stats_2 = [df_gen_2['HP'].mean(), df_gen_2['Attack'].mean(), df_gen_2['Defense'].mean(),

               df_gen_2['Sp. Atk'].mean(), df_gen_2['Sp. Def'].mean(), df_gen_2['Speed'].mean()]



all_stats_3 = [df_gen_3['HP'].mean(), df_gen_3['Attack'].mean(), df_gen_3['Defense'].mean(),

               df_gen_3['Sp. Atk'].mean(), df_gen_3['Sp. Def'].mean(), df_gen_3['Speed'].mean()]



all_stats_4 = [df_gen_4['HP'].mean(), df_gen_4['Attack'].mean(), df_gen_4['Defense'].mean(),

               df_gen_4['Sp. Atk'].mean(), df_gen_4['Sp. Def'].mean(), df_gen_4['Speed'].mean()]



all_stats_5 = [df_gen_5['HP'].mean(), df_gen_5['Attack'].mean(), df_gen_5['Defense'].mean(),

               df_gen_5['Sp. Atk'].mean(), df_gen_5['Sp. Def'].mean(), df_gen_5['Speed'].mean()]



all_stats_6 = [df_gen_6['HP'].mean(), df_gen_6['Attack'].mean(), df_gen_6['Defense'].mean(),

               df_gen_6['Sp. Atk'].mean(), df_gen_6['Sp. Def'].mean(), df_gen_6['Speed'].mean()]





all_stats_1 = [int(i) for i in all_stats_1]

all_stats_2 = [int(i) for i in all_stats_2]

all_stats_3 = [int(i) for i in all_stats_3]

all_stats_4 = [int(i) for i in all_stats_4]

all_stats_5 = [int(i) for i in all_stats_5]

all_stats_6 = [int(i) for i in all_stats_6]



trace1 = go.Scatter(x = status_type,

                    y = all_stats_1,

                    name = 'Generation 1')



trace2 = go.Scatter(x = status_type,

                    y = all_stats_2,

                    name = 'Generation 2')



trace3 = go.Scatter(x = status_type,

                    y = all_stats_3,

                    name = 'Generation 3')



trace4 = go.Scatter(x = status_type,

                    y = all_stats_4,

                    name = 'Generation 4')



trace5 = go.Scatter(x = status_type,

                    y = all_stats_5,

                    name = 'Generation 5')



trace6 = go.Scatter(x = status_type,

                    y = all_stats_6,

                    name = 'Generation 6')



data = [trace1,trace2, trace3, trace4, trace5, trace6]



layout = go.Layout(title='<b>Each Generation Stats Comparison</b>',

                  showlegend=True,

                  yaxis=dict(

                    range=[60, 85],

                    dtick=5)

                  )



fig = go.Figure(data=data, layout=layout)

iplot(fig)



# Status Comparison per each Type



total_mean_values = []

for t in types:

    criteria_1 = df['Type 1'] == t

    criteria_2 = df['Type 2'] == t

    criteria = criteria_1 | criteria_2

    df_type = df[criteria]

    total_mean = df_type['Total'].mean()

    total_mean_values.append(total_mean)

    

# Horizontal

trace_bar = go.Bar(x = total_mean_values,

                   y = types,

                   orientation = 'h',

                   marker=dict(color=colors)

                  )



layout = go.Layout(

                    title='<b>Average Total Power for Each Types</b>',

                    margin=go.layout.Margin(                   

                        pad=5),

                    xaxis=dict(

                            range=[350, 550],

                            dtick=50)

                  )

data = [trace_bar]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
labels = ['Ordinary', 'Legendary']

values = [len(df[df['Legendary'] == False]), len(df[df['Legendary'] == True])]

colors_pie = ['#4777BA', '#DC2634', '#FFC009', '#FBFF09']



trace = go.Pie(labels=labels, 

               values=values,

               textfont=dict(size=16),

               marker=dict(

                   colors=colors_pie,

#                    line=dict(color='#000000', width = 0.8)

               )

              )





layout = go.Layout(

                    title='<b>How Many Legendaries Out There?</b>',

                    margin=go.layout.Margin(                   

                        pad=5)

)



data = [trace]

fig = go.Figure(data=data, layout=layout)



iplot(fig)
count_ordinary_list = []

for i in range(1,7):

    criteria_1 = df['Legendary'] == False

    criteria_2 = df['Generation'] == i

    count_ordinary = df[criteria_1 & criteria_2]

    count_ordinary = len(count_ordinary)

    count_ordinary_list.append(count_ordinary)

    

count_legendary_list = []

for i in range(1,7):

    criteria_1 = df['Legendary'] == True

    criteria_2 = df['Generation'] == i

    count_legendary = df[criteria_1 & criteria_2]

    count_legendary = len(count_legendary)

    count_legendary_list.append(count_legendary)

    

trace1 = go.Bar(x = df['Generation'].unique(),

               y = count_ordinary_list,

               name = 'Ordinary')



trace2 = go.Bar(x = df['Generation'].unique(),

               y = count_legendary_list,

               name = 'Legendary')



data = [trace1, trace2]



layout = go.Layout(

    title='<b>Number of Ordinaries vs Legendaries Pokemon in Each Gen</b>',

    barmode='group',

    xaxis=dict(

        title='Generation',

        titlefont=dict(

            family='Arial, sans-serif',

            size=19,

            color='grey'

        ),

        tickfont=dict(

            family='Arial, serif',

            size=14,

            color='black'

        )

    )

    

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_ordinary = df[df['Legendary'] == False]

ordinary_stats = []

for i in range(5,11):

    m = df_ordinary.iloc[:,i].mean()

    ordinary_stats.append(round(m))



df_legendary = df[df['Legendary'] == True]

legendary_stats = []

for i in range(5,11):

    m = df_legendary.iloc[:,i].mean()

    legendary_stats.append(round(m))



data = [

    go.Scatterpolar(

      r = legendary_stats,

      theta = status_type,

      fill = 'toself',

      name = 'Legendary'

    ),

    go.Scatterpolar(

      r =ordinary_stats,

      theta = status_type,

      fill = 'toself',

      name = 'Ordinary'

    )

]



layout = go.Layout(

    title = '<b>Ordinary vs Legendary Stats Comparison</b>',

    polar = dict(

        radialaxis = dict(

          visible = True,

          range = [0, 150]

        )

      ),

    showlegend = True

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)