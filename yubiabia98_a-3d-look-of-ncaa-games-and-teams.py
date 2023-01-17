import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import plotly.offline as py

from plotly.graph_objs import *

import plotly.graph_objects as go

from skimage import io

import cv2

from skimage.color import rgb2lab, deltaE_cie76

import os

from sklearn.cluster import KMeans

from collections import Counter

from plotly.subplots import make_subplots
M_Event_2019=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2019.csv')[['EventTeamID','X','Y','Area','EventType']]

TeamIDmap=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')[['TeamID','TeamName']]

M_Event_2019=pd.merge(M_Event_2019,TeamIDmap,how='left', left_on=['EventTeamID'], right_on=['TeamID'])[['TeamName','X','Y','EventType']]



made3_2019_M=M_Event_2019[(M_Event_2019['EventType']=='made3')&(M_Event_2019['X']!=0)][['TeamName','X','Y']]

miss3_2019_M=M_Event_2019[(M_Event_2019['EventType']=='miss3')&(M_Event_2019['X']!=0)][['TeamName','X','Y']]

made3_2019_M['X']=made3_2019_M['X'].apply(lambda x: int((x-1)/10))

made3_2019_M['Y']=made3_2019_M['Y'].apply(lambda x: int((x-1)/20))

miss3_2019_M['X']=miss3_2019_M['X'].apply(lambda x: int((x-1)/10))

miss3_2019_M['Y']=miss3_2019_M['Y'].apply(lambda x: int((x-1)/20))

made3_team=made3_2019_M.groupby(['TeamName','X','Y']).size().rename('Made').reset_index()

miss3_team=miss3_2019_M.groupby(['TeamName','X','Y']).size().rename('Miss').reset_index()

Team_3=pd.merge(miss3_team,made3_team,on=['TeamName','X','Y'],how='outer').fillna(0)

Team_3['Made%']=Team_3['Made']/(Team_3['Made']+Team_3['Miss'])

mean3_percentage=Team_3.groupby(['X','Y'])['Made%'].mean().reset_index()

for index,row in mean3_percentage.iterrows():

  x=row['X']

  y=row['Y']

  percentage=row['Made%']

  Team_3.loc[(Team_3['X']==x) & (Team_3['Y']==y),'mean']=percentage

Team_3['diff']=Team_3['Made%']-Team_3['mean']

Team_3['Total']=Team_3['Made']+Team_3['Miss']



made2_2019_M=M_Event_2019[(M_Event_2019['EventType']=='made2')&(M_Event_2019['X']!=0)][['TeamName','X','Y']]

miss2_2019_M=M_Event_2019[(M_Event_2019['EventType']=='miss2')&(M_Event_2019['X']!=0)][['TeamName','X','Y']]

made2_2019_M['X']=made2_2019_M['X'].apply(lambda x: int((x-1)/10))

made2_2019_M['Y']=made2_2019_M['Y'].apply(lambda x: int((x-1)/20))

miss2_2019_M['X']=miss2_2019_M['X'].apply(lambda x: int((x-1)/10))

miss2_2019_M['Y']=miss2_2019_M['Y'].apply(lambda x: int((x-1)/20))

made2_team=made2_2019_M.groupby(['TeamName','X','Y']).size().rename('Made').reset_index()

miss2_team=miss2_2019_M.groupby(['TeamName','X','Y']).size().rename('Miss').reset_index()

Team_2=pd.merge(miss2_team,made2_team,on=['TeamName','X','Y'],how='outer').fillna(0)

Team_2['Made%']=Team_2['Made']/(Team_2['Made']+Team_2['Miss'])

mean_percentage=Team_2.groupby(['X','Y'])['Made%'].mean().reset_index()

for index,row in mean_percentage.iterrows():

  x=row['X']

  y=row['Y']

  percentage=row['Made%']

  Team_2.loc[(Team_2['X']==x) & (Team_2['Y']==y),'mean']=percentage

Team_2['diff']=Team_2['Made%']-Team_2['mean']

Team_2['Total']=Team_2['Made']+Team_2['Miss']
img = cv2.imread('../input/ncaaoldcourt/basketball_court-2_(NCAAOld).png')

modified_image = cv2.resize(img, (100, 100))

modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

clf = KMeans(n_clusters = 13)

labels = clf.fit_predict(modified_image)

center_colors = clf.cluster_centers_

pl_color=[]

for i in range(13):

  pl_color.append([i/13,'rgb('+str(center_colors[i][0])+','+str(center_colors[i][1])+','+str(center_colors[i][2])+')'])

surfcolor=np.reshape(labels,(100,100))

surfcolor=surfcolor/13
def get_team_Z(teamname):

  team=Team_all[Team_all['TeamName']==teamname]

  threshold_total=np.quantile(team['Total'],0.25)

  team=team[team['Total']>threshold_total]

  team_all=np.zeros((100,100))

  for i in list(set(team['X'])):

    y=list(set(team[team['X']==i]['Y']))

    for j in y:

      team_all[j*20:j*20+20,i*10:i*10+10]=np.array([[team[(team['X']==i)&(team['Y']==j)]['diff'].values[0] for r in range(10)] for h in range (20)])

  return team_all



name=['Virginia','Texas Tech','Auburn','Michigan St']

Team_all=pd.concat([Team_2, Team_3], axis=0)

fig = make_subplots(

    subplot_titles=tuple(['<b>'+str(name[i])+'</b>'for i in range(4)]),

    horizontal_spacing = 0.02,

    vertical_spacing=0.02,

    print_grid=False,

    rows=2, cols=2,

    specs=[[{'type': 'surface'}, {'type': 'surface'}],

           [{'type': 'surface'}, {'type': 'surface'}]])

sub_scene=dict(xaxis_visible=True,

                      yaxis_visible=True, 

                      zaxis_visible=True,

                      aspectmode='manual',

                      aspectratio = dict(x=1, y=0.5319, z=0.4)

                      )

for i in range(4):

  fig.add_trace(go.Surface(z=get_team_Z(name[i]),

                      x=np.linspace(0,940,100),

                      y=np.linspace(0,500,100),

                      surfacecolor=surfcolor,

                      colorscale='Spectral',

                      showscale =False

                    ),

                row=int(i/2+1), 

                col=int(i%2+1))



fig.update_layout(

    title='<b>Final Four relative shooting % in respective to all teams</b>', 

          width=800,

          height=800,

          autosize=False,

          scene=sub_scene,

          scene2=sub_scene,

          scene3=sub_scene,

          scene4=sub_scene

                      )

fig.add_layout_image(

    dict(

        source="https://upload.wikimedia.org/wikipedia/en/thumb/d/d1/Virginia_Cavaliers_sabre.svg/1200px-Virginia_Cavaliers_sabre.svg.png",

        xref="paper", yref="paper",

         x=0, y=1,

        sizex=0.15,sizey=0.15,

    )

)

fig.add_layout_image(

    dict(

        source="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Texas_Tech_Athletics_logo.svg/1200px-Texas_Tech_Athletics_logo.svg.png",

        xref="paper", yref="paper",

         x=0.8, y=1,

        sizex=0.15,sizey=0.15,

    )

)

fig.add_layout_image(

    dict(

        source="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Auburn_Tigers_logo.svg/1200px-Auburn_Tigers_logo.svg.png",

        xref="paper", yref="paper",

         x=0, y=0.2,

        sizex=0.15,sizey=0.15,

    )

)

fig.add_layout_image(

    dict(

        source="https://upload.wikimedia.org/wikipedia/en/thumb/a/a7/Michigan_State_Athletics_logo.svg/1200px-Michigan_State_Athletics_logo.svg.png",

        xref="paper", yref="paper",

         x=0.8, y=0.2,

        sizex=0.15,sizey=0.15,

    )

)
def create_ncaa_full_court():

    trace1 = {

      "uid": "3608439e-d007-11e8-bf2c-f2189834773b", 

      "type": "scatter", 

      "x": [47], 

      "y": [25],

      "marker":{"color": "rgba(0,0,0,0)"},

      "showlegend":False

    }

    data = Scatter(trace1)

    layout = {

        "autosize":False,

        "font":dict(

        family="Balto",

        size=18,

        color="rgba(0,0,0,1)"

    ),

      "title": "NCAA Basketball Court", 

      "paper_bgcolor":"rgba(0,0,0,0)",

    "plot_bgcolor":"rgba(0,0,0,0)",

      "shapes": [

        {

          "x0": 4, 

          "x1": 4, 

          "y0": 22, 

          "y1": 28, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "line"

        }, 

        {

          "x0": 90, 

          "x1": 90, 

          "y0": 22, 

          "y1": 28, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "line"

        }, 

        {

          "x0": 0, 

          "x1": 19, 

          "y0": 17, 

          "y1": 33, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        }, 

        {

          "x0": 0, 

          "x1": 19, 

          "y0": 19, 

          "y1": 31, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        }, 

        {

          "x0": 75, 

          "x1": 94, 

          "y0": 17, 

          "y1": 33, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        }, 

        {

          "x0": 75, 

          "x1": 94, 

          "y0": 19, 

          "y1": 31, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        },  

        {

          "x0": 47, 

          "x1": 47, 

          "y0": 0, 

          "y1": 50, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        }, 

        {

          "x0": 6 ,

          "x1": 4.5, 

          "y0": 25.75, 

          "y1": 24.25, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 89.5, 

          "x1": 88, 

          "y0": 25.75, 

          "y1": 24.25, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 25, 

          "x1": 13, 

          "y0": 31, 

          "y1": 19, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 81, 

          "x1": 69, 

          "y0": 31, 

          "y1": 19, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 53, 

          "x1": 41, 

          "y0": 31, 

          "y1": 19, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 49, 

          "x1": 45, 

          "y0": 27, 

          "y1": 23, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        }, 

        {

          "x0": 68, 

          "x1": 109.5, 

          "y0": 4.25, 

          "y1": 45.75, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        },

            {

          "x0": -15.5, 

          "x1": 26, 

          "y0": 4.25, 

          "y1": 45.75, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "circle"

        },

          {

          "x0": -15.5, 

          "x1": 0, 

          "y0": 50, 

          "y1": 0, 

         "fillcolor":"white",

        "line_color":"white",

          "type": "rect"

        },

          {

          "x0": 110.25, 

          "x1": 94, 

          "y0": 50, 

          "y1": 0, 

         "fillcolor":"white",

        "line_color":"white",

          "type": "rect"

        },

            {

          "x0": 0, 

          "x1": 94, 

          "y0": 0, 

          "y1": 50, 

          "line": {

            "color": "rgba(0,0,0,1)", 

            "width": 1

          }, 

          "type": "rect"

        }, 

      ]

    }

    fig = Figure(data=data,layout=layout)

    return fig

empty=create_ncaa_full_court()

empty
Made2_2019_M=M_Event_2019[(M_Event_2019['EventType']=='made2') & (M_Event_2019['X']!=0)]

XY_Made2=Made2_2019_M.groupby(['X','Y']).size().rename('Count').reset_index()

XY_Made2['X_corr']=XY_Made2['X']*0.94

XY_Made2['Y_corr']=XY_Made2['Y']*0.5

XY_Made2['log_Count']=np.log(XY_Made2['Count'])

made2_court=create_ncaa_full_court()

made2_court.add_scatter(x=XY_Made2['X_corr'], y=XY_Made2['Y_corr'],

                    mode='markers',

                    name='2 Points made',

                    marker=dict(color=XY_Made2['log_Count'],

                    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],

                    [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],

                    [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],

                    [1.0, 'rgb(49,54,149)']],

                    reversescale=True,

                    #colorscale='Burg', # one of plotly colorscales

                    showscale=True,),

                    showlegend=False,

                    text=XY_Made2['log_Count'])

made2_court.update_layout({"title": "<b>NCAA 2019 2-points shooting position </b>","font": {

    "family": 'Courier New, monospace',

    "size":18,

  }})
Miss3_2019_M=M_Event_2019[(M_Event_2019['EventType']=='miss3') & (M_Event_2019['X']!=0)]

XY_Miss3=Miss3_2019_M.groupby(['X','Y']).size().rename('Count').reset_index()

Made3_2019_M=M_Event_2019[(M_Event_2019['EventType']=='made3') & (M_Event_2019['X']!=0)]

XY_Made3=Made3_2019_M.groupby(['X','Y']).size().rename('Count').reset_index()

XY_Miss3['X_corr']=XY_Miss3['X']*0.94

XY_Miss3['Y_corr']=XY_Miss3['Y']*0.5

XY_Made3['X_corr']=XY_Made3['X']*0.94

XY_Made3['Y_corr']=XY_Made3['Y']*0.5

XY_Miss3=XY_Miss3.rename(columns={'Count':'Miss'})

XY_Made3=XY_Made3.rename(columns={'Count':'Made'})

XY_3=pd.merge(XY_Miss3,XY_Made3,on=['X_corr','Y_corr'],how='outer').fillna(0)

XY_3['Made%']=XY_3['Made']/(XY_3['Made']+XY_3['Miss'])

XY_3['Total']=XY_3['Made']+XY_3['Miss']

XY_3=XY_3[XY_3['Total']>=3]

Threepts_court=create_ncaa_full_court()

Threepts_court.add_scatter(x=XY_3['X_corr'], y=XY_3['Y_corr'],

                    mode='markers',

                    name='3 Points accuracy',

                    marker=dict(symbol="hexagon",

                        color=np.log(XY_3['Made']),

                    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],

                    [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],

                    [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],

                    [1.0, 'rgb(49,54,149)']],

                    reversescale=True,

                    #colorscale='Burg', # one of plotly colorscales

                    showscale=True,),

                    showlegend=False,

                    text=XY_3['Made'])

Threepts_court.update_layout({"font": {

    "family": 'Courier New, monospace',

    "size":18,

  },"title": "<b>NCAA 2019 3-points shooting position</b>"})
Threepts_court.add_scatter(x=XY_Made2['X_corr'], y=XY_Made2['Y_corr'],

                    mode='markers',

                    name='2 Points made',

                    marker=dict(color=XY_Made2['log_Count'],

                    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],

                    [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],

                    [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],

                    [1.0, 'rgb(49,54,149)']],

                    reversescale=True,

                    #colorscale='Burg', # one of plotly colorscales

                    ),

                    showlegend=False,

                    text=XY_Made2['log_Count'])

Threepts_court.update_layout({"title": "<b>NCAA 2019 2-points & 3-points</b>","font": {

    "family": 'Courier New, monospace',

    "size":18,

  }})
Miss2_2019_M=M_Event_2019[(M_Event_2019['EventType']=='miss2') & (M_Event_2019['X']!=0)]

XY_Miss2=Miss2_2019_M.groupby(['X','Y']).size().rename('Count').reset_index()

XY_Miss2['X_corr']=XY_Miss2['X']*0.94

XY_Miss2['Y_corr']=XY_Miss2['Y']*0.5

XY_Miss2=XY_Miss2.rename(columns={'Count':'Miss'})

XY_Made2=XY_Made2.rename(columns={'Count':'Made'})

XY_2=pd.merge(XY_Miss2,XY_Made2,on=['X_corr','Y_corr'],how='outer').fillna(0)

XY_2['Made%']=XY_2['Made']/(XY_2['Made']+XY_2['Miss'])

XY_2=XY_2[XY_2['Made']>=2]

XY_2=XY_2[XY_2['Miss']>=4]

Twopts_court=create_ncaa_full_court()

Twopts_court.add_scatter(x=XY_2['X_corr'], y=XY_2['Y_corr'],

                    mode='markers',

                    name='2 Points accuracy',

                    marker=dict(color=XY_2['Made%'],

                    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111, 'rgb(215,48,39)'], [0.22222, 'rgb(244,109,67)'],

                    [0.333, 'rgb(253,174,97)'], [0.4444, 'rgb(254,224,144)'], [0.5555, 'rgb(224,243,248)'],

                    [0.6666, 'rgb(171,217,233)'],[0.7777, 'rgb(116,173,209)'], [0.8888, 'rgb(69,117,180)'],

                    [1.0, 'rgb(49,54,149)']],

                    reversescale=True,

                    #colorscale='Burg', # one of plotly colorscales

                    ),

                    showlegend=False,

                    text=XY_2['Made%'])

Twopts_court.add_scatter(x=XY_3['X_corr'], y=XY_3['Y_corr'],

                    mode='markers',

                    name='3 Points accuracy',

                    marker=dict(symbol="hexagon",

                        color=XY_3['Made%'],

                    colorscale=[[0.0, 'rgb(165,0,38)'], [1/7, 'rgb(215,48,39)'], [2/7, 'rgb(244,109,67)'],

                    [3/7, 'rgb(253,174,97)'], [4/7, 'rgb(254,224,144)'],[5/7, 'rgb(224,243,248)'],[6/7, 'rgb(171,217,233)'],[1, 'rgb(116,173,209)']],

                    reversescale=True,

                    #colorscale='Burg', # one of plotly colorscales

                    showscale=True,),

                    showlegend=False,

                    text=XY_3['Made%'])

Twopts_court.update_layout({"font": {

    "family": 'Courier New, monospace',

    "size":18,

  },"title": "<b>NCAA 2019 Field Goals </b>"})
Cities=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/Cities.csv')

Cities_Game=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MGameCities.csv')

Game_count=pd.merge(Cities_Game,Cities).groupby(['Season','State']).size().rename('Game_count').reset_index()
CompactResults=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

home_game=pd.merge(CompactResults,Cities_Game)[['Season','WTeamID','LTeamID','WLoc','CityID']]

home_game=home_game[(home_game['WLoc']=='H') |(home_game['WLoc']=='A')]

home_game['home_team']=home_game.apply(lambda row: row['WTeamID'] if row['WLoc']=='H' else row['LTeamID'],axis=1)

season_team_city=home_game.groupby(['Season','home_team','CityID']).size().rename('games').reset_index()

team_city_state=pd.merge(season_team_city,Cities)

team_state=team_city_state.groupby(['Season','State']).size().rename('NumberofTeams').reset_index()

# Create figure

fig = go.Figure()



# Add traces, one for each slider step

for step in np.arange(2010, 2020, 1):

    fig.add_trace(

      go.Choropleth(

    locations=team_state[team_state['Season']==step]['State'], # Spatial coordinates

    z = team_state[team_state['Season']==step]['NumberofTeams'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],

        [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],

        [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],

        [1.0, 'rgb(49,54,149)']],

    reversescale=True,

    colorbar_title = "#Teams",

    marker_line_color='white',

    text = team_state['State'],

    hovertemplate = '<b>State</b>: <b>%{text}</b>'+

    '<br><b>#Teams </b>: %{z}<br>',

    zmin=0,

    zmax=25

))



# Make 10th trace visible

#fig.data[9].visible = True



# Create and add slider

steps = []

for i in range(len(fig.data)):

    step = dict(

        method="restyle",

        args=["visible", [False] * len(fig.data)],

        label='Season {}'.format(i + 2010)

    )

    step["args"][1][i] = True  # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active=10,

    currentvalue={"prefix": "Selected Season: "},

    pad={"t": 10},

    steps=steps

)]



fig.update_layout(

    title_text='<b>2011-2019 NCAA Regular Season #Teams by State</b><br>(Hover for breakdown)',

    sliders=sliders,

    geo_scope='usa'

)



fig.show()
Tourney_seeds=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv')

Team_name=pd.read_csv('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')[['TeamID','TeamName']]

team_tourney=pd.merge(Tourney_seeds.groupby(['Season','TeamID']).size().reset_index()[['Season','TeamID']],team_city_state.rename(columns={'home_team':'TeamID'}).drop_duplicates(subset=['TeamID','Season','State']))[['Season','TeamID','City','State']]

Team_name_state=pd.merge(team_tourney.groupby(['Season','State','TeamID']).size().reset_index()[['Season','State','TeamID']],Team_name[['TeamID','TeamName']])

State_all_team=Team_name_state.groupby(['Season','State'],as_index=False)[['TeamName']].aggregate(lambda x: set(x))

State_all_team['#Teams']=State_all_team.apply(lambda row: len(row['TeamName']),axis=1)

State_all_team['Text']='<b>Teams</b>:'+State_all_team['TeamName'].apply(lambda x: '<br>'.join(str(s) for s in x))

all_states=State_all_team['State'].unique().tolist()

for i in range(10):

  season=2010+i

  season_state=State_all_team[State_all_team['Season']==season]['State'].unique().tolist()

  not_season_state=set(all_states)-set(season_state)

  not_in=len(not_season_state)

  extra_state=pd.DataFrame()

  extra_state['State']=list(not_season_state)

  extra_state['Season']=[season for s in range(not_in)]

  extra_state['#Teams']=[0 for s in range(not_in)]

  extra_state['TeamName']=['None' for s in range(not_in)]

  extra_state['Text']='None'

  State_all_team=pd.concat([State_all_team,extra_state])

State_all_team=State_all_team.sort_values(by=['Season','State'])
# Create figure

fig = go.Figure()



# Add traces, one for each slider step

for step in np.arange(2010, 2020, 1):

    fig.add_trace(

      go.Choropleth(

    locations=State_all_team[State_all_team['Season']==step]['State'], # Spatial coordinates

    z = State_all_team[State_all_team['Season']==step]['#Teams'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],

        [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],

        [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],

        [1.0, 'rgb(49,54,149)']],

    reversescale=True,

    colorbar_title = "#Teams",

    marker_line_color='white',

    text = State_all_team[State_all_team['Season']==step]['Text'],

    zmin=0,

    zmax=7

))



# Make 10th trace visible

#fig.data[9].visible = True



# Create and add slider

steps = []

for i in range(len(fig.data)):

    step = dict(

        method="restyle",

        args=["visible", [False] * len(fig.data)],

        label='Season {}'.format(i + 2010)

    )

    step["args"][1][i] = True  # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active=10,

    currentvalue={"prefix": "Selected Season: "},

    pad={"t": 10},

    steps=steps

)]



fig.update_layout(

    title_text='<b>2011-2019 NCAA Tourney #Teams by State</b><br>(Hover for breakdown)',

    sliders=sliders,

    geo_scope='usa'

)



fig.show()