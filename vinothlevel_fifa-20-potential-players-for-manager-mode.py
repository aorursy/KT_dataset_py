# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import seaborn as sns

import datetime as dt

import warnings

warnings.filterwarnings("ignore")



color = sns.color_palette()

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
fifa=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')
fifa.head()
fifa.dob=pd.to_datetime(fifa.dob)
fifa.potential
fifa_potential=fifa[(fifa.potential>85 )& (fifa.overall<80)]
fifa_potential_ready=fifa_potential[(fifa_potential.overall<80)&(fifa_potential.overall>70)]
position="GK"

fifa_potential_ready_GK=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_ready_GK
figure,ax=plt.subplots(1,2)

figure.set_size_inches(10,10)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_ready_GK.short_name,fifa_potential_ready_GK.overall,ax=ax[0])

ax[0].tick_params(axis="x", rotation=70)

sns.barplot(fifa_potential_ready_GK.short_name,fifa_potential_ready_GK.potential,ax=ax[1])

count=0

for i in fifa_potential_ready_GK.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential GK in FIFA 20 ',fontsize=20)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_ready_GK.value_eur

cnt_srs.index = fifa_potential_ready_GK.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = [" Potential GK player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="LB"

position1="LWB"

fifa_potential_LB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]
fifa_potential_LB
figure,ax=plt.subplots(1,2)

figure.set_size_inches(10,10)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_LB.short_name,fifa_potential_LB.overall,ax=ax[0])

sns.barplot(fifa_potential_LB.short_name,fifa_potential_LB.potential,ax=ax[1])

count=0

for i in fifa_potential_LB.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential LB in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_LB.value_eur

cnt_srs.index = fifa_potential_LB.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential LB player values(Euros) "]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="RB"

position1="RWB"

fifa_potential_RB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]
fifa_potential_RB
figure,ax=plt.subplots(1,2)

figure.set_size_inches(10,10)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_RB.short_name,fifa_potential_RB.overall,ax=ax[0])

sns.barplot(fifa_potential_RB.short_name,fifa_potential_RB.potential,ax=ax[1])

count=0

for i in fifa_potential_RB.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential RB in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_RB.value_eur

cnt_srs.index = fifa_potential_RB.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential RB player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="CB"

fifa_potential_cb=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cb
figure,ax=plt.subplots(1,2)

figure.set_size_inches(15,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_cb.short_name,fifa_potential_cb.overall,ax=ax[0])

sns.barplot(fifa_potential_cb.short_name,fifa_potential_cb.potential,ax=ax[1])

count=0

for i in fifa_potential_cb.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential CB in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_cb.value_eur

cnt_srs.index = fifa_potential_cb.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential CB player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="CM"

fifa_potential_cm=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cm
figure,ax=plt.subplots(1,2)

figure.set_size_inches(20,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_cm.short_name,fifa_potential_cm.overall,ax=ax[0])

sns.barplot(fifa_potential_cm.short_name,fifa_potential_cm.potential,ax=ax[1])

count=0

for i in fifa_potential_cm.potential:

    plt.text(count-0.15,i-4,str(i),size=11,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential CM in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=90)

ax[1].tick_params(axis="x", rotation=90)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_cm.value_eur

cnt_srs.index = fifa_potential_cm.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential CM player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="CAM"

fifa_potential_cam=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cam
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_cam.short_name,fifa_potential_cam.overall,ax=ax[0])

sns.barplot(fifa_potential_cam.short_name,fifa_potential_cam.potential,ax=ax[1])

count=0

for i in fifa_potential_cam.potential:

    plt.text(count-0.15,i-4,str(i),size=14,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential CAM in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=90)

ax[1].tick_params(axis="x", rotation=90)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_cam.value_eur

cnt_srs.index = fifa_potential_cam.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential CAM player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="CDM"

fifa_potential_cdm=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cdm
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

plt.xticks(rotation="70")

sns.set(style='darkgrid')

plt.xticks(rotation="70")

sns.barplot(fifa_potential_cdm.short_name,fifa_potential_cdm.overall,ax=ax[0])

plt.xticks(rotation="70")

sns.barplot(fifa_potential_cdm.short_name,fifa_potential_cdm.potential,ax=ax[1])

count=0

for i in fifa_potential_cdm.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential CDM in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_cdm.value_eur

cnt_srs.index = fifa_potential_cdm.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential CDM player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="LW"

fifa_potential_lw=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_lw
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_lw.short_name,fifa_potential_lw.overall,ax=ax[0])

sns.barplot(fifa_potential_lw.short_name,fifa_potential_lw.potential,ax=ax[1])

count=0

for i in fifa_potential_lw.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential LW in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_lw.value_eur

cnt_srs.index = fifa_potential_lw.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential LW player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="RW"

fifa_potential_rw=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_rw
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_rw.short_name,fifa_potential_rw.overall,ax=ax[0])

sns.barplot(fifa_potential_rw.short_name,fifa_potential_rw.potential,ax=ax[1])

count=0

for i in fifa_potential_rw.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential RW in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_rw.value_eur

cnt_srs.index = fifa_potential_rw.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential RW player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="ST"

fifa_potential_st=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_st
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_st.short_name,fifa_potential_st.overall,ax=ax[0])

sns.barplot(fifa_potential_st.short_name,fifa_potential_st.potential,ax=ax[1])

count=0

for i in fifa_potential_st.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential ST in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_st.value_eur

cnt_srs.index = fifa_potential_st.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential ST player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')
position="CF"

fifa_potential_cf=fifa_potential_ready[fifa.player_positions.str.contains(position)]
fifa_potential_cf
figure,ax=plt.subplots(1,2)

figure.set_size_inches(18,8)

figure=plt.gcf()

sns.set(style='darkgrid')

sns.barplot(fifa_potential_cf.short_name,fifa_potential_cf.overall,ax=ax[0])

sns.barplot(fifa_potential_cf.short_name,fifa_potential_cf.potential,ax=ax[1])

count=0

for i in fifa_potential_cf.potential:

    plt.text(count-0.15,i-4,str(i),size=15,color='black',rotation=90)

    count+=1

figure.suptitle('Top potential CF in FIFA 20 ',fontsize=20)

ax[0].tick_params(axis="x", rotation=70)

ax[1].tick_params(axis="x", rotation=70)
def scatter_plot(cnt_srs, color):

   trace = go.Scatter(

       x=cnt_srs.index[::-1],

       y=cnt_srs.values[::-1],

       showlegend=False,

       marker=dict(

           color=color,

       ),

   )

   return trace

cnt_srs = fifa_potential_cf.value_eur

cnt_srs.index = fifa_potential_cf.short_name

trace1 = scatter_plot(cnt_srs, 'red')

subtitles = ["Potential CF player values(Euros)"]

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,

                         subplot_titles=subtitles)

fig.append_trace(trace1, 1, 1)

fig['layout'].update(height=600, width=600, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='h2o-plots')