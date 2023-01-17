import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly as plty

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import iplot

from plotly.subplots import make_subplots

import plotly.io as pio

import spacy

import os

from IPython.display import display

%matplotlib inline
path = '../input/chai-time-data-science/'
df_episodes = pd.read_csv(f'{path}Episodes.csv',parse_dates=['recording_date','release_date'])

df_yt = pd.read_csv(f'{path}YouTube Thumbnail Types.csv')

df_anchortn = pd.read_csv(f'{path}Anchor Thumbnail Types.csv')

df_desc = pd.read_csv(f'{path}Description.csv')
pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 100)

pd.set_option('display.expand_frame_repr', False)
print('\033[33m' + 'Episodes Dataset - Exploration stats' + '\033[0m')

df_episodes.describe(include='all').T
fig = go.Figure(data=[go.Table(header=dict(values=['<b>Factor</b>', '<b>Total</b>', '<b>Average</b>'], line_color='darkgray',fill_color='darkorange',font=dict(color='white', size=16)),

                 cells=dict(values=[['Episodes', 'Duration (min)', 'Youtube views', 'Youtube avg watch duration (min)', 'Youtube subscribers','Spotify streams', 'Spotify listeners', 'Apple listeners','Apple avg listen duration (min)'], 

                                    ['85 with 72 unique heroes', 271991, 43616, 450, 1027, 6720,5455,1714,2434], 

                                    ['7 per month', 3200, 513, 5.3, 12, 80, 65, 21, 29.33]], line_color='darkgray',fill_color='white',font=dict(color='black', size=12), height=25))])



fig.update_layout(height=450, width=870)



fig.show()
fig = px.scatter(df_episodes, x='episode_id', y='youtube_subscribers', height=400, title='<b>Episodes Vs New Youtube Subscribers</b>', color='youtube_subscribers',

             color_continuous_scale='Viridis')

mean_s = df_episodes['youtube_subscribers'].mean()

fig.add_shape(type="line", x0='E0', x1='E75', y0=mean_s, y1=mean_s, line=dict(color='darkblue', width=2, dash='dot'),)

fig.add_trace(go.Scatter(x=['E64'], y=[40], text=["--- Mean Subscribers"], mode="text"))

fig.update_layout(plot_bgcolor='rgb(255,255,255)', showlegend=False)

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title='Youtube subscribers')

fig.data[0].update(mode='markers+lines')

fig.show()
df_episodes['yt_subs_cumulative'] = df_episodes['youtube_subscribers'].cumsum()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,vertical_spacing=0.03, subplot_titles=("<b>Youtube Views</b>", "<b>Youtube Subscribers</b>", "<b>Youtube Subscribers cumulative sum<b>", "<b>Episode Duration (s)</b>"))



fig.append_trace(go.Scatter(name='youtube views', x=df_episodes.episode_id, y=df_episodes.youtube_views), row=1, col=1),

fig.append_trace(go.Scatter(name='youtube new subscribers', x=df_episodes.episode_id, y=df_episodes.youtube_subscribers), row=2, col=1),

fig.append_trace(go.Scatter(name='youtube subscribers cumulative sum', x=df_episodes.episode_id, y=df_episodes.yt_subs_cumulative), row=3, col=1)

fig.append_trace(go.Scatter(name='episode duration', x=df_episodes.episode_id, y=df_episodes.episode_duration), row=4, col=1)



fig.update_layout(height=1000, width=800, legend_orientation="h", plot_bgcolor='rgb(10,10,10)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
df_episodes['release_dofweek'] = df_episodes['release_date'].dt.dayofweek

dic = {'youtube_subscribers': ['sum'], 'episode_id': ['count'],'youtube_views': ['sum']}



df_t = df_episodes.groupby(['release_dofweek']).agg(dic).reset_index()

df_t.columns = ['_'.join(col) for col in df_t.columns.values]



df_t.loc[df_t['release_dofweek_'] == 0,'release_dofweek_'] = 'Monday', 

df_t.loc[df_t['release_dofweek_'] == 1,'release_dofweek_'] = 'Tuesday'

df_t.loc[df_t['release_dofweek_'] == 2,'release_dofweek_'] = 'Wednesday'

df_t.loc[df_t['release_dofweek_'] == 3,'release_dofweek_'] = 'Thursday'

df_t.loc[df_t['release_dofweek_'] == 4,'release_dofweek_'] = 'Friday'

df_t.loc[df_t['release_dofweek_'] == 5,'release_dofweek_'] = 'Saturday'

df_t.loc[df_t['release_dofweek_'] == 6,'release_dofweek_'] = 'Sunday'
fig = make_subplots(rows=1, cols=3, subplot_titles=("Episode Count", "Youtube subscribers", "Youtube views"))



colors = ['skyblue'] * 7

colors[3] = 'burlywood'

colors[6] = 'coral'



fig.append_trace(go.Bar(name='Episode Count', x=df_t.release_dofweek_, y=df_t.episode_id_count, showlegend=False, marker_color=colors), row=1, col=1),

fig.append_trace(go.Bar(name='Youtube subscribers', x=df_t.release_dofweek_, y=df_t.youtube_subscribers_sum, showlegend=False, marker_color=colors), row=1, col=2),

fig.append_trace(go.Bar(name='Youtube views', x=df_t.release_dofweek_, y=df_t.youtube_views_sum, showlegend=False, marker_color=colors), row=1, col=3),



fig.update_layout(barmode='stack', height=400, plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()


df_episodes['release_month'] = df_episodes['release_date'].dt.month



temp = df_episodes.drop(df_episodes[df_episodes['episode_id'] == 'E27'].index)

dfr = temp.groupby(['release_dofweek', 'release_month'])['youtube_views'].mean().reset_index()



dfr.loc[dfr['release_dofweek'] == 0,'release_dofweek'] = 'Monday', 

dfr.loc[dfr['release_dofweek'] == 1,'release_dofweek'] = 'Tuesday'

dfr.loc[dfr['release_dofweek'] == 2,'release_dofweek'] = 'Wednesday'

dfr.loc[dfr['release_dofweek'] == 3,'release_dofweek'] = 'Thursday'

dfr.loc[dfr['release_dofweek'] == 4,'release_dofweek'] = 'Friday'

dfr.loc[dfr['release_dofweek'] == 5,'release_dofweek'] = 'Saturday'

dfr.loc[dfr['release_dofweek'] == 6,'release_dofweek'] = 'Sunday'



fig = go.Figure(data=go.Heatmap(x = dfr.release_dofweek, y=dfr.release_month,z=dfr.youtube_views, colorbar = dict(title='youtube_views'), 

                                hovertemplate='Day of week: %{x}<br>Release Month: %{y}<br>Youtube views: %{z:.0f}<br><extra></extra>',

                                hoverongaps=False, colorscale='cividis'))

fig.update_layout(title_text="<b>Release Day and Month effect on youtube views<b>")

fig.update_xaxes(showgrid=False, zeroline=False, title='<b>Release Day</b>')

fig.update_yaxes(showgrid=False, zeroline=False, title='<b>Episode Release Month</b>')

fig.show()
print('\033[33m' + 'Episodes that had missing values in heroes column' + '\033[0m')

df_episodes[df_episodes['heroes'].isnull()]
df = df_episodes.groupby('heroes_gender').agg({'episode_id':'size', 'youtube_views':'mean'}).reset_index()

fig = make_subplots(rows=1, cols=2, x_title='Episodes Featuring', subplot_titles=("<b>Episodes count</b>", "<b>Avg youtube views</b>"))



colors = ['mediumaquamarine'] * 2

colors[0] = 'teal'



fig.append_trace(go.Bar(name='Episodes count', x=df.heroes_gender, y=df.episode_id, showlegend=False, marker_color=colors), row=1, col=1),

fig.append_trace(go.Bar(name='avg youtube views', x=df.heroes_gender, y=df.youtube_views, showlegend=False, marker_color=colors), row=1, col=2),



fig.update_layout(barmode='stack', height=400, width=800, legend_orientation="h", plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
fig, ax = plt.subplots(1,3, figsize = (20,6), sharex=True)

sns.countplot(x='category',data=df_episodes, palette="copper", ax=ax[0])

sns.countplot(x='category',hue='heroes_gender', palette="ocean", data=df_episodes,ax=ax[1])

sns.countplot(x='category',hue='recording_time', palette="cubehelix", data=df_episodes,ax=ax[2])

ax[0].title.set_text('Category count')

ax[1].title.set_text('Category Vs Gender')

ax[2].title.set_text('Category Vs Recording Time')

plt.show()
df_tmp = df_episodes.sort_values(by='heroes')

fig = px.bar(df_tmp, x='heroes', y='youtube_views', color='youtube_views',color_continuous_scale=["red", "green", "blue", "yellow"],

              title = '<b>Heroes Vs Youtube Views</b>', height=500)

mean_v = df_episodes['youtube_views'].mean()

fig.add_shape(type="line", x0='Abhishek Thakur', y0=mean_v, x1='Zachary Mueller', y1=mean_v, name='avg',line=dict(color='white', width=3, dash='dot'),)

fig.update_layout(height=700, plot_bgcolor='rgb(220,220,220)')

fig.update_xaxes(showgrid=False, zeroline=False, title = 'Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title = 'Youtube views')

fig.show()
df_tmp = df_episodes.sort_values(by='youtube_views',ascending=False)





fig = go.Figure(data=[

    go.Bar(name='Spotify listeners', x=df_tmp.heroes, y=df_tmp.spotify_listeners, marker_color='rgb(0, 102, 57)'),

    go.Bar(name='apple listeners', x=df_tmp.heroes, y=df_tmp.apple_listeners, marker_color='rgb(255, 128, 0)')

])

fig.update_layout(barmode='stack', height=700, title='<b>heroes vs spotify-apple listeners</b>', legend=dict(x=-.1, y=1.5),plot_bgcolor='rgb(220,220,220)' )

fig.update_xaxes(showgrid=False, zeroline=False, title = 'Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title='# of Listeners')

fig.show()
print('\033[33m' + 'Average Stats grouped by heroes ' + '\033[0m')

df_tmp = df_episodes[['episode_duration','heroes','youtube_views','spotify_streams','spotify_listeners','apple_listeners']].sort_values(by='episode_duration', ascending=True)

df_tmp.fillna('host').groupby(['heroes']).mean().sort_values(by='episode_duration', ascending=True).head()
print("Total videos that had more than avg youtube views:", len(df_episodes[df_episodes['youtube_views'] > 513]))
path = '../input/ctdsshow-addn-data/'

df_ctds = pd.read_csv(f'{path}CTDS_Addn_Data.csv')

df_ctds_addn = pd.merge(df_ctds,df_episodes, how='left',on='episode_id')
fig = go.Figure(data=[

    go.Scatter(name='Heroes Twitter Followers', x=df_ctds_addn.episode_id, y=df_ctds_addn.heroes_twitter_followers, text=df_ctds_addn.heroes_x, mode='markers+lines',marker_color='rgb(0, 102, 57)'),

    go.Scatter(name='Youtube Views', x=df_ctds_addn.episode_id, y=df_ctds_addn.youtube_views, text=df_ctds_addn.heroes_x, mode='markers+lines',marker_color='rgb(255, 128, 0)'),

])

fig.update_layout(legend=dict(x=-.1, y=1.5),height=600, plot_bgcolor='rgb(255,250,250)' )

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
fig = go.Figure(data=[

    go.Scatter(name='yt impression_views', x=df_episodes.heroes, y=df_episodes.youtube_impression_views, mode='markers+lines', marker_color='mediumaquamarine'),

    go.Scatter(name='yt non impression_views', x=df_episodes.heroes, y=df_episodes.youtube_nonimpression_views, mode='markers+lines', marker_color='peru')

])

fig.update_layout(legend=dict(x=-.1, y=1.5), height=600, plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.update_yaxes(title='# of views')

fig.show()
print("Total Youtube Impression views:", df_episodes.youtube_impression_views.sum())

print("Total Youtube Non Impression views:", df_episodes.youtube_nonimpression_views.sum())
df = df_episodes.groupby('youtube_thumbnail_type').agg({'youtube_views':'sum', 'episode_id':'count','youtube_impression_views': 'sum','youtube_nonimpression_views': 'sum'}).reset_index()

fig = make_subplots(rows=1, cols=4, x_title='Thumbnail Type', column_titles=("Episodes count","youtube - views", "impression views" , "nonimpression views"))



colors = ['lightgrey'] * 4

colors[0] = 'teal'

colors[1] = 'skyblue'

colors[2] = 'burlywood'

colors[3] = 'coral'



fig.append_trace(go.Bar(name='# of episodes', x=df.youtube_thumbnail_type, y=df.episode_id, showlegend=False, marker_color=colors), row=1, col=1),

fig.append_trace(go.Bar(name='youtube views', x=df.youtube_thumbnail_type, y=df.youtube_views, showlegend=False, marker_color=colors), row=1, col=2),

fig.append_trace(go.Bar(name='youtube impression views', x=df.youtube_thumbnail_type, y=df.youtube_impression_views, showlegend=False, marker_color=colors), row=1, col=3),

fig.append_trace(go.Bar(name='youtube nonimpression views', x=df.youtube_thumbnail_type, y=df.youtube_nonimpression_views, showlegend=False, marker_color=colors), row=1, col=4),



fig.update_layout(barmode='stack', font=dict(size=10), legend_orientation="h", height=400, plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
fig = go.Figure(data=[

    go.Scatter(name='Youtube Views - Jun20', x=df_ctds_addn.episode_id, y=df_ctds_addn.youtube_views_Jun20, text=df_ctds_addn.heroes_x, marker_color='rgb(255, 128, 0)'),

    go.Scatter(name='Youtube Views - Jul13', x=df_ctds_addn.episode_id, y=df_ctds_addn.youtube_views_Jul13,text=df_ctds_addn.heroes_x, marker_color='rgb(102, 178, 255)')

])

fig.update_layout(legend=dict(x=-.1, y=1.5),height=600, plot_bgcolor='rgb(255,255,250)' )

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title='Youtube Views')

fig.show()
yt_jun20 = df_ctds_addn['youtube_views_Jun20'].sum()

yt_jul13 = df_ctds_addn['youtube_views_Jul13'].sum()

print(f'Youtube views until Jun 20: {yt_jun20} and youtube views until Jul13: {yt_jul13}')

print(f'percentage increase of youtube views since contest started: {round(((yt_jul13-yt_jun20)/yt_jun20)*100,2)}%')
fig = px.scatter(df_episodes, x = df_episodes.episode_id, y=df_episodes.flavour_of_tea, title='<b>Flavors of Tea across Episodes</b>', color=df_episodes.flavour_of_tea)

fig.update_layout(plot_bgcolor='rgb(0,0,0)', xaxis={'categoryorder':'category ascending'}, showlegend=False)

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title='Flavour of Tea')

fig.data[0].update(marker_symbol='circle', marker_size=8)

fig.data[1].update(marker_symbol='diamond', marker_size=8)

fig.data[2].update(marker_symbol='pentagon', marker_size=7)

fig.data[3].update(marker_symbol='hexagon', marker_size=7)

fig.data[4].update(marker_symbol='octagon', marker_size=7)

fig.data[5].update(marker_symbol='star', marker_size=7)

fig.data[6].update(marker_symbol='cross', marker_size=7)

fig.data[7].update(marker_symbol='cross-dot', marker_size=7)

fig.data[8].update(marker_symbol='triangle-down', marker_size=7)

fig.show()
data = [dict(type = 'bar',x = df_episodes.flavour_of_tea, y = df_episodes.youtube_views, mode = 'markers',marker = dict(color = 'coral'),

             transforms = [dict(type = 'aggregate',groups = df_episodes.flavour_of_tea,

                                aggregations = [dict(target = 'y', func = 'avg', enabled = True),])])]



layout = dict(title = '<b>Tea Flavour vs Mean Youtube views<b>',xaxis = dict(title = 'Tea Flavour'),yaxis = dict(title = 'Mean Youtube views'))



fig_dict = dict(data=data,layout=layout)



pio.show(fig_dict, validate=False)
dff = df_episodes.groupby(['recording_time', 'flavour_of_tea'])['youtube_views'].mean().reset_index()

fig = go.Figure(data=go.Heatmap(x = dff.flavour_of_tea, y=dff.recording_time,z=dff.youtube_views, colorbar = dict(title='youtube_views'), 

                                hovertemplate='flavour_of_tea: %{x}<br>recording_time: %{y}<br>youtube_views: %{z:.0f}<extra></extra>', 

                                hoverongaps=False, colorscale='Viridis'))

fig.update_xaxes(showgrid=False, zeroline=False, title='<b>Flavour of tea</b>')

fig.update_yaxes(showgrid=False, zeroline=False, title='<b>Recording Time</b>')

fig.show()
sub_path = '../input/chai-time-data-science/Cleaned Subtitles/'
def questions(df):

    df_qt = df[df['Text'].str.contains("\?") & df['Speaker'].str.contains("Sanyam Bhutani")]

    df_ttemp = df_qt['Text'].copy()

    return df_ttemp
nlp = spacy.load('en', entity=False)
def word_count(df,e):

    df['tokens'] = df['Text'].apply(lambda x: nlp(x))

    df['Word_count'] = [len(token) for token in df.tokens]

    df_t = df.groupby(['Speaker'])['Word_count'].sum().reset_index()

    df_t['Episode'] = e

    return df_t
def q_count(df):

    df_qt = df[df['Text'].str.contains("\?") & df['Speaker'].str.contains("Sanyam Bhutani")]

    length = len(df_qt)

    return length
def c_count(df,e):

    df_ct = df.groupby('Speaker').agg({'char_count':'sum'}).reset_index()

    df_ct['episode_id'] = e

#     length = len(df_qt)

    return df_ct
!pip install natsort
ss_list = []

for f_name in os.listdir(f'{sub_path}'):

    ss_list.append(f_name)
from natsort import natsorted

s_list = natsorted(ss_list)
df_qct = pd.DataFrame(columns=['episode', 'q_count'])

for i in range(len(s_list)):

    Episodes = pd.read_csv(f'{sub_path}'+s_list[i])

    ep_id = s_list[i].split('.')[0]

    get_df = q_count(Episodes)

    df_qct = df_qct.append({'episode': ep_id,'q_count': get_df}, ignore_index=True)
df_lct = pd.DataFrame(columns=['episode_id', 'Speaker','char_count'])

for i in range(len(s_list)):

    Episodes = pd.read_csv(f'{sub_path}'+s_list[i])

    ep_id = s_list[i].split('.')[0]

    Episodes['char_count'] = Episodes['Text'].apply(len)

    get_df = c_count(Episodes,ep_id)

    df_lct = df_lct.append(get_df, ignore_index=True)
# Getting questions

df_qs = pd.DataFrame(columns=['episode', 'questions'])

for i in range(len(s_list)):

    Episodes1 = pd.read_csv(f'{sub_path}'+s_list[i])

    ep_id = s_list[i].split('.')[0]

    get_df1 = questions(Episodes1)

    df_qs = df_qs.append({'episode': ep_id,'questions': get_df1}, ignore_index=True)
df_lct['speaker_g'] = df_lct['Speaker'].map({'Sanyam Bhutani': 'Host'})

df_lct["speaker_g"].fillna("Heroes", inplace = True)
fig = go.Figure(data = go.Scatter(x=df_qct.episode, y=df_qct.q_count, mode='markers+lines'))

fig.update_layout(title = '<b>Episodes Vs # of Questions</b>', height=600, width=900, xaxis_title="Episodes", yaxis_title="# of questions",legend_orientation="h", plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
pd.set_option('max_colwidth', None)
df_qs['questions'] = df_qs['questions'].map(str)

df_qs['AMA'] = df_qs['questions'].str.contains(" AMA ")
df_qs['episode_id'] = df_qs['episode']

df_ama = pd.merge(df_episodes,df_qs, how='left',on='episode_id')

df_ama['AMA'].fillna(False, inplace=True)
fig = px.scatter(df_ama, x="episode_id", y="youtube_views",size="youtube_views", color="AMA", size_max=40)

fig.update_layout(legend_orientation="h", plot_bgcolor='rgb(240,240,240)')

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title='Youtube views')

fig.update_layout(title_text="<b>AMA Presence in Episodes & Youtube Views<b>")

fig.show()
#Duration calculation

df_dur = pd.DataFrame(columns=['episode', 'intro_duration'])

for i in range(len(s_list)):

    Episodes = pd.read_csv(f'{sub_path}'+s_list[i])

    Episodes['Duration_Sec'] = Episodes['Time'].str.split(':').apply(lambda t: int(t[0]) * 60 + int(t[1]))

    ep_id = s_list[i].split('.')[0]

    intro_time = Episodes['Duration_Sec'][1]

#     get_df = q_count(Episodes)

    df_dur = df_dur.append({'episode': ep_id,'intro_duration': intro_time}, ignore_index=True)
fig = make_subplots(rows=2, cols=1, subplot_titles=("<b>Episode Intro Duration (sec)</b>", "<b>Youtube Views</b>"))



fig.append_trace(go.Scatter(name='<b>Episode Intro Duration </b>', x=df_dur.episode, y=df_dur.intro_duration, marker_color='rgb(0, 102, 57)'), row=1, col=1),

fig.append_trace(go.Scatter(name='<b>Youtube Views</b>', x=df_episodes.episode_id, y=df_episodes.youtube_views, marker_color='rgb(0, 76, 153)'), row=2, col=1),



fig.update_layout(height=800, width=900, legend_orientation="h", plot_bgcolor='rgb(255,255,255)')

fig.update_xaxes(showgrid=False, zeroline=False)

fig.update_yaxes(showgrid=False, zeroline=False)

fig.show()
yy = df_lct[df_lct['speaker_g'].str.contains("Host")]

fig = px.bar(yy, x = yy.episode_id, y=yy.char_count, title='<b>Episodes Vs Conversation Text length of Host</b>')

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title= 'Host - Text Length')

fig.update_layout(plot_bgcolor='rgb(250,250,250)')

fig.show()
tea = pd.merge(df_episodes,yy, how='inner',on='episode_id')

tea['char_count'] = tea['char_count'].astype(str).astype(int)

tt = tea.groupby("flavour_of_tea").agg({"char_count": [np.mean, np.sum]})

tt.columns = ['_'.join(col) for col in tt.columns.values]

display(tt)
# Most Viewed Episodes

top_ed = ['E27.csv', 'E49.csv', 'E1.csv', 'E33.csv', 'E38.csv', 'E26.csv', 'E60.csv', 'E35.csv', 'E34.csv', 'E25.csv']



df_wordct_t = pd.DataFrame()

for i in range(len(top_ed)):

    ep_id = top_ed[i].split('.')[0]

    Episodes = pd.read_csv(f'{sub_path}'+top_ed[i])

    content = word_count(Episodes,ep_id)

    df_wordct_t = df_wordct_t.append(content, ignore_index=True)
# Least Viewed Episodes

least_ed = ['E14.csv', 'E20.csv', 'E7.csv', 'E3.csv', 'E16.csv', 'E10.csv', 'E12.csv', 'E2.csv', 'E8.csv']



df_wordct_l = pd.DataFrame()

for i in range(len(least_ed)):

    ep_id = least_ed[i].split('.')[0]

    Episodes = pd.read_csv(f'{sub_path}'+least_ed[i])

    content = word_count(Episodes,ep_id)

    df_wordct_l = df_wordct_l.append(content, ignore_index=True)
fig = px.bar(df_wordct_t, x='Episode', y='Word_count', color='Speaker', color_discrete_sequence=px.colors.qualitative.Prism,

              title = '<b>Top Viewed Episodes - word count host Vs heros</b>', height=500, width=800)

fig.update_xaxes(showgrid=False, zeroline=False, title='Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title = 'Host - word count')

fig.update_layout(plot_bgcolor='rgb(0,0,0)')

fig.show()
fig = px.bar(df_wordct_l, x='Episode', y='Word_count', color='Speaker',color_discrete_sequence=px.colors.qualitative.Prism,

              title = '<b>Least Viewed Episodes - word count host Vs heros</b>', height=500, width=800)

fig.update_xaxes(showgrid=False, zeroline=False, title = 'Episodes')

fig.update_yaxes(showgrid=False, zeroline=False, title = 'Host - word count')

fig.update_layout(plot_bgcolor='rgb(0,0,0)')

fig.show()