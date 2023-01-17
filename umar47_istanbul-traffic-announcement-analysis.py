import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px #visualization

import chart_studio.plotly as py #visualization

import plotly.figure_factory as ff #visualization

import plotly.graph_objs as go #visualization

from wordcloud import WordCloud, STOPWORDS #wordcloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/Trafik - Trafik Duyurular.csv')

df.info()
df=df.rename(columns={'Lon': 'Lat', 'Lat': 'Lon'})

df['EndDate']=pd.to_datetime(df['EndDate'])

df['ResponseDate']=pd.to_datetime(df['ResponseDate'])

df['EntryDate']=pd.to_datetime(df['EntryDate'])

df.info()
df.sample(5)
z=(df['EntryDate'].where(df['EntryDate']<'2019-12-31')).dt.week.value_counts().sort_index()

fig = px.line(x=z.index, y=z.values)

fig.show()
event_dist1=df['AnnouncementType'].value_counts()

fig = px.bar(x=event_dist1.values, y=event_dist1.index, orientation='h', title="AnnouncementType")

fig.show()
df['response_time']= (df['ResponseDate'] - df['EntryDate']).astype('timedelta64[h]')

df['weekday']=df['EntryDate'].dt.weekday_name

df['month']=df['EntryDate'].dt.month_name()

df['hour']=df['EntryDate'].dt.hour
pivot=pd.DataFrame(df.groupby(['month', 'AnnouncementType']).size().unstack().reset_index())



fig = go.Figure()

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Alt Yapı Çalışması'], name='Alt Yapı Çalışması'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Araç Arızası'], name='Araç Arızası'), )

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Bakım-Onarım Çalışması'], name='Bakım-Onarım'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Haber'], name='Haber'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Kaza Bildirimi'], name='Kaza Bildirimi'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Yolu Etkileyen Hava Koşulu'], name='Yolu Etkileyen Hava Koş.'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Yolun Trafiğe Kapanması'], name='Yolun Trafiğe Kapanması'))

fig.add_trace(go.Scatter(x=pivot['month'], y=pivot['Çevre Düzenlemesi'], name='Çevre Düzenlemesi' ))

fig.update_layout(

    title="Announcement Type's Monthly distributions",

    xaxis_title="Month",

    yaxis_title="# of Announcement",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.show()
pivot=pd.DataFrame(df.groupby(['weekday', 'AnnouncementType']).size().unstack().reset_index())



fig = go.Figure()

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Alt Yapı Çalışması'], name='Alt Yapı Çalışması'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Araç Arızası'], name='Araç Arızası'), )

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Bakım-Onarım Çalışması'], name='Bakım-Onarım'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Haber'], name='Haber'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Kaza Bildirimi'], name='Kaza Bildirimi'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Yolu Etkileyen Hava Koşulu'], name='Yolu Etkileyen Hava Koş'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Yolun Trafiğe Kapanması'], name='Yolun Trafiğe Kapanması'))

fig.add_trace(go.Scatter(x=pivot['weekday'], y=pivot['Çevre Düzenlemesi'], name='Çevre Düzenlemesi' ))

fig.update_layout(

    title="Announcement Type's Daily distributions",

     xaxis_title="WeekDay",

    yaxis_title="# of Announcement",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)

fig.show()
Accidents_Announcements=df['location'].where(df['AnnouncementType']=='Kaza Bildirimi').value_counts()

fig = px.bar(x=Accidents_Announcements.values[:10], y=Accidents_Announcements.index[:10], orientation='h', title="Locations with the most accident announcement")

fig.show()
month_dist=df['month'].value_counts()

fig = px.bar(x=month_dist.index, y=month_dist.values, title="Announcement's month distribution")

fig.show()
weekday_dist=df['weekday'].value_counts()

fig = px.bar(x=weekday_dist.index, y=weekday_dist.values, title="Announcement's weekday distribution")

fig.show()
hour_dist=df['hour'].value_counts()

fig = px.bar(x=hour_dist.index, y=hour_dist.values, title="Announcement's hourly distribution")

fig.show()
def plotly_wordcloud(text):

    wc = WordCloud(stopwords = set(STOPWORDS),

                   max_words = 200,

                   max_font_size = 100)

    wc.generate(text)

    

    word_list=[]

    freq_list=[]

    fontsize_list=[]

    position_list=[]

    orientation_list=[]

    color_list=[]



    for (word, freq), fontsize, position, orientation, color in wc.layout_:

        word_list.append(word)

        freq_list.append(freq)

        fontsize_list.append(fontsize)

        position_list.append(position)

        orientation_list.append(orientation)

        color_list.append(color)

        

    # get the positions

    x=[]

    y=[]

    for i in position_list:

        x.append(i[0])

        y.append(i[1])

            

    # get the relative occurence frequencies

    new_freq_list = []

    for i in freq_list:

        new_freq_list.append(i*100)

    new_freq_list

    

    trace = go.Scatter(x=x, 

                       y=y, 

                       textfont = dict(size=new_freq_list,

                                       color=color_list),

                       hoverinfo='text',

                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],

                       mode='text',  

                       text=word_list

                      )

    

    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},

                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

  



    fig = go.Figure(data=[trace], layout=layout)

    fig.update_layout(

    autosize=False,

    width=1000,

    height=500,

    margin=go.layout.Margin(

        l=10,

        r=10,

        b=10,

        t=10,

        pad=4

    ),

    paper_bgcolor="white",

)

    return fig



text1 = " ".join(str(texts) for texts in df.AnnouncementText.values)

plotly_wordcloud(text1)
Accidents_Announcements_1=df.where(df['AnnouncementType']=='Kaza Bildirimi')

fig = px.scatter_mapbox(Accidents_Announcements_1, lat="Lat", lon="Lon",  hover_name="AnnouncementTitle",

                           color_discrete_sequence=["fuchsia"], zoom=8, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})

fig.show()
x=Accidents_Announcements_1.groupby('location')['response_time', 'Lat', 'Lon'].mean()

location_value_counts=df['location'].value_counts().fillna(0)

average_response_time=pd.concat([location_value_counts, x.reindex(location_value_counts.index)], axis=1)



fig = px.scatter_mapbox(average_response_time, lat="Lat", lon="Lon",  hover_name="response_time",

                           color_discrete_sequence=["fuchsia"], zoom=8, height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":1,"l":0,"b":0})



fig.show()
AccidentSituation=Accidents_Announcements_1['AccidentSituation'].value_counts()

fig = px.bar(x=AccidentSituation.index, y=AccidentSituation.values, title="What has done after Accidents Announcements?")

 

fig.update_layout(

    autosize=True,

    margin=go.layout.Margin(

        r=40,

        b=130,

        t=40,

        pad=4

    ),

    paper_bgcolor="white",

)

fig.show()