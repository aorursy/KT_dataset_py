import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df2 = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()
#まとめる

'''
df2['daily_excisting']=df2['confirmed'].values - df['Deaths'].diff() - df2['Recovered'].diff()
#残りの感染者の出し方(上のはcumelative)
'''

fig = px.choropleth(df2,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Rainbow',range_color=(0.1,100000.))
#映像化
fig.update_layout(title_text='Yu Okui',title_x=0.5)
#レイアウトのアップデート,title_x=0.5 ->centering
fig.show()
df = pd.read_csv("../input/20150425/20150425.csv",header=0)
df.index = pd.to_datetime(df['time'])
#convert index to datetime
df['time']= df.index.strftime('%Y-%m-%d %H:00:00')
#同じ時間に設定しなおし
fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(5.,7.))
fig.update_layout(title_text='Yu Okui',title_x=0.5)
fig.show()