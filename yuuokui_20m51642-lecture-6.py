import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)#printで全データ観れる

selected_country ='Sweden'#set the country
date ='06/12/2020'#set the date before yesterday('mm/dd/yyyy')

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))#同じ国同士まとめて全て表示。ここから国を選ぶ。
df_coun = df[df['Country/Region']==selected_country]
df_coun_cases = df_coun.groupby('ObservationDate').sum()
#groupby().sumで同じ()のデータを合計してまとめる。
#print(df_coun_cases)
df_coun_cases['daily_confirmed'] = df_coun_cases['Confirmed'].diff()
df_coun_cases['daily_deaths'] = df_coun_cases['Deaths'].diff()
df_coun_cases['daily_recovery'] = df_coun_cases['Recovered'].diff()
#新しいcolumnとして追加
df_coun_cases['daily_confirmed'].plot()
df_coun_cases['daily_recovery'].plot()
#plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df_coun_cases.index,y=df_coun_cases['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df_coun_cases.index,y=df_coun_cases['daily_deaths'].values,name='Daily deaths')

layout_object = go.Layout(title='{0} daily cases 20M51642'.format(selected_country),xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('{0}_daily_cases_20M51642.html'.format(selected_country))
df1 = df_coun_cases
df1 = df1.fillna(0.)#NA -->  0.0
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('daily Summaries')
#gist_ncarはcolormap python で検索して決めたカラーバンド。
display(styled_object)
f = open('table_20M51642.html','w')
f.write(styled_object.render())
df2 = df.groupby(['Country/Region','ObservationDate'],as_index=False).sum()
#df2.groups
sort_confirmed = df2[df2['ObservationDate']==date].sort_values(by=['Confirmed'],ascending=False).reset_index()
print('Ranking of {0}: '.format(selected_country), sort_confirmed[sort_confirmed['Country/Region']==selected_country].index.values[0]+1)
recent7 = df1.tail(7)['daily_confirmed'].sum()
recent14 = df1.tail(14)['daily_confirmed'].sum()
if recent14 - recent7 < recent7:
    print('Increasing')
elif recent14 - recent7 == recent7:
        print('Constant')
else:
        print('Decreasing')
df3 = df1[df1['daily_confirmed']==df1['daily_confirmed'].max()]
print('Data when daily cases reached the largest\n\n',df3)