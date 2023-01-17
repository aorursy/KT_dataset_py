import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

data=pd.read_csv('../input/windows-store/msft.csv')
data.head()
data.describe()
df=data['Category'].value_counts()
df=pd.DataFrame(df)
df=df.rename(columns={'Category': 'Count'})
df.reset_index(inplace=True)
df=df.rename(columns={'index': 'Category'})
df
plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='Category',
    y='Count',
    palette='Set1'
)
chart=chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
df=data[data['Rating']==data['Rating'].max()].groupby('Category')['Name'].count()
df=pd.DataFrame(df)
df=df.rename(columns={'Name': 'No.of apps having 5* Rating'})
df.reset_index(inplace=True)
df=df.rename(columns={'index': 'Category'})
df=df.sort_values('No.of apps having 5* Rating',ascending=False).head(10)
plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='Category',
    y='No.of apps having 5* Rating',
    palette='Set1',
)
chart=chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
def fn(x):
    if x!='Free' and x!='NaN':
        #print(x)
        x=str(x)
        if '₹' in x:
            x=x.split('₹')[1]
        if ',' in x:       
            x,y=x.split(',')
            x=x+y
        x=float(x)
        return x
    else:
        return 0
data['Price']=data['Price'].apply(lambda x: fn(x))
df=data.sort_values(['Price'],ascending=False)[['Name','Price']]
df=df[df['Price']>=1000.0]

plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='Name',
    y='Price'
)
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
chart.axes.yaxis.label.set_text("Apps having price greater than 1000")
data['Date']=pd.to_datetime(data['Date'])
data=data.sort_values('Date')
df=data.groupby(data['Date'].dt.year)['Rating'].mean()
df=pd.DataFrame(df).reset_index().rename(columns={'Date': 'year'})
df
fig = px.line(df, x='year', y='Rating')
fig.show()
df=data.groupby('Category')['No of people Rated'].sum()
df=pd.DataFrame(df).reset_index().sort_values(['No of people Rated'],ascending=False)
plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='Category',
    y='No of people Rated'
)
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
chart.axes.yaxis.label.set_text("Number of people Rated by Category")
fig = px.pie(df, values='No of people Rated', names='Category')
fig.show()
quality=[]
for r in data['Rating']:
    if r>=4.0:
        quality.append('Good')
    elif r>=3.5 and r<4.0:
        quality.append('Average')
    else:
        quality.append('Poor')
data['app_quality']=quality
df=data.groupby('app_quality')['Name'].count()
df=pd.DataFrame(df).reset_index().rename(columns={'Name': 'Count'})
plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='app_quality',
    y='Count',
    palette='Set1',
)
chart=chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
Category=data['Category'][~pd.isnull(data['Category'])]
wordCloud = WordCloud(width=450,height= 300).generate(' '.join(Category))
plt.figure(figsize=(19,9))
plt.axis('off')
plt.title(data['Category'].name,fontsize=20)
plt.imshow(wordCloud)
plt.show()
df=data[data['Price']!=0.0].sort_values('Rating')
sns.scatterplot(df['Rating'],df['Price'])
df=data.groupby(data['Date'].dt.year)['No of people Rated'].sum()
df=pd.DataFrame(df).reset_index().rename(columns={'Date': 'year'})
df=df.rename(columns={'Name': 'No of people Rated'})
df.year=df.year.astype('int')

plt.figure(figsize=(10,5))
chart = sns.barplot(
    data=df,
    x='year',
    y='No of people Rated'
)
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=65, 
    horizontalalignment='right',
    fontweight='light',
 
)
chart.axes.yaxis.label.set_text("No. of people rated each year")
good=data.groupby('app_quality').get_group('Good')
good=good.groupby('Category')['Name'].count()

average=data.groupby('app_quality').get_group('Average')
average=average.groupby('Category')['Name'].count()

poor=data.groupby('app_quality').get_group('Poor')
poor=poor.groupby('Category')['Name'].count()

fig2 = go.Figure(
    data=[
        go.Bar(
            name="good",
            x=good.index,
            y=good.values,
            offsetgroup=0,
            marker_color='green',
        ),
        go.Bar(
            name="average",
            x=average.index,
            y=average.values,
            offsetgroup=1,
            marker_color='orange'
        ),
        go.Bar(
            name="poor",
            x=poor.index,
            y=poor.values,
            offsetgroup=2,
            marker_color='red',
        )
    ],
    layout=go.Layout(
        title="Good , average , poor rating category wise",
        yaxis_title="Number of apps in a category"
    )
)
fig2.show()

count=data.groupby(data['Date'].dt.year)['Name'].count()

fig = go.Figure(data=go.Scatter(x=data['Date'].dt.year.unique(),
                                y=count.values,
                                mode='lines',
                               marker_color='darkblue')) 

fig.update_xaxes(rangeslider_visible=True) # Range Slider is made true

fig.update_layout(title='No of apps yearly',xaxis_title="Date",yaxis_title="count of apps ")
fig.show()