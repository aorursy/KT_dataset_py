# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')



# Any results you write to the current directory are saved as output.
#üniversitelerin dünya genelindeki sıralaması

timesData=pd.read_csv('../input/timesData.csv')
timesData.head(10)
timesData.info()
timesData.columns
df=timesData.iloc[:100,:]

import plotly.graph_objs as go

trace1=go.Scatter(x=df.world_rank,

                 y=df.citations,

                 mode="lines",

                 name="citations",

                 marker=dict(color='rgba(16,122,250,0.8)'),

                 text=df.university_name)

trace2=go.Scatter(x=df.world_rank,

                 y=df.teaching,

                 mode="lines",

                 name="teaching",

                 marker=dict(color='rgba(163,15,42,0.8)'),

                 text=df.university_name)

data=[trace1,trace2]

layout=dict(title='Citation and Teaching vs World Rank Of Top 100 Universities',

           xaxis=dict(title='World Rank',ticklen=5,zeroline=False))

fig = dict(data=data,layout=layout)

iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:100,:]

df2015=timesData[timesData.year==2015].iloc[:100,:]

df2016=timesData[timesData.year==2016].iloc[:100,:]

import plotly.graph_objs as go

trace1=go.Scatter(x=df2014.world_rank,

                 y=df2014.citations,

                 mode="markers",

                 name="2014",

                 marker=dict(color='rgba(165,0,165,0.8)'),

                 text=df2014.university_name)

trace2=go.Scatter(x=df2015.world_rank,

                 y=df2015.citations,

                 mode="markers",

                 name="2015",

                 marker=dict(color='rgba(23,45,85,0.8)'),

                 text=df2015.university_name)

trace3=go.Scatter(x=df2016.world_rank,

                 y=df2016.citations,

                 mode="markers",

                 name="2016",

                 marker=dict(color='rgba(49,150,56,0.8)'),

                 text=df2014.university_name)

data=[trace1,trace2,trace3]

layout=dict(title='Citation vs worls rank of top 100',

           xaxis=dict(title='World Rank',ticklen=5,zeroline=False),

           yaxis=dict(title='Citation',ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:100,:]

df2015=timesData[timesData.year==2015].iloc[:100,:]

df2016=timesData[timesData.year==2016].iloc[:100,:]

trace1=go.Scatter(x=df2014.world_rank,

                 y=df2014.total_score,

                 name="2014",

                 mode="markers",

                 marker=dict(color='rgba(165,15,25,0.8)'),

                 text=df2014.university_name)

trace2=go.Scatter(x=df2015.world_rank,

                 y=df2015.total_score,

                 name="2015",

                 mode="markers",

                 marker=dict(color='rgba(12,75,245,0.7)'),

                 text=df2015.university_name)

trace3=go.Scatter(x=df2016.world_rank,

                 y=df2016.total_score,

                 name="2016",

                 text=df2016.university_name,

                 mode="markers",

                 marker=dict(color='rgba(120,250,33)'))

data=[trace1,trace2,trace3]

layout=dict(title="Total Score vs Worlds Rank of top 100",

           xaxis=dict(title="World Rank",ticklen=5,zeroline=False),

           yaxis=dict(title="Total Score",ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)

df2014=timesData[timesData.year==2014].iloc[:3,:]

import plotly.graph_objs as go

trace1=go.Bar(x=df2014.university_name,

              y=df2014.citations,

              name="citations",

              marker=dict(color='rgba(255,190,230,0.8)'),

              text=df2014.country)

trace2=go.Bar(x=df2014.university_name,

              y=df2014.teaching,

              name="teaching",

              marker=dict(color='rgba(200,150,255)',

                         line=dict(color='rgba(0,0,0,0.8)',width=1.5)),

              text=df2014.country)

data=[trace1,trace2]

#barmode=group yanyana yap sütunları

layout=go.Layout(barmode="group")

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df2014=timesData[timesData.year==2014].iloc[:3,:]

x=df2014.university_name

trace1={

    'x':x,

    'y':df2014.citations,

    'name':'citations',

    'type':'bar'

};

trace2={

    'x':x,

    'y':df2014.teaching,

    'name':'Teaching',

    'type':'bar'

};

data=[trace1,trace2]

layout={

    'xaxis':{'title':'Top 3 University'},

    #barmode=relative altlı üstlü koy

    'barmode':'relative',

    'title':'citations and teaching of top 3 universities in 2014'

};

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df2016.info()
df2016=timesData[timesData.year==2016].iloc[:7,:]

pie1=df2016.num_students



pie1_list=[float(each.replace(',','.')) for each in df2016.num_students]

labels=df2016.university_name

fig={

    #Trace1=

    "data":[{

    "values":pie1_list,

    "labels":labels,

    #domain = grafiğin büyüklüğü

    "domain":{"x":[0, .8]},

    "name":"Number of Students Rates",

   #hoverinfo : oran+yüzde+isim

    "hoverinfo":"label+percent+name",

    #ortadaki beyaz deliğin büyüklüğü hole

    "hole": .3,

    "type":"pie"

    

}],#layout=

     "layout":{

         "title":"Universities Number of Students rates",

         "annotations":[

             {

                 "font":{"size":20}, 

                 # sharrow=false sol yukardaki number of students yazısı kaybolur

                 "showarrow":True,

                 "text":"Number of Students",

                 "x":0.20,

                 "y":1

                 

             },

         ]

     }

    }

iplot(fig)
df2016.info()

df2016=timesData[timesData.year==2016].iloc[:20,:]

num_students_size=[float(each.replace(',','.')) for each in df2016.num_students]

international_color=[float(each) for each in df2016.international]

data=[

    {

        'y':df2016.teaching,

        'x':df2016.world_rank,

        'mode':'markers',

        'marker':{

            'color':international_color,

            'size':num_students_size,

            'showscale':True

        },

        'text':df2016.university_name

        

    }

]

#layout kullanmadığımız için direkt datayı yazdırdık



iplot(data)

#1. üniversitenin gözükmeme sebebi number of studentin size ının çok az yada hiç olmaması
df2011=timesData.student_staff_ratio[timesData.year==2011]

df2012=timesData.student_staff_ratio[timesData.year==2012]

trace1=go.Histogram(

    x=df2011,

    name ='2011 student staff ratio',

    opacity=0.75,

    marker=dict(color='rgba(122,130,213,0.8)')

    )

trace2=go.Histogram(

    x=df2012,

    name ='2012 student staff ratio',

    opacity=0.75,

    marker=dict(color='rgba(231,15,150,0.8)')

    )

data=[trace1,trace2]

layout=go.Layout(barmode="overlay",

                title="students-staff ratio in 2011-2012")

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df2011=timesData.country[timesData.year==2014]

plt.subplots(figsize=(10,10))

wordcloud=WordCloud(background_color='white',

                   width=512,

                   height=384).generate("".join(df2011))

#.generate("".join(Df2011))= kelimeleri ayır en çok kullanılan kelimeleri

#büyük bir şekilde çizdir

plt.imshow(wordcloud) #image show

plt.axis('off') #x ve y ekseni çıkmasın diye off deriz

plt.savefig('graph.png')

plt.show()
df2015=timesData[timesData.year==2015]

trace1=go.Box(y=df2015.total_score,

             name='total score of universities in 2015',

             marker=dict(color='rgb(12,12,140)',))

trace2=go.Box(y=df2015.research,

             name='research of universities in 2015',

             marker=dict(color='rgb(12,128,128)',))

data=[trace1,trace2]

#layout yoksa figure e gerek yok

iplot(data)
import plotly.figure_factory as ff

dataframe=timesData[timesData.year==2015]

#karşılaştırmak istediğimiz özellikler "research","international","total_score"

#bunları dataframe e atıp da karşılaştırıyoruz.

data2015=dataframe.loc[:,["research","international","total_score"]]

#dataframe in uzunluğunda liste hazırlıyoruz dataframein indexine eşitliyoruz.

data2015["index"]=np.arange(1,len(data2015)+1)



fig=ff.create_scatterplotmatrix(data2015,diag='box',index='index',colormap='Portland',

                               colormap_type='cat',

                               height=700,width=700)

iplot(fig)
trace1=go.Scatter(

x=dataframe.world_rank,

y=dataframe.teaching,

name="teaching",

marker=dict(color='rgba(210,112,113,0.8)'),)



trace2=go.Scatter(

x=dataframe.world_rank,

y=dataframe.income,

xaxis='x2',

yaxis='y2',

name="income",

marker=dict(color='rgba(50,50,150,0.8)'),)



data=[trace1,trace2]



layout=go.Layout(

    xaxis2=dict(

        domain=[0.6,0.95],

        anchor='y2', ),

    yaxis2=dict(

        domain=[0.6,0.95],

        anchor='x2',),

    #2 plot iç içe olduğu için anchor kullandık

    #domain yerimizi belirtir

    #2.plotu çizmek için kullanılan şey = anchor

    title='Income and Teaching vs World Rank of Universities')



fig=go.Figure(data=data,layout=layout)



iplot(fig)
trace1=go.Scatter3d(

x=dataframe.world_rank,

y=dataframe.research,

z=dataframe.citations,

mode='markers',

marker=dict(

    size=10,

    color='rgba(155,155,155)')

)

data=[trace1]

layout=go.Layout(margin=dict(l=0,r=0,t=0,b=0))

fig=go.Figure(data=data,layout=layout)

iplot(fig)

    
trace1=go.Scatter(

x=dataframe.world_rank,

y=dataframe.research,

name="research")



trace2=go.Scatter(

x=dataframe.world_rank,

y=dataframe.citations,

xaxis='x2',

yaxis='y2',

name='citations')



trace3=go.Scatter(

x=dataframe.world_rank,

y=dataframe.income,

xaxis='x3',

yaxis='y3',

name='income')

trace4=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

xaxis='x4',

yaxis='y4',

name="total_score")

data=[trace1,trace2,trace3,trace4]

layout=go.Layout(

    # x axis domain = yataydaki yeri belirtr= 0. 0.45 arsında olsun 0.1 boşluk olsun 0.55 0.1 arasında olsun

    # y axis domain = dikeydeki yeri belirtr= 0. 0.45 arsında olsun 0.1 boşluk olsun 0.55 0.1 arasında olsun

xaxis=dict(domain=[0,0.45]),

yaxis=dict(domain=[0,0.45]),

xaxis2=dict(domain=[0.55,1]),

xaxis3=dict(

    domain=[0,0.45],

    anchor='y3'),

xaxis4=dict(

    domain=[0.55,1],

    anchor='y4'),

yaxis2=dict(

    

    domain=[0,0.45],

    anchor='x2'),

yaxis3=dict(domain=[0.55,1]),

yaxis4=dict(

    domain=[0.55,1],

    anchor='y4'))

fig=go.Figure(data=data,layout=layout)

iplot(fig)

    
timesData.info()

timesData.total_score.value_counts()

timesData.total_score=[float(each.replace('-','0')) for each in timesData.total_score]
timesData.country[timesData.total_score>59.0].iloc[:100]