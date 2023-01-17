import numpy as np # นำเข้า numpy โดยเรียกมันว่า np
import pandas as pd # นำเข้า pandas โดยเรียกมันว่า pd

# นำเข้า plotly 
# นำเข้า plotly โดยเรียกมันว่า py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
from plotly import tools
import plotly.graph_objs as go

# word cloud library หรือการจับกลุ่มคำ
from wordcloud import WordCloud

# นำเข้า matplotlib.pyplot โดยเรียกว่า plt
import matplotlib.pyplot as plt

# นำเข้า os
import os
print(os.listdir("../input"))


# โหลดชุดข้อมูลที่ต้องการจะใช้
timesData = pd.read_csv("../input/world-university-rankings/cwurData.csv")
timesData.head()
#คแสดงจำนวนแถวและคอลัมน์ของชุดข้อมูล
timesData.shape
# แสดงผลข้อมูลเกี่ยวกับ timedata
timesData.info()
#ตรวจสอบความสมบูรณ์ของชุดข้อมูลโดยที่ไห้ 
#Fales = ไม่มีการขาดหายของข้อมูล
#True = มีการสูญหายของชุดข้อมูล
timesData.isna().any()
#ตรวจสอบข้อมูลที่หายไปของ broad_impact
timesData['broad_impact'].isna().sum()
#บอกจำนวนข้อมูลของชุดข้อมูล
timesData.nunique()
#สร้างข้อมูลheatmap ของ timesdata
#import matplotlib.pyplot
import matplotlib.pyplot as plt
#import seabornโดยใช้คำย่อเป็นsns
import seaborn as sns
corr =timesData.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
timesData1 = pd.read_csv("../input/world-university-rankings/timesData.csv")
timesData1.head()
#Citation ของมหาวิทยาลัยแต่ละปีใน WorldRank Top 10 ตั้งแต่ปี 2011 - 2016
df2011=timesData[timesData.year==2011].iloc[:10,:] 
df2012=timesData[timesData.year==2012].iloc[:10,:]
df2013=timesData[timesData.year==2013].iloc[:10,:]
df2014=timesData[timesData.year==2014].iloc[:10,:]
df2015=timesData[timesData.year==2015].iloc[:10,:]
df2016=timesData[timesData.year==2016].iloc[:10,:]

import plotly.graph_objs as go
trace1=go.Scatter(x=df2011.world_rank, 
                 y=df2011.citations,
                 mode="markers",
                 name="2011",
                 marker=dict(color='rgba(255,0,255,0.8)'),
                 text=df2011.institution)
trace2=go.Scatter(x=df2012.world_rank,
                 y=df2012.citations,
                 mode="markers",
                 name="2012",
                 marker=dict(color='rgba(255,255,0,0.8)'),
                 text=df2012.institution)
trace3=go.Scatter(x=df2013.world_rank,
                 y=df2013.citations,
                 mode="markers",
                 name="2013",
                 marker=dict(color='rgba(165,165,0,0.8)'),
                 text=df2013.institution)
trace4=go.Scatter(x=df2014.world_rank,
                 y=df2014.citations,
                 mode="markers",
                 name="2014",
                 marker=dict(color='rgba(165,0,0,165.8)'),
                 text=df2014.institution)
trace5=go.Scatter(x=df2015.world_rank,
                 y=df2015.citations,
                 mode="markers",
                 name="2015",
                 marker=dict(color='rgba(23,45,85,0.8)'),
                 text=df2015.institution)
trace6=go.Scatter(x=df2016.world_rank,
                 y=df2016.citations,
                 mode="markers",
                 name="2016",
                 marker=dict(color='rgba(49,150,56,0.8)'),
                 text=df2016.institution)
data=[trace1,trace2,trace3,trace4,trace5,trace6]  #แสดง data ตั้งแต่ trace1 - trace6
layout=dict(title='Citation ของมหาวิทยาลัยแต่ละปีใน WorldRank Top 10', #แสดง layout โดยตั้ง title
           xaxis=dict(title='World Rank',ticklen=5,zeroline=False),#ตั้งแกนxหรือแนวนอนเป็นชื่อ World Rank
           yaxis=dict(title='Citation',ticklen=5,zeroline=False))#ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2011
df2011 = timesData1[timesData1.year == 2011].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2011.university_name, #สร้างกราฟแบบ ฺBar โดยให้แนว นอนคือ university_name
                y = df2011.citations,   #แนวตั้่งคือ citations
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2011.country)
trace2 = go.Bar(x = df2011.university_name,
                y = df2011.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2011.country)

data = [trace1,trace2]  #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2011',  #แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False), #ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False)) #ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2012
df2012 = timesData1[timesData1.year == 2012].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2012.university_name,#สร้างกราฟแบบ ฺBar โดยให้แนว นอนคือ university_name
                y = df2012.citations, #แนวตั้่งคือ citations
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2012.country)
trace2 = go.Bar(x = df2012.university_name,
                y = df2012.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2012.country)

data = [trace1,trace2] #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2012',#แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False), #ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False)) #ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2013
df2013 = timesData1[timesData1.year == 2013].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2013.university_name,
                y = df2013.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2013.country)
trace2 = go.Bar(x = df2013.university_name,
                y = df2013.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2013.country)

data = [trace1,trace2] #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2013',#แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False),  #ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False)) #ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2014
df2014 = timesData1[timesData1.year == 2014].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2014.university_name,
                y = df2014.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
trace2 = go.Bar(x = df2014.university_name,
                y = df2014.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)

data = [trace1,trace2] #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2014',#แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False),#ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False))#ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2015
df2015 = timesData1[timesData1.year == 2015].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2015.university_name,
                y = df2015.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2015.country)
trace2 = go.Bar(x = df2015.university_name,
                y = df2015.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2015.country)

data = [trace1,trace2] #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2015',#แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False),#ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False))#ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#ให้ไปเรียกข้อมูลในชื่อdf2014
df2016 = timesData1[timesData1.year == 2016].iloc[:10,:]

#เรียกใช้กราฟด้วยgo
import plotly.graph_objs as go

#สร้างtrace1-trace2
trace1 = go.Bar(x = df2016.university_name,
                y = df2016.citations,
                name = 'citations',
                marker = dict(color = 'rgba(255,174,255,0.5)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2016.country)
trace2 = go.Bar(x = df2016.university_name,
                y = df2016.total_score,
                name = 'total_score',
                marker = dict(color = 'rgba(0,255,200,0.8)', 
                              line = dict(color='rgb(0,0,0)',width=1.5)),
                text = df2016.country)

data = [trace1,trace2] #แสดง data ตั้งแต่ trace1 - trace2
layout = dict(title='2016',#แสดง layout โดยตั้ง title
           xaxis=dict(title='University',ticklen=5,zeroline=False), #ตั้งแกนxหรือแนวนอนเป็นชื่อ University
           yaxis=dict(title='Citation',ticklen=5,zeroline=False)) #ตั้งแกนyหรือแนวตั้งเป็นชื่อ Citation
fig = go.Figure (data=data, layout=layout)
iplot(fig)
plt.savefig('plotly-bar.png')
#เรียกใช้ข้อมูลจาก timeData โดยไห้แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2012 ถึงปี 2016
th_year12 = timesData.loc[(timesData['year'] == 2012) & (timesData['country']=='Thailand')]
th_year13 = timesData.loc[(timesData['year'] == 2013) & (timesData['country']=='Thailand')]
th_year14 = timesData.loc[(timesData['year'] == 2014) & (timesData['country']=='Thailand')]
th_year15 = timesData.loc[(timesData['year'] == 2015) & (timesData['country']=='Thailand')]
th_year16 = timesData.loc[(timesData['year'] == 2016) & (timesData['country']=='Thailand')]

#แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2012
th_year12
#แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2013
th_year13
#แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2014
th_year14
#แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2015
th_year15
#แสดงผลอันดับของมหาวิทยาลัยในไทยปี 2016
th_year16
#กำหนดชื่อที่เราจะเรียกใช้โดยเราจะเรียกข้อมูลจากปี 2012 โดยจะตั้งชื่อในการเรียกแตกต่างกัน
Chinaall =timesData1.loc[(timesData1['year'] == 2012) & (timesData['country']=='China')]
Japanall =timesData1.loc[(timesData1['year'] == 2012) & (timesData['country']=='Japan')]
Koreall =timesData1.loc[(timesData1['year'] == 2012) & (timesData['country']=='South Korea')]

#แสดง teaching citation international research ในปี 2012 ของประเทศ จีน
trace0 = go.Scatter(
x = Chinaall.world_rank,
y = Chinaall.teaching,
mode = "lines",
name = "teaching",
marker = dict(color = 'rgba(12, 12, 140,.4)'),
text = Chinaall.university_name
)
trace1 = go.Scatter(
x = Chinaall.world_rank,
y = Chinaall.citations,    
mode ="lines + markers",
name = "citation",
marker = dict(color = "rgba(155,98,160,.6)"),
xaxis = "x2",
yaxis = "y2",
text = Chinaall.university_name
)
trace2 = go.Scatter(
x = Chinaall.world_rank,
y= Chinaall.international,
mode = "lines",
name = "international",
marker = {"color":"rgba(36,120,153,.4)"},
xaxis = "x3",
yaxis = "y3",
text = Chinaall.university_name
)
trace3 = go.Scatter(
x = Chinaall.world_rank,
y = Chinaall.research,
mode = "lines + markers",
name = "research",
marker = {"color":"rgba(65,46,178,0.4)"},
xaxis = "x4",
yaxis = "y4",
text = Chinaall.university_name
)
data9 = [trace0,trace1,trace2,trace3]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = "Multiple Of China 2012 "
)
fig = go.Figure(data = data9,layout = layout)
iplot(fig)
#แสดง teaching citation international research ในปี 2012 ของประเทศ ญี่ปุ่น
trace0 = go.Scatter(
x = Japanall.world_rank,
y = Japanall.teaching,
mode = "lines",
name = "teaching",
marker = dict(color = 'rgba(12, 12, 140,.4)'),
text = Japanall.university_name
)
trace1 = go.Scatter(
x = Japanall.world_rank,
y = Japanall.citations,    
mode ="lines + markers",
name = "citation",
marker = dict(color = "rgba(155,98,160,.6)"),
xaxis = "x2",
yaxis = "y2",
text = Japanall.university_name
)
trace2 = go.Scatter(
x = Japanall.world_rank,
y= Japanall.international,
mode = "lines",
name = "international",
marker = {"color":"rgba(36,120,153,.4)"},
xaxis = "x3",
yaxis = "y3",
text = Japanall.university_name
)
trace3 = go.Scatter(
x = Japanall.world_rank,
y = Japanall.research,
mode = "lines + markers",
name = "research",
marker = {"color":"rgba(65,46,178,0.4)"},
xaxis = "x4",
yaxis = "y4",
text = Japanall.university_name
)
data9 = [trace0,trace1,trace2,trace3]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = "Multiple Of Japan 2012 "
)
fig = go.Figure(data = data9,layout = layout)
iplot(fig)
#แสดง teaching citation international research ในปี 2012 ของประเทศ เกาหลีใต้
trace0 = go.Scatter(
x = Koreall.world_rank,
y = Koreall.teaching,
mode = "lines",
name = "teaching",
marker = dict(color = 'rgba(12, 12, 140,.4)'),
text = Koreall.university_name
)
trace1 = go.Scatter(
x = Koreall.world_rank,
y = Koreall.citations,    
mode ="lines + markers",
name = "citation",
marker = dict(color = "rgba(155,98,160,.6)"),
xaxis = "x2",
yaxis = "y2",
text = Koreall.university_name
)
trace2 = go.Scatter(
x = Koreall.world_rank,
y= Koreall.international,
mode = "lines",
name = "international",
marker = {"color":"rgba(36,120,153,.4)"},
xaxis = "x3",
yaxis = "y3",
text = Koreall.university_name
)
trace3 = go.Scatter(
x = Koreall.world_rank,
y = Koreall.research,
mode = "lines + markers",
name = "research",
marker = {"color":"rgba(65,46,178,0.4)"},
xaxis = "x4",
yaxis = "y4",
text = Koreall.university_name
)
data9 = [trace0,trace1,trace2,trace3]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = "Multiple Of South Korea 2012 "
)
fig = go.Figure(data = data9,layout = layout)
iplot(fig)
#มหาวิทยาลัยในไทย
# สร้างฐานข้อมูลของแต่ละมหาวิทยาลัย
mu = timesData[timesData['institution'] =='Mahidol University']
snu = timesData[timesData['institution'] =='Chiang Mai University']
cu = timesData[timesData['institution'] =='Chulalongkorn University']

# สร้างตัวแปรแรกคือ มหาวิทยาลัยมหิดล
trace1 = go.Scatter(
                    x = mu.year, 
                    y = mu.score,
                    mode = "lines+markers",
                    name = "Mahidol University",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= mu.world_rank)
# สร้างตัวแปรที่สองคือ มหาวิทยาลัยเชียงใหม่
trace2 = go.Scatter(
                    x = snu.year,
                    y = snu.score,
                    mode = "lines+markers",
                    name = "Chiang Mai University",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= snu.world_rank)

# สร้างตัวแปรที่สามคือมหาวิทยาลัย จุฬาลงกรณ์
trace3 = go.Scatter(
                    x = cu.year,
                    y = cu.score,
                    mode = "lines+markers",
                    name = "Chulalongkorn University",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= cu.world_rank)

data = [trace1, trace2,trace3]
layout = dict(title = ' University Of Thailand in Worldrank.',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)

#มหาวิทยาลัยในญี่ปุ่น
# สร้างฐานข้อมูลของแต่ละมหาวิทยาลัย
kto = timesData[timesData['institution'] =='Kyoto University']
osu = timesData[timesData['institution'] =='Osaka University']
ngu = timesData[timesData['institution'] =='Nagoya University']
thu = timesData[timesData['institution'] =='Tohoku University']

# สร้างตัวแปร4ตัวแปร
trace1 = go.Scatter(
                    x = kto.year, 
                    y = kto.score,
                    mode = "lines+markers",
                    name = "Kyoto University",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= kto.world_rank)

trace2 = go.Scatter(
                    x = osu.year,
                    y = osu.score,
                    mode = "lines+markers",
                    name = "Osaka University",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= osu.world_rank)


trace3 = go.Scatter(
                    x = ngu.year,
                    y = ngu.score,
                    mode = "lines+markers",
                    name = "Nagoya University",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= ngu.world_rank)

trace4 = go.Scatter(
                    x = thu.year,
                    y = thu.score,
                    mode = "lines+markers",
                    name = "Tohoku University",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= thu.world_rank)



data = [trace1, trace2,trace3,trace4]
layout = dict(title = ' University Of Japan in Worldrank.',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)
#มหาวิทยาลัยในเกาหลี
# สร้างฐานข้อมูลของแต่ละมหาวิทยาลัย
senu = timesData[timesData['institution'] =='Seoul National University']
ysu = timesData[timesData['institution'] =='Yonsei University']
kais = timesData[timesData['institution'] =='Korea Advanced Institute of Science and Technology (KAIST)']
kru = timesData[timesData['institution'] =='Korea University']

# สร้างตัวแปร4ตัวแปร
trace1 = go.Scatter(
                    x = senu.year, 
                    y = senu.score,
                    mode = "lines+markers",
                    name = "Seoul National University",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= senu.world_rank)

trace2 = go.Scatter(
                    x = ysu.year,
                    y = ysu.score,
                    mode = "lines+markers",
                    name = "Yonsei University",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= ysu.world_rank)


trace3 = go.Scatter(
                    x = kais.year,
                    y = kais.score,
                    mode = "lines+markers",
                    name = "Korea Advanced Institute of Science and Technology (KAIST)",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= kais.world_rank)

trace4 = go.Scatter(
                    x = kru.year,
                    y = kru.score,
                    mode = "lines+markers",
                    name = "Korea University",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= kru.world_rank)



data = [trace1, trace2,trace3,trace4]
layout = dict(title = ' University Of South Korea in Worldrank.',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)

#มหาวิทยาลัยในจีน
# สร้างฐานข้อมูลของแต่ละมหาวิทยาลัย
pu = timesData[timesData['institution'] =='Peking University']
tu = timesData[timesData['institution'] =='Tsinghua University']
fu = timesData[timesData['institution'] =='Fudan University']
sjtu = timesData[timesData['institution'] =='Shanghai Jiao Tong University']

# สร้างตัวแปร4ตัวแปร
trace1 = go.Scatter(
                    x = pu.year, 
                    y = pu.score,
                    mode = "lines+markers",
                    name = "Peking University",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= pu.world_rank)

trace2 = go.Scatter(
                    x = tu.year,
                    y = tu.score,
                    mode = "lines+markers",
                    name = "Tsinghua University",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= tu.world_rank)


trace3 = go.Scatter(
                    x = fu.year,
                    y = fu.score,
                    mode = "lines+markers",
                    name = "Fudan University",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= fu.world_rank)

trace4 = go.Scatter(
                    x = sjtu.year,
                    y = sjtu.score,
                    mode = "lines+markers",
                    name = "Shanghai Jiao Tong University",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= sjtu.world_rank)



data = [trace1, trace2,trace3,trace4]
layout = dict(title = ' University Of China in Worldrank.',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)

# สร้างฐานข้อมูลของแต่ละมหาวิทยาลัย
senu = timesData[timesData['institution'] =='Seoul National University']
ysu = timesData[timesData['institution'] =='Yonsei University']
kais = timesData[timesData['institution'] =='Korea Advanced Institute of Science and Technology (KAIST)']
kru = timesData[timesData['institution'] =='Korea University']
kto = timesData[timesData['institution'] =='Kyoto University']
osu = timesData[timesData['institution'] =='Osaka University']
ngu = timesData[timesData['institution'] =='Nagoya University']
thu = timesData[timesData['institution'] =='Tohoku University']
mu = timesData[timesData['institution'] =='Mahidol University']
snu = timesData[timesData['institution'] =='Chiang Mai University']
cu = timesData[timesData['institution'] =='Chulalongkorn University']
pu = timesData[timesData['institution'] =='Peking University']
tu = timesData[timesData['institution'] =='Tsinghua University']
fu = timesData[timesData['institution'] =='Fudan University']
sjtu = timesData[timesData['institution'] =='Shanghai Jiao Tong University']

# สร้างตัวแปร15ตัวแปร
trace1 = go.Scatter(
                    x = senu.year,  
                    y = senu.score,
                    mode = "lines+markers",
                    name = "Seoul National University (KOR)",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= senu.world_rank)

trace2 = go.Scatter(
                    x = ysu.year,
                    y = ysu.score,
                    mode = "lines+markers",
                    name = "Yonsei University (KOR)",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= ysu.world_rank)


trace3 = go.Scatter(
                    x = kais.year,
                    y = kais.score,
                    mode = "lines+markers",
                    name = "Korea Advanced Institute of Science and Technology (KOR)",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= kais.world_rank)

trace4 = go.Scatter(
                    x = kru.year,
                    y = kru.score,
                    mode = "lines+markers",
                    name = "'Korea University (KOR)",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= kru.world_rank)

trace5 = go.Scatter(
                    x = kto.year, 
                    y = kto.score,
                    mode = "lines+markers",
                    name = "Kyoto University (JP)",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= kto.world_rank)

trace6 = go.Scatter(
                    x = osu.year,
                    y = osu.score,
                    mode = "lines+markers",
                    name = "Osaka University (JP)",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= osu.world_rank)

trace7 = go.Scatter(
                    x = ngu.year,
                    y = ngu.score,
                    mode = "lines+markers",
                    name = "Nagoya University (JP)",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= ngu.world_rank)

trace8 = go.Scatter(
                    x = thu.year,
                    y = thu.score,
                    mode = "lines+markers",
                    name = "Tohoku University (JP)",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= thu.world_rank)

trace9 = go.Scatter(
                    x = mu.year, 
                    y = mu.score,
                    mode = "lines+markers",
                    name = "Mahidol University (TH)",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= mu.world_rank)

trace10 = go.Scatter(
                    x = snu.year,
                    y = snu.score,
                    mode = "lines+markers",
                    name = "Chiang Mai University (TH)",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= snu.world_rank)


trace11 = go.Scatter(
                    x = cu.year,
                    y = cu.score,
                    mode = "lines+markers",
                    name = "Chulalongkorn University (TH)",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= cu.world_rank)

trace12 = go.Scatter(
                    x = pu.year, 
                    y = pu.score,
                    mode = "lines+markers",
                    name = "Peking University (CN)",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= pu.world_rank)

trace13 = go.Scatter(
                    x = tu.year,
                    y = tu.score,
                    mode = "lines+markers",
                    name = "Tsinghua University (CN)",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= tu.world_rank)

trace14 = go.Scatter(
                    x = fu.year,
                    y = fu.score,
                    mode = "lines+markers",
                    name = "Fudan University (CN)",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= fu.world_rank)

trace15 = go.Scatter(
                    x = sjtu.year,
                    y = sjtu.score,
                    mode = "lines+markers",
                    name = "Shanghai Jiao Tong University (CN)",
                    marker = dict(color = 'rgb(107, 50, 41)'),
                    text= sjtu.world_rank)

data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12,
       trace13,trace14,trace15]
layout = dict(title = ' SouthKorea , Japan , Thailand , China University in Worldrank.',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)

