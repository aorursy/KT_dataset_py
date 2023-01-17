







# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


Data=pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")

Data.info()

Data.head(10)
June=Data[:2].copy()

sum_of_June_for_open=sum([June.Open])

sum_of_June_for_High=sum([June.High])

sum_of_June_for_Low=sum([June.Low])

sum_of_June_for_Close=sum([June.Close])

sum_of_June_for_Volume=sum([June.Volume])



July=Data[2:23].copy()

sum_of_July_for_open=sum([July.Open])

sum_of_July_for_High=sum([July.High])

sum_of_July_for_Low=sum([July.Low])

sum_of_July_for_Close=sum([July.Close])

sum_of_July_for_Volume=sum([July.Volume])





August=Data[24:45].copy()

sum_of_August_for_open=sum([August.Open])

sum_of_August_for_High=sum([August.High])

sum_of_August_for_Low=sum([August.Low])

sum_of_August_for_Close=sum([August.Close])

sum_of_August_for_Volume=sum([August.Volume])





September=Data[46:66].copy()

sum_of_September_for_open=sum([September.Open])

sum_of_September_for_High=sum([September.High])

sum_of_September_for_Low=sum([September.Low])

sum_of_September_for_Close=sum([September.Close])

sum_of_September_for_Volume=sum([September.Volume])



October = Data[66:87].copy()

sum_of_October_for_open=sum([October.Open])

sum_of_October_for_High=sum([October.High])

sum_of_October_for_Low=sum([October.Low])

sum_of_October_for_Close=sum([October.Close])

sum_of_October_for_Volume=sum([October.Volume])



November = Data[87:100].copy()

sum_of_November_for_open=sum([November.Open])

sum_of_November_for_High=sum([November.High])

sum_of_November_for_Low=sum([November.Low])

sum_of_November_for_Close=sum([November.Close])

sum_of_November_for_Volume=sum([November.Volume])







    

    

print(sum_of_July_for_open)

   


JUNE_OPEN=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_open,

    mode="lines",

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=June["Date"]

    

)



JUNE_HİGH=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_High,

    mode="lines",

    name="HIGH PRİCE ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=June["Date"]

    

)



JUNE_LOW=go.Scatter(



    x=June.Date,

    y=sum_of_June_for_Low,

    mode='lines',

    name='LOW PRICE' ,

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=June["Date"]





)



JUNE_CLOSE=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Close,

    mode='lines',

    name='CLOSE PRICE ',

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=June["Date"]





)

JUNE_VOLUME=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Volume,

    mode='lines',

    name='VOLUME PRICE ',

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=June["Date"]





)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



JULY_OPEN=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=July["Date"]

)





JULY_HIGH=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=July["Date"]

)



JULY_LOW=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=July["Date"]

)



JULY_CLOSE=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=July["Date"]

    )

JULY_VOLUME=go.Scatter(

    x=July.Date,

    y=sum_of_July_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=July["Date"]





)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





AUGUST_OPEN=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=August["Date"]

)



AUGUST_HIGH=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=August["Date"]

)



AUGUST_LOW=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=August["Date"]

)



AUGUST_CLOSE=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=August["Date"]

)

AUGUST_VOLUME = go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=August["Date"]

)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------



SEPTEMBER_OPEN=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=September["Date"]

)

SEPTEMBER_HIGH=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_LOW=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=September["Date"]

)



SEPTEMBER_CLOSE=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_VOLUME=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=September["Date"]

)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





OCTOBER_OPEN=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



OCTOBER_HIGH=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_High,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)

OCTOBER_LOW=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Low,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)



OCTOBER_CLOSE=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Close,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



OCTOBER_VOLUME=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Volume,

    mode="lines",

    name=" ",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------





NOVEMBER_OPEN=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_open,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_HIGH=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_High,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_LOW=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Low,

    mode="lines",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)

NOVEMBER_CLOSE=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Close,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_VOLUME=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Volume,

    mode="lines",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)





#---------------------------------------------------------------------------------------------------------------------------------------------------------------------









data=[JUNE_OPEN ,JUNE_LOW , JUNE_HİGH ,JUNE_CLOSE,

      JULY_OPEN ,JULY_HIGH,JULY_LOW,JULY_CLOSE,

      AUGUST_OPEN,AUGUST_HIGH,AUGUST_CLOSE,

      SEPTEMBER_OPEN,SEPTEMBER_HIGH,SEPTEMBER_LOW,

      OCTOBER_OPEN,OCTOBER_HIGH,OCTOBER_LOW,OCTOBER_CLOSE,

      NOVEMBER_OPEN,NOVEMBER_HIGH,NOVEMBER_LOW,NOVEMBER_CLOSE]



layout=dict(title =" REWİEV ON PRİCE FOR MOUNTH FROM TESLA  ",

           xaxis=dict(title="date" , ticklen=5  , zeroline=False)

            

           )

fig=dict(data=data, layout=layout)



iplot(fig)
JUNE_OPEN=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_open,

    mode="markers",

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=June["Date"]

    

)



JUNE_HİGH=go.Scatter(

    x=June.Date,

    y=sum_of_June_for_High,

    mode="markers",

    name="HIGH PRİCE ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=June["Date"]

    

)



JUNE_LOW=go.Scatter(



    x=June.Date,

    y=sum_of_June_for_Low,

    mode='markers',

    name='LOW PRICE' ,

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=June["Date"]





)



JUNE_CLOSE=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Close,

    mode='markers',

    name='CLOSE PRICE ',

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=June["Date"]





)

JUNE_VOLUME=go.Scatter(

    

    x=June.Date,

    y=sum_of_June_for_Volume,

    mode='markers',

    name='VOLUME PRICE ',

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=June["Date"]





)



#-------------------------------------------------------------------------------------------------------------------------------------------------

JULY_OPEN=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=July["Date"]

)





JULY_HIGH=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_High,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=July["Date"]

)



JULY_LOW=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Low,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=July["Date"]

)



JULY_CLOSE=go.Scatter(



    x=July.Date,

    y=sum_of_July_for_Close,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=July["Date"]

    )

JULY_VOLUME=go.Scatter(

    x=July.Date,

    y=sum_of_July_for_Volume,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=July["Date"]





)





#-------------------------------------------------------------------------------------------------------------------------------------------------

AUGUST_OPEN=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=August["Date"]

)



AUGUST_HIGH=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_High,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=August["Date"]

)



AUGUST_LOW=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Low,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=August["Date"]

)



AUGUST_CLOSE=go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Close,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=August["Date"]

)

AUGUST_VOLUME = go.Scatter(



    x=August.Date,

    y=sum_of_August_for_Volume,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=August["Date"]

)







#-------------------------------------------------------------------------------------------------------------------------------------------------

SEPTEMBER_OPEN=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=September["Date"]

)

SEPTEMBER_HIGH=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_High,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_LOW=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Low,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=September["Date"]

)



SEPTEMBER_CLOSE=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Close,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=September["Date"]

)



SEPTEMBER_VOLUME=go.Scatter(



    x=September.Date,

    y=sum_of_September_for_Volume,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=September["Date"]

)

#-------------------------------------------------------------------------------------------------------------------------------------------------

OCTOBER_OPEN=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



OCTOBER_HIGH=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_High,

    mode="markers",

    name=" ",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)

OCTOBER_LOW=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Low,

    mode="markers",

    name=" ",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)



OCTOBER_CLOSE=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Close,

    mode="markers",

    name=" ",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



OCTOBER_VOLUME=go.Scatter(



    x=October.Date,

    y=sum_of_October_for_Volume,

    mode="markers",

    name=" ",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)







#-------------------------------------------------------------------------------------------------------------------------------------------------



NOVEMBER_OPEN=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_open,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_HIGH=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_High,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_LOW=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Low,

    mode="markers",

    name="",

    marker=dict(color="rgba(0,0,255,0.8)"),

    text=October["Date"]

)

NOVEMBER_CLOSE=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Close,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,255,0,0.8)"),

    text=October["Date"]

)



NOVEMBER_VOLUME=go.Scatter(



    x=November.Date,

    y=sum_of_November_for_Volume,

    mode="markers",

    name="",

    marker=dict(color="rgba(255,0,255,0.8)"),

    text=October["Date"]

)





#-------------------------------------------------------------------------------------------------------------------------------------------------

data=[JUNE_OPEN,JUNE_HİGH,JUNE_LOW,JUNE_CLOSE,

     JULY_OPEN,JULY_HIGH,JULY_LOW,JULY_CLOSE,

      AUGUST_OPEN,AUGUST_HIGH,AUGUST_LOW,AUGUST_CLOSE,

      SEPTEMBER_OPEN,SEPTEMBER_HIGH,SEPTEMBER_LOW,SEPTEMBER_CLOSE,

      OCTOBER_OPEN ,OCTOBER_HIGH , OCTOBER_LOW , OCTOBER_CLOSE,

      NOVEMBER_OPEN , NOVEMBER_HIGH  , NOVEMBER_LOW , NOVEMBER_CLOSE

     ]

layout=dict(title="PRICE 'S SPLITTED UP FOR MOUNTH SUCH AS BELOW (OPENING,HIGH , LOW ,CLOSE),",

           xaxis=dict(title="MONTHS",ticklen=5 ,zeroline=False),

            yaxis=dict(title="AMOUNT OF PRICE(K)" , ticklen=5 , zeroline=False)

           )

fig=dict(data=data ,  layout=layout)

iplot(fig)
JUNE_OPEN=go.Bar(

    x=June.Date,

    y=sum_of_June_for_open,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,0,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)



JUNE_HİGH=go.Bar(

    x=June.Date,

    y=sum_of_June_for_High,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)

JUNE_LOW=go.Bar(

    x=June.Date,

    y=sum_of_June_for_Low,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)



JUNE_CLOSE=go.Bar(

    x=June.Date,

    y=sum_of_June_for_Close,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)

JUNE_VOLUME=go.Bar(

    x=June.Date,

    y=sum_of_June_for_Volume,

    name="OPENING PRİCE ",

    marker=dict(color="rgba(255,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)),

    text=June["Date"]

    

)

#----------------------------------------------------------------------------------------------------------------------------

JULY_OPEN=go.Bar(



    x=July.Date,

    y=sum_of_July_for_open,

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=July["Date"]

)





JULY_HIGH=go.Bar(



    x=July.Date,

    y=sum_of_July_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=July["Date"]

)



JULY_LOW=go.Bar(



    x=July.Date,

    y=sum_of_July_for_Low,

    name="",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=July["Date"]

)



JULY_CLOSE=go.Bar(



    x=July.Date,

    y=sum_of_July_for_Close,

    name="",

    marker=dict(color="rgba(255,255,0,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=July["Date"]

    )

JULY_VOLUME=go.Bar(

    x=July.Date,

    y=sum_of_July_for_Volume,

    name="",

    marker=dict(color="rgba(255,0,255,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=July["Date"]





)



#----------------------------------------------------------------------------------------------------------------------------

AUGUST_OPEN=go.Bar(



    x=August.Date,

    y=sum_of_August_for_open,

    name="",

    marker=dict(color="rgba(255,0,0,0.8)",

    line=dict(color='rgb(0,0,0)',width=3)),

    text=August["Date"]

)





AUGUST_HIGH=go.Bar(



    x=August.Date,

    y=sum_of_August_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=August["Date"]

)



AUGUST_LOW=go.Bar(



    x=August.Date,

    y=sum_of_August_for_Low,

    name="",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=August["Date"]

)



AUGUST_CLOSE=go.Bar(



    x=August.Date,

    y=sum_of_August_for_Close,

    name="",

    marker=dict(color="rgba(255,255,0,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=August["Date"]

    )

AUGUST_VOLUME=go.Bar(

    x=August.Date,

    y=sum_of_August_for_Volume,

    name="",

    marker=dict(color="rgba(255,0,255,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=August["Date"]





)



#----------------------------------------------------------------------------------------------------------------------------



SEPTEMBER_OPEN=go.Bar(



    x=September.Date,

    y=sum_of_September_for_open,

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=September["Date"]

)





SEPTEMBER_HIGH=go.Bar(



    x=September.Date,

    y=sum_of_September_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=September["Date"]

)



SEPTEMBER_LOW=go.Bar(



    x=September.Date,

    y=sum_of_September_for_Low,

    name="",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=September["Date"]

)



SEPTEMBER_CLOSE=go.Bar(



    x=September.Date,

    y=sum_of_September_for_Close,

    name="",

    marker=dict(color="rgba(255,255,0,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=September["Date"]

    )

SEPTEMBER_VOLUME=go.Bar(

    x=September.Date,

    y=sum_of_September_for_Volume,

    name="",

    marker=dict(color="rgba(255,0,255,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=September["Date"]





)





#----------------------------------------------------------------------------------------------------------------------------



OCTOBER_OPEN=go.Bar(



    x=October.Date,

    y=sum_of_October_for_open,

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=October["Date"]

)





OCTOBER_HIGH=go.Bar(



    x=October.Date,

    y=sum_of_October_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=October["Date"]

)



OCTOBER_LOW=go.Bar(



    x=October.Date,

    y=sum_of_October_for_Low,

    name="",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=October["Date"]

)



OCTOBER_CLOSE=go.Bar(



    x=October.Date,

    y=sum_of_October_for_Close,

    name="",

    marker=dict(color="rgba(255,255,0,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=October["Date"]

    )

OCTOBER_VOLUME=go.Bar(

    x=October.Date,

    y=sum_of_October_for_Volume,

    name="",

    marker=dict(color="rgba(255,0,255,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=October["Date"]





)





#----------------------------------------------------------------------------------------------------------------------------

NOVEMBER_OPEN=go.Bar(



    x=November.Date,

    y=sum_of_November_for_open,

    name="",

    marker=dict(color="rgba(255,0,0,0.8)"),

    text=November["Date"]

)





NOVEMBER_HIGH=go.Bar(



    x=November.Date,

    y=sum_of_November_for_High,

    name="",

    marker=dict(color="rgba(0,255,0,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=November["Date"]

)



NOVEMBER_LOW=go.Bar(



    x=November.Date,

    y=sum_of_November_for_Low,

    name="",

    marker=dict(color="rgba(0,0,255,0.8)",

                line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=November["Date"]

)



NOVEMBER_CLOSE=go.Bar(



    x=November.Date,

    y=sum_of_November_for_Close,

    name="",

    marker=dict(color="rgba(255,255,0,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=November["Date"]

    )

NOVEMBER_VOLUME=go.Bar(

    x=November.Date,

    y=sum_of_November_for_Volume,

    name="",

    marker=dict(color="rgba(255,0,255,0.8)",

               line=dict(color='rgb(0,0,0)',width=3)

               ),

    text=November["Date"]





)



#----------------------------------------------------------------------------------------------------------------------------



data = [JUNE_OPEN , JUNE_HİGH,JUNE_LOW , JUNE_CLOSE ,

       JULY_OPEN , JULY_HIGH , JULY_LOW , JULY_CLOSE,

       AUGUST_OPEN , AUGUST_HIGH ,AUGUST_LOW , AUGUST_CLOSE,

       SEPTEMBER_OPEN , SEPTEMBER_HIGH ,SEPTEMBER_LOW , SEPTEMBER_CLOSE,

       OCTOBER_OPEN , OCTOBER_HIGH ,OCTOBER_LOW , OCTOBER_CLOSE,

       NOVEMBER_OPEN ,NOVEMBER_HIGH ,NOVEMBER_LOW ,NOVEMBER_CLOSE,

        

        

        

       

       ]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)



import pandas as pd

TSLA = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")