# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

#import plotly.plotly as py



#from plotly.offline import init_notebook_mode iplot

#init_notebook_mode(connected=True)

from wordcloud import WordCloud



#plotly

#import plotly.graph_objs as go

#from plotly.offline import init_notebook_mode, plot

#init_notebook_mode(connected=True)

# import figure factory

#import plotly.figure_factory as ff



from collections import Counter



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#needed for plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import plotly as py



import plotly.graph_objs as go



init_notebook_mode(connected=True)


data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

data.head()
data.columns
data.describe()

data.info()#there are nan values in publisher and year
#publisher vs jp_sales    

data[["Publisher","JP_Sales"]].groupby(["Publisher"], as_index=False).mean().sort_values(by="JP_Sales",ascending=False)
data[["Publisher","Global_Sales","JP_Sales"]].groupby(["Publisher"], as_index=False).mean().sort_values(by=["Global_Sales","JP_Sales"],ascending=False)
#comparing sale-publisher

data[["Publisher","EU_Sales","JP_Sales"]].groupby(["Publisher"], as_index=False).mean().sort_values(by=["EU_Sales","JP_Sales"],ascending=False)

#platform vs sales    

data[["Platform","Global_Sales"]].groupby(["Platform"], as_index=False).mean().sort_values(by="Global_Sales",ascending=False)  
data.isnull().sum()

#to show missing values on graph

import missingno as msno

msno.matrix(data)

plt.xlabel("Variables")

plt.ylabel("Missing Values")

plt.figure(figsize=(15,9))

plt.show()

# missingno bar plot

msno.bar(data)

plt.ylabel("non-missing values")

plt.xlabel("Variables")

plt.show()
#Let's fill the nan years

#according to graph, main sales are applyied between 1995-2017

#according to graph, detected values between 1995-2017 gives us average year-frequency 

average_year = (data["Year"]>1995.0)

average_year2 = data["Year"]<2017.0

data[average_year & average_year2].Year.mean() #average 2007

plt.figure(figsize=(15,10))

sns.countplot(data.Year)

plt.xlabel("Years"),plt.ylabel("Frequency(Quantity of Each Year)"),plt.title("Year-Frequency")

plt.xticks(rotation=90)

plt.grid()



#year-firm

#grafige gore ortlama yil 2006-2007 gibi bir sey

#yukarida ortalamaya gore 2007 degeri baz alinabilir.

data["Genre"].unique()

plt.figure(figsize=(15,10))

sns.boxplot(x="Genre",y="Year",data=data)

plt.show()
data[data["Year"].isnull()]
#done

data["Year"] = data["Year"].fillna(2017.0)

data[data["Year"].isnull()]
#to fill publishwer

data["Publisher"].unique()

#when looked at the graph,frequency of 2004 and 2007 can be realized

#let's focus on 2004-2007

 

data[data["Publisher"].isnull()]
var1 = data["Publisher"].value_counts().head(10)

var2 = data["Platform"].value_counts().head(10)
#Let's make frequency graph for publisher and platform

plt.figure(figsize=(15,10))

sns.barplot(x=var1,y=var1.index)

plt.title("Publisher-Frequency")

plt.xticks()

plt.grid()



plt.figure(figsize=(15,10))

sns.barplot(x=var2,y=var2.index)

plt.title("Publisher-Frequency")

plt.xlabel("Frequency")

plt.grid()
#after filling publisher

data["Publisher"] = data["Publisher"].fillna("Nintendo")

data[data["Publisher"].isnull()]

#done

#year---count#

#firstly year must be converted to int

data.Year = data.Year.astype(int)

plt.figure(figsize=(15,10))

sns.countplot(data.Year)

plt.title("Year-Frequency")

plt.xticks(rotation=90)

plt.show()
#genre---count#

plt.figure(figsize=(15,10))

sns.countplot(data.Genre)

plt.xticks(rotation=90)

plt.title("Genre-Frequency")

plt.show()
data.columns

corr_list = ["Year","NA_Sales","EU_Sales",

             "JP_Sales","Other_Sales","Global_Sales"]

sns.heatmap(data[corr_list].corr() , annot=True , fmt=".2f")

plt.show()
NA_Sales = data.iloc[:,6]

EU_Sales = data.iloc[:,7]

JP_Sales = data.iloc[:,8]

Other_Sales = data.iloc[:,9]

Global_Sales = data.iloc[:,10]

data_plot = data.drop(["Rank","Name","Year","Genre","Publisher"],axis=1)

# donut plot

feature_names = "NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"

feature_size = [len(NA_Sales),len(EU_Sales),len(JP_Sales),len(Other_Sales),len(Global_Sales)]

# create a circle for the center of plot

circle = plt.Circle((0,0),0.2,color = "white")

plt.pie(feature_size, labels = feature_names, colors = ["red","green","blue","cyan","yellow"] )

p = plt.gcf()

p.gca().add_artist(circle)

plt.title("Number of Each Features")

plt.show()

#top 20 publishers making discount (according to number of discount)

var3 = data["Publisher"].value_counts().head(20)
publisher_list = [i for i in var3.index]

na_sales = []

eu_sales = []

jp_sales = []

other_sales = []

global_sales = []



for i in publisher_list:

    x = data[data["Publisher"]==i] #x is dataframe

    na_sales.append(sum(x.NA_Sales)/len(x))

    eu_sales.append(sum(x.EU_Sales)/len(x))

    jp_sales.append(sum(x.JP_Sales)/len(x))

    other_sales.append(sum(x.Other_Sales)/len(x))

    global_sales.append(sum(x.Global_Sales)/len(x))



# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=na_sales,y=publisher_list,color='green',alpha = 0.8,label='Nort America' )

sns.barplot(x=eu_sales,y=publisher_list,color='red',alpha = 0.7,label='Europa')

sns.barplot(x=jp_sales,y=publisher_list,color='cyan',alpha = 0.9,label='Japonia')

sns.barplot(x=other_sales,y=publisher_list,color='yellow',alpha = 0.6,label='Other')

sns.barplot(x=global_sales,y=publisher_list,color='orange',alpha = 0.6,label='Global')





ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Regions', ylabel='Publisher'

       ,title = "Avarage of Discount's  According to Regions ")

plt.show()
#Platforms making discount (with number of discount)

var4 = data["Platform"].value_counts()

platform_list = [i for i in var4.index]



na_sales = []

eu_sales = []

jp_sales = []

other_sales = []

global_sales = []



for i in platform_list:

    x = data[data["Platform"]==i]

    

    na_sales.append(sum(x.NA_Sales)/len(x))

    eu_sales.append(sum(x.EU_Sales)/len(x))

    jp_sales.append(sum(x.JP_Sales)/len(x))

    other_sales.append(sum(x.JP_Sales)/len(x))

    global_sales.append(sum(x.Global_Sales)/len(x))



na_sales



f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=na_sales,y=platform_list,color="green",alpha=0.9,label="Nort America")

sns.barplot(x=eu_sales,y=platform_list,color="red",alpha=0.7,label="Europa")

sns.barplot(x=jp_sales,y=platform_list,color="cyan",alpha=0.9,label="Japonia")

sns.barplot(x=other_sales,y=platform_list,color="yellow",alpha=0.6,label="Other")

sns.barplot(x=global_sales,y=platform_list,color="orange",alpha=0.6,label="Global")



ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Regions', ylabel='Platforms'

       ,title = "Avarage of Discounts According to Regions ")

plt.show()
#top 30 video game making global discount (according to number of discount)

var5 = data["Name"].value_counts().head(30)
discount = []

the_games_list = [i for i in var5.index]



for i in the_games_list:

    x = data[data["Name"]==i]

    discount.append(sum(x.Global_Sales)/len(x))

    

#sorting to be shown more tidy in graph

data2 = pd.DataFrame({"Name":the_games_list,"Global_Sales":discount})

new_index = (data2["Global_Sales"].sort_values(ascending=True)).index.values

sorted_data = data2.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['Name'], y=sorted_data['Global_Sales'])

plt.xticks(rotation=90)

plt.xlabel('Video Games')

plt.ylabel('Avarage Global Sales')

plt.title('The Games with Global Sales')

plt.grid()
g = sns.factorplot(x="Platform", y="Global_Sales", data = data, kind = "bar", size = 10)

g.set_ylabels("Global Sales Probability")

plt.xticks(rotation=90)

plt.show()
sales = []

years = []

JP_sales = []

NA_sales = []

for i in publisher_list:

    x = data[data["Publisher"]==i]

    sales.append(sum(x.Global_Sales))

    JP_sales.append(sum(x.JP_Sales))

    NA_sales.append(sum(x.NA_Sales))

    

    



plt.figure(figsize=(9,15))

sns.barplot(x=publisher_list,y=sales)

plt.xticks(rotation=90)

plt.xlabel("Publisher")

plt.ylabel("Sum of Global Sales")

plt.title("Sum of Global Sales with Publisher")

plt.grid()
plt.figure(figsize=(9,15))

sns.barplot(x=publisher_list,y=JP_sales)

plt.xticks(rotation=90)

plt.xlabel("Publisher")

plt.ylabel("Sum of Japonia Sales")

plt.title("Sum of Japonia Sales with Publisher")

plt.grid()

plt.show()
plt.figure(figsize=(9,15))

sns.barplot(x=publisher_list,y=NA_sales)

plt.xticks(rotation=90)

plt.xlabel("Publisher")

plt.ylabel("Sum of North America Sales")

plt.title("Sum of North America Sales with Publisher")

plt.grid()

plt.show()
#pip install plotly==3.10.0
# prepare data frame

df = data.iloc[:100,:]

# import graph objects as "go"

#2017 must be evalueated because mots game selling is in 2007

df2007 = data[data.Year==2007].iloc[:10,:]



import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.Rank,

                    y = df.Global_Sales,

                    mode = "lines+markers",

                    name = "Global Sales",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.Name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.Rank,

                    y = df.NA_Sales,

                    mode = "lines+markers",

                    name = "Nort America Sales",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.Name)





trace3 = go.Scatter(

                    x = df.Rank,

                    y = df.EU_Sales,

                    mode = "lines+markers",

                    name = "Europe Sales",

                    marker = dict(color = 'pink'),

                    text= df.Name)



trace4 = go.Scatter(

                    x = df.Rank,

                    y = df.JP_Sales,

                    mode = "lines+markers",

                    name = "Japonia Sales",

                    marker = dict(color = 'orange'),

                    text= df.Name)



trace5 = go.Scatter(

                    x = df.Rank,

                    y = df.Other_Sales,

                    mode = "lines+markers",

                    name = "Other Sales",

                    marker = dict(color ='LightSkyBlue'),

                    text= df.Name)







dataa = [trace1, trace2,trace3,trace4,trace5]

layout = dict(title = 'Type of Sale vs World Rank of Top 100 Games',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = dataa, layout = layout)

iplot(fig)
# create trace1 

trace1 = go.Bar(

                x = df2007.Publisher,

                y = df2007.EU_Sales,

                name = "Europa Sales",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2007.Platform)

# create trace2 

trace2 = go.Bar(

                x = df2007.Publisher,

                y = df2007.NA_Sales,

                name = "North America Sales",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2007.Platform)

                

trace3 = go.Bar(

                x = df2007.Publisher,

                y = df2007.JP_Sales,

                name = "Japonia Sales",

                marker = dict(color = 'LightSkyBlue',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2007.Platform)

                

                

data = [trace1, trace2,trace3]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
