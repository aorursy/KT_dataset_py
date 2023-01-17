# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.pandas.set_option('display.max_columns', None)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import plotly

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from matplotlib import rcParams

%matplotlib inline

from wordcloud import WordCloud
nakuri = pd.read_csv("/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")

nakuri.head()
rcParams["figure.figsize"] = 20,10

nakuri.isna().sum().plot(kind="bar")
nakuri.duplicated().sum()
job = nakuri["Job Title"].value_counts()

job_df = pd.DataFrame({"job_title":job.index,"frequency":job.values})

exper = nakuri["Job Experience Required"].value_counts()

exper_df = pd.DataFrame({"experience":exper.index,"frequency":exper.values})
fig = px.bar(data_frame=job_df[:15],x="frequency",y="job_title",color="job_title",title="Most Looking Job Title")

fig.show()
fig = make_subplots(1,2, 

                    subplot_titles = ["Most Prefered Job Tile", 

                                      "Expected Job experience"])

fig.append_trace(go.Bar(x=job_df["frequency"][:15],

                       y=job_df["job_title"][:15],

                       orientation='h',showlegend=False,

                       marker=dict(color=job_df["frequency"][:15], coloraxis="coloraxis", showscale=False)),row=1,col=1)





fig.append_trace(go.Bar(x=exper_df["frequency"][:15],

                       y=exper_df["experience"][:15],

                       orientation="h"),row=1,col=2)

fig.show()
role = nakuri['Role Category'].value_counts().nlargest(n=10)

fig = px.pie(role, 

       values = role.values, 

       names = role.index, 

       title="Top 10 Role Categories", 

       color=role.values,

       color_discrete_sequence=px.colors.qualitative.Prism)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5)

fig.update_layout(title_x=0.5)

fig.show()

location = nakuri['Location'].value_counts().nlargest(n=10)

fig = px.bar(y=location.values,

       x=location.index,

       orientation='v',

       color=location.index,

       text=location.values,

       color_discrete_sequence= px.colors.qualitative.Bold)



fig.update_traces(texttemplate='%{text:.2s}', 

                  textposition='outside', 

                  marker_line_color='rgb(8,48,107)', 

                  marker_line_width=1.5, 

                  opacity=0.7)



fig.update_layout(width=800, 

                  showlegend=False, 

                  xaxis_title="City",

                  yaxis_title="Count",

                  title="Top 10 cities by job count")

fig.show()
loca = nakuri["Location"].value_counts()

loc_df = pd.DataFrame({"location":loca.index,"frequency":loca.values})
fig = px.bar(data_frame=loc_df[:30],x="frequency",y="location",title="location which has more job openings",color="location")

fig.show()
pay_split = nakuri['Job Salary'].str[1:-1].str.split('-', expand=True)

#remove space in left and right 

pay_split[0] =  pay_split[0].str.strip()

#remove comma 

pay_split[0] = pay_split[0].str.replace(',', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

pay_split[0] = pay_split[0].str.replace(r'\D.*', '')

#display 

pay_split[0].head()
#remove space in left and right 

pay_split[1] =  pay_split[1].str.strip()

#remove comma 

pay_split[1] = pay_split[1].str.replace(',', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

pay_split[1] = pay_split[1].str.replace(r'\D.*', '')

#display 

pay_split[1].head()
pay_split[0] = pd.to_numeric(pay_split[0], errors='coerce')

pay_split[1] = pd.to_numeric(pay_split[1], errors='coerce')

pay=pd.concat([pay_split[0], pay_split[1]], axis=1, sort=False)
pay.rename(columns={0:'min_pay', 1:'max_pay'}, inplace=True)

pay.head()
nakuri=pd.concat([nakuri, pay], axis=1, sort=False)
experience_split = nakuri['Job Experience Required'].str[0:-1].str.split('-', expand=True)

experience_split.head()
#remove space in left and right 

experience_split[1] =  experience_split[1].str.strip()

#remove comma 

experience_split[1] = experience_split[1].str.replace('yr', '')

#remove all character in two condition

# 1 remove if only character

# 2 if start in number remove after all character

experience_split[1] = experience_split[1].str.replace(r'yr', '')

#display 

experience_split[1].head()
experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce')

experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce')
experience=pd.concat([experience_split[0], experience_split[1]], axis=1, sort=False)
experience.rename(columns={0:'min_experience', 1:'max_experience'}, inplace=True)

experience.head()
nakuri=pd.concat([nakuri, experience], axis=1, sort=False)

nakuri.head()
nakuri['avg_pay']=(nakuri['min_pay'].values + nakuri['max_pay'].values)/2

nakuri['avg_experience']=(nakuri['min_experience'].values + nakuri['max_experience'].values)/2
f,ax=plt.subplots(figsize=(15,5))



sns.stripplot(x='min_experience', y='min_pay', data=nakuri)

f,ax=plt.subplots(figsize=(15,5))

sns.pointplot(x='min_experience', y='min_pay', data=nakuri)
f,ax=plt.subplots(figsize=(15,5))

sns.stripplot(x='max_experience', y='max_pay', data=nakuri)
f,ax=plt.subplots(figsize=(30,10))

sns.pointplot(x='min_experience', y='min_pay', data=nakuri)
sns.pairplot(nakuri, 

             size=5, aspect=0.9, 

             x_vars=["min_experience","max_experience"],

             y_vars=["min_pay"],

             kind="reg")




sns.pairplot(nakuri, 

             size=5, aspect=0.9, 

             x_vars=["min_experience","max_experience"],

             y_vars=["max_pay"],

             kind="reg")

sns.jointplot(x='avg_experience', y='avg_pay', data=nakuri, 

              kind="kde",xlim={0,15}, ylim={0,1000000})
f,ax=plt.subplots(figsize=(30,10))

sns.stripplot(x='avg_experience', y='avg_pay', data=nakuri, jitter=True)
f,ax=plt.subplots(figsize=(30,10))

sns.pointplot(x='avg_experience', y='avg_pay', data=nakuri)
rcParams["figure.figsize"] = 15,5

nakuri[['min_pay','Industry']].groupby(["Industry"]).median().sort_values(by='min_pay',ascending=False).head(10).plot.bar(color='lightgreen')

rcParams["figure.figsize"] = 15,5

nakuri[['max_pay','Industry']].groupby(["Industry"]).median().sort_values(by='max_pay',ascending=False).head(10).plot.bar(color='lightblue')
rcParams["figure.figsize"] = 15,5

nakuri[['avg_pay','Key Skills']].groupby(["Key Skills"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='lightgreen')
rcParams["figure.figsize"] = 15,5

nakuri[['avg_pay','Job Title']].groupby(["Job Title"]).median().sort_values(by='avg_pay',ascending=False).head(10).plot.bar(color='y')
mumbai = nakuri[nakuri["Location"]=="Mumbai"]

mumbai.head()
mum_job = mumbai["Job Title"].value_counts()

mum_job_df = pd.DataFrame({

    "title":mum_job.index,

    "frequency":mum_job.values

})
fig = px.bar(data_frame=mum_job_df[:20],x="frequency",y="title",color="title",title="Job availabe in Mumabi")

fig.show()
chennai = nakuri[nakuri["Location"]=="Chennai"]

chennai.head()
chn_job = chennai["Job Title"].value_counts()

chn_job_df = pd.DataFrame({

    "title":chn_job.index,

    "frequency":chn_job.values

})
fig = px.bar(data_frame=chn_job_df[:20],x="frequency",y="title",color="title",title="Job availabe in Chennai")

fig.show()
def generate_word_cloud(text):

    wordcloud = WordCloud(

        width = 3000,

        height = 2000,

        background_color = 'black').generate(str(text))

    fig = plt.figure(

        figsize = (10,10),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()
generate_word_cloud(nakuri["Key Skills"].values[:1000])
generate_word_cloud(nakuri["Functional Area"].values[:1000])