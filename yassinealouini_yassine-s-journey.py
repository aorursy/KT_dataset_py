import os

import string

import datetime

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



print(os.listdir("../input"))
### Enter the user name here ###

user_name = "yassinealouini"
users_df = pd.read_csv("../input/meta-kaggle/Users.csv")

user_df = users_df[users_df["UserName"]==user_name]

user_id = user_df["Id"].values[0]

user_display = user_df["DisplayName"].values[0]

print("The user id for the given user name is : ",user_id)

print("The display name for the given user name is : ",user_display)
team_members_df = pd.read_csv("../input/meta-kaggle/TeamMemberships.csv")

team_df = pd.read_csv("../input/meta-kaggle/Teams.csv")

comp_df = pd.read_csv("../input/meta-kaggle/Competitions.csv")



temp_df = team_members_df[team_members_df["UserId"]==user_id]

temp_df = pd.merge(temp_df, team_df, left_on="TeamId", right_on="Id", how="left")

temp_df = pd.merge(temp_df, comp_df, left_on="CompetitionId", right_on="Id", how="left")

temp_df["DeadlineDate"] = pd.to_datetime(temp_df["DeadlineDate"], format="%m/%d/%Y %H:%M:%S")

temp_df["DeadlineYear"] = temp_df["DeadlineDate"].dt.year

temp_df["DeadlineDate"] = temp_df["DeadlineDate"].apply(lambda x: datetime.date(x.year,x.month,1))



temp_df = temp_df[~np.isnan(temp_df["PrivateLeaderboardRank"])]

temp_df.head()
def scatter_plot(cnt_srs, color):

    trace = go.Scatter(

        x=cnt_srs.index[::-1],

        y=cnt_srs.values[::-1],

        showlegend=False,

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_df = temp_df.groupby('DeadlineYear')['PrivateLeaderboardRank'].agg(["size", "mean", "min"])

cnt_srs = cnt_df["size"]

cnt_srs = cnt_srs.sort_index()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="green",

    ),

)



layout = go.Layout(

    title='Count of competitions over years',

    font=dict(size=14),

    width=800,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="TargetCount")





### Mean Private Rank ###

cnt_srs = cnt_df["mean"]

cnt_srs = cnt_srs.sort_index()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="blue",

    ),

)



layout = go.Layout(

    title='Mean Rank over years',

    font=dict(size=14),

    width=800,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="TargetCount")







### Best rank each year ###

cnt_srs = cnt_df["min"]

cnt_srs = cnt_srs.sort_index()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color="red",

    ),

)



layout = go.Layout(

    title='Best Rank in each year',

    font=dict(size=14),

    width=800,

    height=500,

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="TargetCount")
#temp_df["Medal"].value_counts()



cnt_df = temp_df.pivot_table(index="DeadlineYear", columns="Medal", values="PrivateLeaderboardRank", aggfunc="count")

cnt_df = cnt_df.fillna(0)



def get_bar_chart(cnt_srs, name, color):

    trace = go.Bar(

        x=cnt_srs.index,

        y=cnt_srs.values,

        name=name,

        marker=dict(

            color=color,

        ),

    )

    return trace



medal_map = {1.:"Gold", 2.:"Silver", 3.:"Bronze"}

color_map = {1.:"gold", 2.:"silver", 3.:"darkorange"}

traces = []

for col in np.array(cnt_df.columns)[::-1]:

    cnt_srs = cnt_df[col]

    traces.append(get_bar_chart(cnt_srs, medal_map[col], color_map[col]))



layout = go.Layout(

    title='Competition Medals in each year',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')

cnt_srs = temp_df["HostSegmentTitle"].value_counts()



labels = (np.array(cnt_srs.index))

sizes = (np.array((cnt_srs / cnt_srs.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Competition Type Distribution',

    font=dict(size=14),

    width=600,

    height=600,

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="usertype")
comp_tags_df = pd.read_csv("../input/meta-kaggle/CompetitionTags.csv")

tags_df = pd.read_csv("../input/meta-kaggle/Tags.csv")

cnt_df = pd.merge(temp_df[["CompetitionId"]], comp_tags_df, on="CompetitionId", how="inner")

cnt_df = pd.merge(cnt_df, tags_df, left_on="TagId", right_on="Id", how="inner")

cnt_df["Name"].value_counts()



def bar_chart(cnt_srs, color):

    trace = go.Bar(

        x=cnt_srs.index,

        y=cnt_srs.values,

        showlegend=False,

        marker=dict(

            color=color,

        ),

    )

    return trace



cnt_srs = cnt_df["Name"].value_counts().head(10)

traces = [bar_chart(cnt_srs, "orange")]

layout = go.Layout(

    title='Data type of competitions',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')

cnt_srs = temp_df["TeamName"].value_counts().head(5)

traces = [bar_chart(cnt_srs, "blue")]

layout = go.Layout(

    title='Favorite Team Name',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
temp_df = team_members_df[team_members_df["UserId"]==user_id]

temp_df = pd.merge(team_members_df, temp_df, on="TeamId", how="inner", suffixes=('', '_y'))

temp_df = temp_df[temp_df["UserId"]!=user_id]

temp_df = pd.merge(temp_df, users_df, left_on="UserId", right_on="Id", how="left")



cnt_srs = temp_df["DisplayName"].value_counts().head(7)

traces = [bar_chart(cnt_srs, "green")]

layout = go.Layout(

    title='Number of competitions with favorite team members ',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
kernels_df = pd.read_csv("../input/meta-kaggle/Kernels.csv")

temp_df = kernels_df[kernels_df["AuthorUserId"]==user_id]

temp_df["MadePublicDate"] = pd.to_datetime(temp_df["MadePublicDate"], format="%m/%d/%Y")

temp_df["MadePublicYear"] = temp_df["MadePublicDate"].dt.year

temp_df.head()
# Number of kernels

cnt_srs = temp_df["MadePublicYear"].value_counts()

traces = [bar_chart(cnt_srs, "blue")]

layout = go.Layout(

    title='Number of kernels in each year',

    font=dict(size=14),

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')





# Number of views

cnt_srs = temp_df.groupby("MadePublicYear")["TotalViews"].mean()

traces = [bar_chart(cnt_srs, "green")]

layout = go.Layout(

    title='Mean number of views per kernel',

    font=dict(size=14),

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')



# Number of votes

cnt_srs = temp_df.groupby("MadePublicYear")["TotalVotes"].mean()

traces = [bar_chart(cnt_srs, "red")]

layout = go.Layout(

    title='Mean number of votes per kernel',

    font=dict(size=14),

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
cnt_df = temp_df.pivot_table(index="MadePublicYear", columns="Medal", values="AuthorUserId", aggfunc="count")

cnt_df = cnt_df.fillna(0)



def get_bar_chart(cnt_srs, name, color):

    trace = go.Bar(

        x=cnt_srs.index,

        y=cnt_srs.values,

        name=name,

        marker=dict(

            color=color,

        ),

    )

    return trace



medal_map = {1.:"Gold", 2.:"Silver", 3.:"Bronze"}

color_map = {1.:"gold", 2.:"silver", 3.:"darkorange"}

traces = []

for col in np.array(cnt_df.columns)[::-1]:

    cnt_srs = cnt_df[col]

    traces.append(get_bar_chart(cnt_srs, medal_map[col], color_map[col]))



layout = go.Layout(

    title='Kernel Medals in each year',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
temp_df = kernels_df[kernels_df["AuthorUserId"]==user_id]

kernel_tags_df = pd.read_csv("../input/meta-kaggle/KernelTags.csv")

temp_df = pd.merge(temp_df, kernel_tags_df, left_on="Id", right_on="KernelId", how="inner")

temp_df = pd.merge(temp_df, tags_df, left_on="TagId", right_on="Id", how="inner")



cnt_srs = temp_df["Name"].value_counts().head(10)

traces = [bar_chart(cnt_srs, "orange")]

layout = go.Layout(

    title='Tag count of the kernels',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
#temp_df = kernels_df[kernels_df["AuthorUserId"]==user_id]

forum_message_df = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")

temp_df = forum_message_df[forum_message_df["PostUserId"]==user_id]

temp_df["PostDate"] = pd.to_datetime(temp_df["PostDate"], format="%m/%d/%Y %H:%M:%S")

temp_df["PostYear"] = temp_df["PostDate"].dt.year

temp_df.head()
# Number of kernels

cnt_srs = temp_df["PostYear"].value_counts()

traces = [bar_chart(cnt_srs, "blue")]

layout = go.Layout(

    title='Number of forum posts in each year',

    font=dict(size=14),

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
cnt_df = temp_df.pivot_table(index="PostYear", columns="Medal", values="PostUserId", aggfunc="count")

cnt_df = cnt_df.fillna(0)



def get_bar_chart(cnt_srs, name, color):

    trace = go.Bar(

        x=cnt_srs.index,

        y=cnt_srs.values,

        name=name,

        marker=dict(

            color=color,

        ),

    )

    return trace



medal_map = {1.:"Gold", 2.:"Silver", 3.:"Bronze"}

color_map = {1.:"gold", 2.:"silver", 3.:"darkorange"}

traces = []

for col in np.array(cnt_df.columns)[::-1]:

    cnt_srs = cnt_df[col]

    traces.append(get_bar_chart(cnt_srs, medal_map[col], color_map[col]))



layout = go.Layout(

    title='Discussion Medals in each year',

    font=dict(size=14),

    barmode='stack',

    width=800,

    height=500,

)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig, filename='stacked-bar')
import re

def clean_string(txt):

    txt = str(txt)

    txt = re.sub("<.*?>", "", txt)

    txt = re.sub(' +', ' ', txt)

    return txt



temp_df["Message"] = temp_df["Message"].apply(lambda x: clean_string(x))

#temp_df.head()



from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(temp_df["Message"], title="Word Cloud of the common words in forum messages")