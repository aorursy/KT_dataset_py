%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
import os
print(os.listdir("../input"))
from wordcloud import WordCloud, STOPWORDS
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

don = pd.read_csv("../input/Donors.csv", low_memory=False)
teach_not = don["Donor Is Teacher"].value_counts()
labels = ["not a teacher", " a teacher"]
data = [teach_not["No"], teach_not["Yes"]]
iplot([go.Pie(labels=labels, values=data, hoverinfo="label+percent", textfont=dict(size=20))])
cities = don["Donor City"].value_counts()
perc = cities.sum() / 100 
percent_each = []
for i in cities.index:
    part = cities[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)
fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="darkred")
plt.fill_between(list(range(len(cont))), cont, color="salmon")
plt.xlabel("cities sorted in descending order by donors", fontsize=15)
plt.ylabel("donor %", fontsize=15)
plt.title("city cumulative load into donor quantity", fontsize=20)
plt.show()
data = [go.Bar(y=cities.iloc[:20], 
              x=cities.index[:20],
             marker=dict(color="salmon"))]

layout = go.Layout(title="Donors per city: top 20", titlefont=dict(size=25), font=dict(size=15),
                  xaxis=dict(title="city", titlefont=dict(size=20)),
                  yaxis=dict(title="count", titlefont=dict(size=20)))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
states = don["Donor State"].value_counts()
# lets show which cities explain most donorship
perc = states.sum() / 100 
percent_each_state = []
for i in states.index:
    part = states[i] / perc
    percent_each_state.append((i, part))

cont_state = []
load  = 0 
for i in percent_each_state:
    load += i[1]
    cont_state.append(load)
# cumulative load of states into the donorship
fig_cum = plt.figure(figsize=(10,8))
plt.plot(cont_state, lw=2, color="darkgreen")
plt.fill_between(list(range(len(cont_state))), cont_state, color="lightgreen")

plt.xlabel("states sorted in descending order by donors", fontsize=15)
plt.ylabel("donor %", fontsize=15)
plt.title("states cumulative load into donor quantity", fontsize=20)
plt.show()
data = [go.Bar(y=states.iloc[:10], 
              x=states.index[:10],
             marker=dict(color="lightgreen"))]

layout = go.Layout(title="Donors per state: top 10", titlefont=dict(size=25), font=dict(size=20),
                  xaxis=dict(title="donors", titlefont=dict(size=20)),
                  yaxis=dict(title="count", titlefont=dict(size=20)))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
sch = pd.read_csv("../input/Schools.csv")
sch_type = sch["School Metro Type"].value_counts()
sch_type
labels = sch_type.index
data = [go.Pie(labels=labels, values=sch_type, hoverinfo="labal+percent", textfont=dict(size=20))]
layout = go.Layout(title="School type", font=dict(size=20))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
state_sch = sch["School State"].value_counts()
perc = state_sch.sum() / 100 
percent_each = []
for i in state_sch.index:
    part = state_sch[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)
fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="blue")
plt.fill_between(list(range(len(cont))), cont, color="lightblue")
plt.xlabel("states sorted in descending order by number of donated schools", fontsize=15)
plt.ylabel("%", fontsize=15)
plt.title("load of states into the donated schools", fontsize=20)
plt.show()
data = [go.Bar(y=state_sch.iloc[:10], 
              x=state_sch.index[:10],
             marker=dict(color="lightblue"))]

layout = go.Layout(title="Donated schools per state: top 10", titlefont=dict(size=25), font=dict(size=20),
                  xaxis=dict(title="states", titlefont=dict(size=20)),
                  yaxis=dict(title="count", titlefont=dict(size=20)) )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
city_sch = sch["School City"].value_counts()
perc = city_sch.sum() / 100 
percent_each = []
for i in city_sch.index:
    part = city_sch[i] / perc
    percent_each.append((i, part))
    
cont = []
load  = 0 
for i in percent_each:
    load += i[1]
    cont.append(load)
fig_cum = plt.figure(figsize=(10,6))
plt.plot(cont, lw=5, color="darkred")
plt.fill_between(list(range(len(cont))), cont, color="salmon")
plt.xlabel("cities sorted in descending order by number of schools", fontsize=15)
plt.ylabel("%", fontsize=15)
plt.title("load of cities into the donated schools", fontsize=20)
plt.show()
data = [go.Bar(y=city_sch.iloc[:20], 
              x=city_sch.index[:20],
             marker=dict(color="salmon"))]

layout = go.Layout(title="Donated schools per city: top 20", titlefont=dict(size=25), font=dict(size=20),
                  xaxis=dict(title="cities", titlefont=dict(size=20)),
                  yaxis=dict(title="count", titlefont=dict(size=20)) )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
donors_states = pd.DataFrame({"donors" : states, "schools" : state_sch})
donors_states.shape
donors_states.dropna(inplace=True)
donors_states.shape
data = [go.Scatter(x=donors_states["donors"], y=donors_states["schools"], mode="markers")]

layout = go.Layout(title="donors vs states", titlefont=dict(size=25), font=dict(size=12),
                  xaxis=dict(title="donors in the state", titlefont=dict(size=20)),
                  yaxis=dict(title="donated schools in the state", titlefont=dict(size=20)))
fig = go.Figure(data=data, layout=layout)

iplot(fig)
donors_states.corr()
donors_cities = pd.DataFrame({"donors" : cities, "schools" : city_sch})
donors_cities.shape
donors_cities.dropna(inplace=True)
donors_cities.shape    
data = [go.Scatter(x=donors_cities["donors"], y=donors_cities["schools"], mode="markers")]

layout = go.Layout(title="donors vs cities", titlefont=dict(size=25), font=dict(size=12),
                  xaxis=dict(title="donors in the city", titlefont=dict(size=20)),
                  yaxis=dict(title="donated schools in the city", titlefont=dict(size=20)))
fig = go.Figure(data=data, layout=layout)

iplot(fig)
donors_cities.corr()
donation = pd.read_csv("../input/Donations.csv")
donation.head()
donors_freq = donation["Donor ID"].value_counts()
fig_money = plt.figure(figsize=(10,8))
sns.distplot(donation["Donation Amount"], hist=False)
plt.title("distribution of donation amount ($)", fontsize=20)
plt.show()
print("min donation: ", donation["Donation Amount"].min())
print("max donation : ", donation["Donation Amount"].max())
print("mean donation: ", donation["Donation Amount"].mean())
print("median donation: ", donation["Donation Amount"].median())
print("mode donation: ", donation["Donation Amount"].mode())
fig_money = plt.figure(figsize=(10,8))
sns.distplot(donors_freq, hist=False)
plt.xlabel("counts of money giving")
plt.title("distribution of donation frequency", fontsize=20)
plt.show()
data = [go.Bar(y=donors_freq.iloc[:50], 
              x=donors_freq.index[:50],
             marker=dict(color="salmon"))]

layout = go.Layout(title="Top 50 most generous donors", titlefont=dict(size=25), font=dict(size=12),
                  xaxis=dict(title="donors", titlefont=dict(size=20)),
                  yaxis=dict(title="donation count", titlefont=dict(size=20)) )

fig = go.Figure(data=data, layout=layout)
iplot(fig)
prj = pd.read_csv("../input/Projects.csv")
prj.head(10)
prj_res = prj["Project Resource Category"].value_counts()
data = [go.Bar(y=prj_res.index, 
              x=prj_res,
             marker=dict(color="grey"),
              orientation="h")]

layout = go.Layout(title="Project Resources Needed", titlefont=dict(size=25), font=dict(size=12),
                  xaxis=dict(title="donors", titlefont=dict(size=20)))

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig_res = plt.figure(figsize=(10, 8))
plt.barh(range(1, len(prj_res) + 1), prj_res, color="salmon")
plt.yticks(list(range(1, len(prj_res) + 1)), prj_res.index, fontsize=15)
plt.title("Project Resources Needed", fontsize=25)
plt.xlabel("counts", fontsize=15)
plt.show()
prj["Project Cost"].describe()
print("min cost: ", prj["Project Cost"].min())
print("max cost : ", prj["Project Cost"].max())
print("mean cost: ", prj["Project Cost"].mean())
print("median cost: ", prj["Project Cost"].median())
print("mode cost: ", prj["Project Cost"].mode())
fig_prj_cost = plt.figure(figsize=(10, 8))
sns.distplot(prj["Project Cost"], hist=False)
plt.title("Project cost distribution", fontsize=20)
plt.show()
prj_type = prj["Project Type"].value_counts()
prj_type
fig_pie_prj_type = plt.figure(figsize=(8,8))
labels = ["Teacher-Led", "Professional Development", "Student-Led"]


plt.pie(prj_type, labels=labels,  shadow=True)
plt.title("Project Type", fontsize=20)
plt.axis("equal")
my_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
prj_status = prj["Project Current Status"].value_counts()

prj_status
fig_pie_teach = plt.figure(figsize=(8,8))
labels = ["Fully Funded", "Expired", "Live"]


plt.pie(prj_status, labels=labels,  shadow=True)
plt.title("Project Status", fontsize=20)
plt.axis("equal")
my_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
text = ""
for el in prj["Project Title"]:
    try:
        text += el + " "
    except Exception as e:
        print(e, el)
    
print("length of the resulting text is ", len(text), " chars")
print(text[:100])
word_cloud = WordCloud(width=480, height=480, margin=0).generate(text)
fig_word = plt.figure(figsize=(10, 10))
plt.imshow(word_cloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("word cloud based on project title", fontsize=20)
plt.show()

text_need = ""
for el in prj["Project Need Statement"]:
    try:
        text_need += el + " "
    except Exception as e:
        print(e, el)
len(text_need)
word_cloud_need = WordCloud(width=480, height=480, margin=0).generate(text_need)
fig_word_need = plt.figure(figsize=(10, 10))
plt.imshow(word_cloud_need, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("word cloud based on project need statement", fontsize=20)
plt.show()
stop = STOPWORDS
stop.update(["need", "help", "student", "Student", "students", "Students"])
word_cloud_need1 = WordCloud(width=480, height=480, margin=0, stopwords=stop).generate(text_need)
fig_word_need1 = plt.figure(figsize=(10, 10))
plt.imshow(word_cloud_need1, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("word cloud based on project need statement: 'help', 'need', and 'student' excluded", fontsize=20)
plt.show()
prj_cat = prj["Project Subject Category Tree"].value_counts()
prj_cat.shape
fig_cat = plt.figure(figsize=(10, 8))
plt.barh(range(1, len(prj_cat[:10]) + 1), prj_cat[:10], color="olive", alpha=0.7)
plt.yticks(list(range(1, len(prj_cat[:11]))), prj_cat.index[:11], fontsize=15)
plt.title("top 10 project subject categories", fontsize=20)
plt.xlabel("counts", fontsize=15)
plt.show()
text_category = ""
for el in prj["Project Subject Category Tree"]:
    try:
        text_category += el + " "
    except Exception as e:
        print(e, el)
word_cloud_cat = WordCloud(width=480, height=480, margin=0).generate(text_category)
fig_word = plt.figure(figsize=(10, 10))
plt.imshow(word_cloud_cat, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("word cloud based on project subject category tree", fontsize=20)
plt.show()
prj_cat_sub = prj["Project Subject Subcategory Tree"].value_counts()
prj_cat_sub[:10]
fig_cat = plt.figure(figsize=(10, 8))
plt.barh(range(1, len(prj_cat_sub[:15]) + 1), prj_cat_sub[:15], color="darkgreen", alpha=0.7)
plt.yticks(list(range(1, len(prj_cat_sub[:16]))), prj_cat_sub.index[:16], fontsize=15)
plt.title("Top 15 project subject subcategories", fontsize=20)
plt.xlabel("counts", fontsize=15)
plt.show()
text_subcat = ""
for el in prj["Project Subject Subcategory Tree"]:
    try:
        text_subcat += el + " "
    except Exception as e:
        print(e, el)
word_cloud_subcat = WordCloud(width=480, height=480, margin=0).generate(text_subcat)
fig_word = plt.figure(figsize=(10, 10))
plt.imshow(word_cloud_subcat, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title("word cloud based on project subject subcategory tree", fontsize=20)
plt.show()
