import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
from plotly import tools

py.init_notebook_mode(connected=True)
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
# read hero info and hero power files 
hero_info = pd.read_csv("../input/heroes_information.csv")
hero_pow = pd.read_csv("../input/super_hero_powers.csv")
# show hero_info file
hero_info.head()
# show hero_pow file 
hero_pow.head()
# show missing values in Publisher, Skin color, and Weight 
print("Number of missing values in 'Publisher': ", hero_info['Publisher'].isnull().sum())
print("Number of missing values in 'Weight': ", hero_info['Weight'].isnull().sum())
# show list of columns in hero_info
hero_info.info()

# replace all cells that contain '-' and NaN with 'unknown' in Publisher attribute 
hero_info.replace(to_replace = '-', value = 'Unknown', inplace = True)
hero_info.replace(to_replace = 'NaN', value = 'Unknown', inplace = True)

# drop 'Unnamed: 0' column
new_heroinfo = hero_info.drop(['Unnamed: 0'], axis=1)
# since some of the weights are empty we will replace it with 'NaN' 
hero_info[hero_info['Weight'].isnull()]

# since some of the weights and Heights are negative, replace it with 'NaN' 
hero_info.replace(-99.0, np.nan, inplace = True)

hero_info.info()
# gets all values of Height and Weight from hero_info
ht_wt = hero_info[['Height','Weight']]
# imputing missing heights and weights with median 
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

X = imputer.fit_transform(ht_wt)

hero_ht_wt = pd.DataFrame(X, columns = ht_wt.columns)
heroes_wo_ht_wt = hero_info.drop(['Height', 'Weight'],axis = 1)
hero_info = pd.concat([heroes_wo_ht_wt, hero_ht_wt], axis=1)
hero_info.head()
# create pie chart that specifies how many male, female, and unknown superheroes there are
gender_pie = hero_info['Gender'].value_counts()
genders = list(gender_pie.index)
amt_genders = list((gender_pie/gender_pie.sum())*100)
trace = go.Pie(labels = genders, values = amt_genders)
layout = go.Layout(
            title='Male vs. Female',
            height = 700, 
            width = 700
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Hero Genders')

villian_gender_series = hero_info['Gender'].loc[hero_info['Alignment']=='bad'].value_counts()
villian_genders = list(villian_gender_series.index)
villian_distribution = list((villian_gender_series/villian_gender_series.sum())*100)
villian_gender_series = hero_info['Gender'].loc[hero_info['Alignment']=='bad'].value_counts()
villian_genders = list(villian_gender_series.index)
villian_distribution = list((villian_gender_series/villian_gender_series.sum())*100)

trace = go.Pie(labels = villian_genders, values = villian_distribution)
layout = go.Layout(
            title='Evil Males vs. Evil Females',
            height = 700, 
            width = 700, 
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Gender Alignments')
alignment_series = hero_info['Alignment'].loc[hero_info['Gender']=='Female'].value_counts()
alignment_fm = list(alignment_series.index)
alignment_distribution = list((alignment_series/alignment_series.sum())*100)

trace = go.Pie(labels = alignment_fm, values = alignment_distribution)
layout = go.Layout(
            title='Female Superhero Distribution',
            height = 700, 
            width = 700, 
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Female Alignments')
publisher_pie = hero_info['Publisher'].value_counts()
publishers = list(publisher_pie.index)
publications = list((publisher_pie/publisher_pie.sum())*100)
trace = go.Pie(labels = publishers, values = publications)
layout = go.Layout(
            title='Heroes by Publisher',
            height = 700, 
            width = 700
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Publisher Heroes')



layout = go.Layout(
            title='Evil Males vs. Evil Females',
            height = 700, 
            width = 700
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Hero Genders')


# collecting all male and female values from 'Gender'
male = hero_info.loc[hero_info['Gender']=='Male']
female = hero_info.loc[hero_info['Gender']=='Female']

# creating bar graph for male alignments
trace_m = go.Bar( 
    x = male['Alignment'].value_counts().index,
    y = male['Alignment'].value_counts().values,
    name='male'
)

# creating bar graph for female alignments 
trace_f = go.Bar( 
    x = female['Alignment'].value_counts().index,
    y = female['Alignment'].value_counts().values,
    name='female'
)

# gathering data from trace_m and trace_f to transfer to bar graph 
data = [trace_m, trace_f]

# creating bar graph 
layout = go.Layout(
    title = 'Alignment of Superheroes by Gender',
    barmode = 'group',
    height = 500,
    width = 800
)

# plotting bar graph 
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'Alignment by Gender')


trace_race = go.Bar( 
    x = hero_info['Race'].value_counts().index, 
    y = hero_info['Race'].value_counts().values, 
    name = 'Races'
)

layout = go.Layout(
    title = 'Superheroes by Race', 
    barmode = 'bar'
)

fig = go.Figure(data = [trace_race], layout = layout)
py.iplot(fig, filename = 'Race of Superhero')