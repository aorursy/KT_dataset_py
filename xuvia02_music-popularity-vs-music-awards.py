import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool

from bokeh.io import show, output_notebook

from scipy.ndimage import gaussian_gradient_magnitude

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns

from matplotlib_venn import venn2

import scipy as sp

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import random
# import datasets

spotify = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')

grammy = pd.read_csv('/kaggle/input/data-on-songs-from-billboard-19992019/BillboardFromLast20/grammySongs_1999-2019.csv')

billboard = pd.read_csv('/kaggle/input/data-on-songs-from-billboard-19992019/BillboardFromLast20/billboardHot100_1999-2019.csv')
len(spotify)
len(grammy)
len(billboard)
spotify.head()
grammy.head()
billboard.head()
# put spotify into the same format

spotify['artists'] = spotify['artists'].str.strip("[]")

spotify['artists'] = spotify['artists'].str.replace("'", "").str.replace(" &", ",")

spotify.rename(columns = {'name':'Name', 'artists':'Artist'}, inplace = True)

spotify['Name'] = spotify['Name'].str.lower()

spotify['Artist'] = spotify['Artist'].str.lower()



# put grammy into the same format

grammy['Artist'] = grammy['Artist'].str.replace(" &", ",")

grammy['Name'] = grammy['Name'].str.lower()

grammy['Artist'] = grammy['Artist'].str.lower()



# put billboard into the same format

billboard.rename(columns = {'Artists':'Artist'}, inplace = True)

billboard['Artist'] = billboard['Artist'].str.replace(" &", ",")

billboard['Name'] = billboard['Name'].str.lower()

billboard['Artist'] = billboard['Artist'].str.lower()
# clean spotify dataset (in case it contains the same songs by the same artists)

songs = spotify.groupby(['Name', 'Artist'], as_index = False).agg({'acousticness' : 'mean', 'acousticness' : 'mean', 'danceability' : 'mean', 'duration_ms' : 'mean', 'energy' : 'mean', 'explicit' : 'max', 'instrumentalness' : 'mean', 'key' : 'median',  'liveness' : 'mean',  'loudness' : 'mean',  'mode' : 'max', 'popularity' : 'sum', 'speechiness' : 'mean', 'tempo' : 'mean', 'valence' : 'mean', 'year' : 'min'})



# clean grammy songs, merge it with songs(spotify) to get more info, drop useless columns

gr = grammy.merge(songs, on = ['Name', 'Artist'])

gr = gr.drop(columns = 'Unnamed: 0').drop(columns = 'X')



# clean billboard

bb1 = billboard.groupby(['Name', 'Artist', 'Week', 'Weekly.rank'], as_index = False).agg({'Weeks.on.chart' : 'max', 'Peak.position' : 'min', 'Genre' : 'first', 'Date':'first'})

bb1 = bb1.merge(songs, on = ['Name', 'Artist'])

bb2 = bb1.groupby(['Name','Artist'], as_index = False).agg({'Weeks.on.chart' : 'max', 'Peak.position' : 'min'})

bb2 = bb2.dropna(subset = ['Peak.position', 'Weeks.on.chart'])

bb3 = bb1.groupby(['Name','Artist'], as_index = False).agg({'acousticness' : 'mean', 'danceability' : 'mean', 'duration_ms' : 'mean', 'energy' : 'mean', 'explicit' : 'max', 'instrumentalness' : 'mean', 'key' : 'median',  'liveness' : 'mean',  'loudness' : 'mean',  'mode' : 'max', 'speechiness' : 'mean', 'tempo' : 'mean', 'valence' : 'mean', 'year' : 'min'})



# capitalize each word (reformatting)

songs['Name'] = songs['Name'].str.title()

songs['Artist'] = songs['Artist'].str.title()

gr['Name'] = gr['Name'].str.title()

gr['Artist'] = gr['Artist'].str.title()

bb1['Name'] = bb1['Name'].str.title()

bb1['Artist'] = bb1['Artist'].str.title()

bb2['Name'] = bb2['Name'].str.title()

bb2['Artist'] = bb2['Artist'].str.title()

bb3['Name'] = bb3['Name'].str.title()

bb3['Artist'] = bb3['Artist'].str.title()

bb3['loudness'] = bb3['loudness']/60 + 1

songs['loudness'] = songs['loudness']/60 + 1

gr['loudness'] = gr['loudness']/60 + 1
songs.head()
gr.head()
bb1.head()
bb2.head()
bb3.head()
fig = go.Figure()

fig.add_trace(go.Box(y=songs['acousticness'], name = 'acousticness - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['acousticness'], name = 'acousticness - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['acousticness'], name = 'acousticness - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['danceability'], name = 'danceability - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['danceability'], name = 'danceability - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['danceability'], name = 'danceability - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['energy'], name = 'energy - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['energy'], name = 'energy - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['energy'], name = 'energy - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['instrumentalness'], name = 'instrumentalness - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['instrumentalness'], name = 'instrumentalness - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['instrumentalness'], name = 'instrumentalness - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['liveness'], name = 'liveness - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['liveness'], name = 'liveness - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['liveness'], name = 'liveness - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['loudness'], name = 'loudness - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['loudness'], name = 'loudness - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['loudness'], name = 'loudness - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=songs['valence'], name = 'valence - all', hovertext= songs['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=gr['valence'], name = 'valence - grammy', hovertext= gr['Name'],

    hoverinfo="y+text"))

fig.add_trace(go.Box(y=bb3['valence'], name = 'valence - popular', hovertext= bb3['Name'],

    hoverinfo="y+text"))



fig.update_layout(

    title='Audio Profile Comparison between popular songs, award-winning songs, and all songs',

    yaxis=dict(

        zerolinecolor='rgb(0, 0, 0)',

        zerolinewidth=2,

    ),

    paper_bgcolor='rgb(250, 250, 250)',

    plot_bgcolor='rgb(230, 230, 240)',

    showlegend=False, 

)

fig.show()
f = plt.figure(figsize=(10, 10))

corr = spotify.corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,  cmap = sns.diverging_palette(220, 10, n=100),  square=True)

_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
STOPWORDS.add("featuring")

STOPWORDS.add("songwriter")

STOPWORDS.add("nan")

STOPWORDS.add("artist")

STOPWORDS.add("the")

stopwords = set(STOPWORDS)
def green_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(140, 25%%, %d%%)" % random.randint(1, 60)

_words = '' 

  

# iterate through the csv file 

for val in gr.Artist: 

      

    # typecaste each val to string 

    val = str(val)

  

    # split the value 

    tokens = val.split()

      

    _words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 800, height = 800,

                background_color ='white',

                stopwords = stopwords,

                min_font_size = 10,random_state=1).generate(_words)



# plot the WordCloud image

plt.figure(figsize = (8, 8))

plt.imshow(wordcloud.recolor(color_func=green_color_func, random_state=3),

           interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad = 0)

plt.show()
def blue_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(200, 250%%, %d%%)" % random.randint(1, 60)

_words = ''  

  

# iterate through the csv file 

for val in billboard.Artist: 

      

    # typecaste each val to string 

    val = str(val).title()

  

    # split the value 

    tokens = val.split()

      

    _words += " ".join(tokens)+" "

  

wordcloud = WordCloud(width = 800, height = 800,

                background_color ='white',

                stopwords = stopwords,

                min_font_size = 10,random_state=1).generate(_words)



# plot the WordCloud image

plt.figure(figsize = (8, 8))

plt.imshow(wordcloud.recolor(color_func=blue_color_func, random_state=3),

           interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad = 0)

plt.show()
def purple_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(267, 100%%, %d%%)" % random.randint(1, 60)

d = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()

_words_2 = '' 



  

# iterate through the csv file 

for val in songs.Artist: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split()

      

    _words_2 += " ".join(tokens)+" "

    

music_color = np.array(Image.open(os.path.join(d, "../input/images/cd1.png"))) # a cd image found online



stopwords = set(STOPWORDS) 

STOPWORDS.add("featuring")

STOPWORDS.add("songwriter")

STOPWORDS.add("nan")

STOPWORDS.add("artist")



wc = WordCloud(width = 400, height = 400,background_color="white", max_words=2000, mask=music_color,

               stopwords=stopwords, max_font_size=40, random_state=42)



wc.generate(_words_2)



# create coloring from image

image_colors = ImageColorGenerator(music_color)

# show

plt.figure(figsize = (15, 15))

plt.imshow(wc.recolor(color_func=purple_color_func), interpolation="bilinear")

plt.axis("off")

plt.figure()

plt.tight_layout(pad = 0)

plt.show()
coeffs = np.polyfit(bb2["Peak.position"], bb2["Weeks.on.chart"], 1)

plt.scatter(bb2["Peak.position"], bb2["Weeks.on.chart"])

plt.xlabel("Chart Position")

plt.ylabel("Weeks on Chart")

plt.plot(bb2["Peak.position"], coeffs[0] * bb2["Peak.position"] + coeffs[1], color = "black")
bb2_mean = bb2.groupby('Peak.position',as_index = False).agg({'Weeks.on.chart' : 'mean'})

plt.scatter(bb2_mean["Peak.position"], bb2_mean["Weeks.on.chart"])

plt.xlabel("Peak Position")

plt.ylabel("Mean weeks on chart of songs with the same peak position")

plt.plot(bb2_mean["Peak.position"], coeffs[0] * bb2_mean["Peak.position"] + coeffs[1], color = "black")
output_notebook()
data2 = ColumnDataSource(bb2)
TOOLTIPS = [("(Name, Artist, Peak, Weeks)", "(@Name, @Artist, @{Peak.position}, @{Weeks.on.chart})")]



p = figure(title = 'Weeks On Chart vs. Peak Position', plot_width=500, plot_height=400, tooltips = TOOLTIPS)



# add a circle renderer with a size, color, and alpha

# plt.scatter(pz['contributions'], pz['GPA'])

p.circle("Peak.position", "Weeks.on.chart", size = 3, color="blue", source = data2)

p.xaxis.axis_label = 'Weekly Rank'

p.yaxis.axis_label = 'Weeks On Chart'

# show the results

show(p)
rad = bb1[(bb1['Name'] == 'Radioactive') & (bb1['Artist'] == 'Imagine Dragons')]

costar = bb1[(bb1['Name'] == 'Counting Stars') & (bb1['Artist'] == 'Onerepublic')]

roll = bb1[(bb1['Name'] == 'Rolling In The Deep') & (bb1['Artist'] == 'Adele')]

imy = bb1[(bb1['Name'] == 'I\'M Yours') & (bb1['Artist'] == 'Jason Mraz')]

soy = bb1[(bb1['Name'] == 'Shape Of You') & (bb1['Artist'] == 'Ed Sheeran')]



ynt = bb1[(bb1['Name'] == 'You Need To Calm Down') & (bb1['Artist'] == 'Taylor Swift')]

ibe = bb1[(bb1['Name'] == 'I Believe') & (bb1['Artist'] == 'Fantasia')]

badg = bb1[(bb1['Name'] == 'Bad Guy') & (bb1['Artist'] == 'Billie Eilish')]

ks = bb1[(bb1['Name'] == 'Killshot') & (bb1['Artist'] == 'Eminem')]

atm = bb1[(bb1['Name'] == 'Atm') & (bb1['Artist'] == 'J. Cole')]
fig = go.Figure()

fig.add_trace(go.Scatter(x = rad['Weeks.on.chart'], y = rad['Weekly.rank'], mode = 'lines', name = 'Radioactive - Imagine Dragons'))

fig.add_trace(go.Scatter(x = costar['Weeks.on.chart'], y = costar['Weekly.rank'], mode = 'lines', name = 'Counting Stars - OneRepublic'))

fig.add_trace(go.Scatter(x = roll['Weeks.on.chart'], y = roll['Weekly.rank'], mode = 'lines', name = 'Rolling In The Deep - Adele'))

fig.add_trace(go.Scatter(x = imy['Weeks.on.chart'], y = imy['Weekly.rank'], mode = 'lines', name = 'I\'m Yours - Jason Mraz'))

fig.add_trace(go.Scatter(x = soy['Weeks.on.chart'], y = soy['Weekly.rank'], mode = 'lines', name = 'Shape Of You - Ed Sheeran'))



fig.add_trace(go.Scatter(x = ynt['Weeks.on.chart'], y = ynt['Weekly.rank'], mode = 'lines', name = 'You Need To Calm Down - Taylor Swift'))

fig.add_trace(go.Scatter(x = ibe['Weeks.on.chart'], y = ibe['Weekly.rank'], mode = 'lines', name = 'I Believe - Fantasia'))

fig.add_trace(go.Scatter(x = badg['Weeks.on.chart'], y = badg['Weekly.rank'], mode = 'lines', name = 'Bad Guy - Billie Eilish'))

fig.add_trace(go.Scatter(x = ks['Weeks.on.chart'], y = ks['Weekly.rank'], mode = 'lines', name = 'Killshot - Eminem'))

fig.add_trace(go.Scatter(x = atm['Weeks.on.chart'], y = atm['Weekly.rank'], mode = 'lines', name = 'Atm - J. Cole'))



fig.update_layout(

    title='Weekly Rank vs. Weeks on Billboard',

    xaxis_title="Weeks On Chart",

    yaxis_title="Weekly Rank",

    legend_title="Song - Artist",

    paper_bgcolor='rgb(250, 250, 250)',

    plot_bgcolor='rgb(230, 230, 240)'

)

fig['layout']['yaxis']['autorange'] = "reversed"

fig.show()
gr[gr['Name'] == 'Rolling In The Deep']
gr[gr['Name'] == 'Shape Of You']
gr[gr['Name'] == 'Radioactive']
plt.figure(figsize=(6,6))

v = venn2([set(gr['Name']), set(bb1['Name'])], 

          set_labels = ('Songs got Grammy', 'Songs on Billboard'), 

          set_colors=('darkblue', 'lightblue'), 

          )

v.get_label_by_id('A').set_size(20)

v.get_label_by_id('A').set_color('darkblue')

v.get_label_by_id('B').set_size(15)

v.get_label_by_id('A').set_color('darkblue')
grammy_for_pie = gr.groupby("Genre", as_index = False).agg({"Name": "count"})
my_colors = ['palegreen', 'paleturquoise', 'lightpink', 'lightsteelblue', 

           'khaki', 'tomato', 'aqua', 'lightseagreen', 'lightsalmon', 'lightskyblue',"plum"]
plt.figure(figsize = (10,15))

_ = plt.pie(grammy_for_pie["Name"],labels= grammy_for_pie["Genre"],autopct="%1.2f%%",colors= my_colors)