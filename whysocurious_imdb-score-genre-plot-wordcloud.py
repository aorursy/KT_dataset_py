# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_movie = pd.read_csv("../input/movie_metadata.csv")

df_movie.columns
df_movie.genres.head()
df_plt_key = df_movie[['movie_title','duration','genres','imdb_score','title_year','language','country','content_rating','plot_keywords']]
df_plt_key = df_plt_key[~df_plt_key.plot_keywords.isnull()]

df_plt_key.head(6)
df_plt_key.shape
genres_list = []

for genre in df_plt_key.genres.unique():

    genres_list = genres_list + genre.split("|")

genres_list = list(set(genres_list))
print (genres_list)
pltkey_list = []

for key in df_plt_key.plot_keywords.unique():

    pltkey_list = pltkey_list + key.split("|")

pltkey_list = list(set(pltkey_list))
len(pltkey_list)
genre_plot = {}

for k,v in df_plt_key[df_plt_key.imdb_score > 7.5].iterrows():

    for genre in v['genres'].split("|"):

        

        if genre not in genre_plot.keys():

            genre_plot[genre] = []

            genre_plot[genre] = list(genre_plot[genre] + v['plot_keywords'].split("|"))

        else:

            genre_plot[genre] = list(genre_plot[genre] + v['plot_keywords'].split("|"))

        
for keys in genre_plot.keys():

    print (keys, ": ", len(genre_plot[keys]))
genre_plot['Action']
import matplotlib.pyplot as plt

from wordcloud import WordCloud



#text = ''

#for ind, val in enumerate(genre_plot['Mystery']):

#    for key in val:

#        text = " ".join([text, "_".join(key.strip().split(" "))])

text = " ".join(["_".join(key.split(" ")) for key in genre_plot['Romance']]).strip()



plt.figure(figsize=(12,6))

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, 

                      max_words=200).generate(text)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

#plt.title("Wordcloud for features", fontsize=30)

plt.axis("off")

plt.show()