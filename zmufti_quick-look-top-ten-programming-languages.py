import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.integrate import simps

from numpy import trapz

from numpy import mean

from wordcloud import WordCloud

%matplotlib inline





# Keeps pandas from truncating output of df.head() too much.

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# csv to DataFrame

data = pd.read_csv("/kaggle/input/most-popular-programming-languages-since-2004/Most Popular Programming Languages from 2004 to 2020.csv")

data.head(1)
data["Date"] = pd.to_datetime(data["Date"])

data.set_index("Date", inplace = True)

data.head(1)
# Moving average with a window of 10

data_mov = pd.DataFrame()

for i in data.columns:

    data_mov[i] = data[i].rolling(window=10).mean()

#remove first 9 rows which are NaN now

data_mov = data_mov.iloc[9:]

data_mov.head(1)
data_mov['Python'].plot(kind="line",figsize=(15,10),color='skyblue',linewidth=10)

title = plt.title("Popularity over time: Python",fontsize=22)

title.set_position([.5, 1.009])

plt.xlabel("Time",fontsize=15,labelpad=20)

plt.ylabel("Popularity",fontsize=15, labelpad=20)

plt.fill_between(x=data_mov.index.values,y1=data_mov['Python'].values,color='lightseagreen')

axes = plt.gca()

axes.set_ylim([0,35])

plt.show()
programming_languages = []

popularity_mean = []

popularity_simps = []

popularity_trapz = []



for i in data.columns:

    programming_languages.append(i)

    popularity_mean.append(mean(data[i].values))

    popularity_simps.append(simps(data[i].values))

    popularity_trapz.append(trapz(data[i].values))



popularity_list = list(zip(popularity_mean,popularity_simps,popularity_trapz))

popularity_df = pd.DataFrame(popularity_list, columns=['Mean','Simpsons','Trapezoidal'],index=programming_languages)

popularity_df.head(2)
top_ten_mean = popularity_df.nlargest(10,'Mean')['Mean']

top_ten_simps = popularity_df.nlargest(10,'Simpsons')['Simpsons']

top_ten_trapz = popularity_df.nlargest(10,'Trapezoidal')['Trapezoidal']

results_list = [top_ten_mean,top_ten_simps,top_ten_trapz]
def wordclouds_tiled(n, df_list):

    plt.figure(figsize=(12,12))

    j = np.ceil(n/3)

    subtitles = ["Mean","Simpson's Rule","Trapezoidal"]

    for t in range(n):

        i=t+1

        title = plt.subplot(j, 3, i).set_title("Top ten: " + subtitles[t])

        plt.plot()

        title.set_position([.5, 1.1])

        plt.imshow(WordCloud(background_color="white",width=200,height=200,prefer_horizontal=1,colormap='Dark2').generate_from_frequencies(frequencies=df_list[t]))

        plt.axis("off")

    plt.show()



wordclouds_tiled(3,results_list)
top_ten_simps.sort_values(ascending=True).plot(kind="bar",figsize=(12,5),color='lightseagreen',linewidth=10)

title = plt.title("Top ten programming languages (since 2004)",fontsize=22)

title.set_position([.5, 1.009])

plt.xticks(size=12,rotation=0, horizontalalignment="center")

plt.xlabel("Programming Language",fontsize=15,labelpad=20)

plt.ylabel("Popularity",fontsize=15, labelpad=20)

plt.show()
top_tens = list(top_ten_simps.index)



data_mov[top_tens].plot(kind="line",figsize=(11,8),linewidth=3.5,legend=None)

title = plt.title("Programming language popularity over time",fontsize=22)

title.set_position([.5, 1.009])



plt.xlabel("Time",fontsize=17,labelpad=20)

plt.ylabel("Popularity",fontsize=17, labelpad=20)

axes = plt.gca()

axes.set_ylim([0,35])

axes.set_xlim(['2005-04-01','2020-08-01'])



for squiggle, lang in zip(axes.lines, data_mov[top_tens].columns):

    y_val = squiggle.get_ydata()[-1]

    if lang == 'PHP':

        axes.annotate(lang, xy=(1,y_val), xytext=(5,5), color=squiggle.get_color(), 

                xycoords = axes.get_yaxis_transform(), textcoords="offset points",

                size=10, va="center")

    elif lang == 'Visual Basic':

        axes.annotate(lang, xy=(1,y_val), xytext=(5,3), color=squiggle.get_color(), 

                xycoords = axes.get_yaxis_transform(), textcoords="offset points",

                size=10, va="center")

    else:

        axes.annotate(lang, xy=(1,y_val), xytext=(5,0), color=squiggle.get_color(), 

                xycoords = axes.get_yaxis_transform(), textcoords="offset points",

                size=10, va="center")

plt.show()