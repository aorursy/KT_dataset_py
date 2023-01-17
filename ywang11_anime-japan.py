import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from wordcloud import WordCloud, ImageColorGenerator
from plotly import graph_objs as go, offline as offline, plotly as py
offline.init_notebook_mode(connected=True)
% matplotlib inline
# read in market share data
df_market = pd.read_excel('../input/World_movie_market_2017.xlsx', header=None, names=['country','box_office'])
# calculate revenue of all the other countries besides the top 3
revenue_others = df_market.box_office[3:].sum()
df_market.iloc[21,1] = revenue_others
df_market = df_market.iloc[[0,1,2,21],:]
colors = ['rgba(0, 109, 204, 1)', 'rgba(204, 102, 0, 1)', 'rgba(119, 204, 0, 1)']
for i in range(0,19):
    colors.append('rgba(230, 230, 230, 1)')

fig = {
  'data': [
    {
        'values': list(df_market['box_office'].values.flatten()),
        'labels': list(df_market['country'].values.flatten()),
        'marker': {'colors': colors},
        'hoverinfo':'label+percent',
        'textinfo':'label+percent',
        'textposition':'outside',
        'hole': .7,
        'type': 'pie',
        'sort':False,
        'showlegend':False,
        'textfont': 
            {
                'color':'rgba(0, 0, 0, 1)', 
                'size':20
            }
    },
    ],
    'layout': {
        'title':'2017 Global Movie Markets Share by Box Office',
        'titlefont': {
          'size': 30},
        'margin':{
            't':150,
            'b':80
        },
        'annotations': [
            {
                'font': {
                    'size': 35
                },
                'showarrow': False,
                'text': 'Top 3',
                'x': 0.5,
                'y': 0.75
            },
            {
                'font': {
                    'size': 35
                },
                'showarrow': False,
                'text': '>',
                'x': 0.5,
                'y': 0.55
            },
            {
                'font': {
                    'size': 50
                },
                'showarrow': False,
                'text': '50%',
                'x': 0.5,
                'y': 0.40
            },
      ]
    }
}


offline.iplot(fig)
# read in data and clean
df_rank = pd.read_excel('../input/movie_top3_top20.xlsx')

df_rank.replace('Adventrue','Adventure',inplace=True)
df_rank.replace('Comic-Based','Comic',inplace=True)
df_rank.replace('Music','Musical',inplace=True)
df_rank.replace('\xa0War', 'War', inplace=True)
df_rank.country.replace('US', 'US/Canada', inplace=True)
# attract movie genre, country and domestic gross
country_lis = ['Japan', 'China', 'US/Canada']
df_country_genre = pd.DataFrame()
df_country_genre_ratio = pd.DataFrame()

for country in country_lis:
    df_country = df_rank[df_rank.country==country].iloc[:,[1,2,7,8,9,10,11,12,13,14]].reset_index(drop=True)
    country_top20_total = df_rank.loc[df_rank.country==country,'gross'].sum()
    for i in range(2,9):
        df_genre = df_country.iloc[:,[0,1,i]].reset_index(drop=True)
        df_genre.columns = ['gross','country','genre']
        df_country_genre = pd.concat([df_country_genre,df_genre], axis=0, sort=True).reset_index(drop=True).dropna(axis=0)
    df_country_genre = df_country_genre.groupby(['country','genre'], as_index=False).sum()
# Caculate ratio of genre gross by total gross
df_country_genre_total = df_country_genre[['country','gross']].groupby('country', as_index=False).sum() 
df_country_genre_total.rename(columns={'gross':'total'}, inplace=True)
df_country_genre_ratio = df_country_genre.merge(df_country_genre_total, on='country',how='left')
df_country_genre_ratio['ratio'] = df_country_genre_ratio['gross']/df_country_genre_ratio['total']
# Restructure data 
genre_lis = df_country_genre.genre.unique()
genre_ratio_lis = []
df_genre_ratio = pd.DataFrame()

for i in range(0,len(country_lis)):
    genre_ratio_lis = []
    country = country_lis[i]
    genre_ratio_lis.append(country)
    for j in range(0, len(genre_lis)):
        if len(df_country_genre_ratio.loc[
            (df_country_genre_ratio.country==country_lis[i])& (df_country_genre_ratio.genre==genre_lis[j])])>0:
            ratio = df_country_genre_ratio.loc[
                (df_country_genre_ratio.country==country_lis[i])& (df_country_genre_ratio.genre==genre_lis[j]), 'ratio'].values[0]
        else:
            ratio = 0
        genre_ratio_lis.append(ratio)
    genre_ratio_s = pd.Series(genre_ratio_lis, name=False)
    df_genre_ratio = pd.concat((df_genre_ratio, genre_ratio_s), axis=1)
    

df_genre_ratio.columns = df_genre_ratio.iloc[0,:]
df_genre_ratio = df_genre_ratio.drop(0).reset_index()
df_genre_ratio['genre'] = pd.Series(genre_lis)
genre_lis = df_country_genre.genre.unique()
colors = {'Action':'rgba(194, 0, 0, 1)', 'Adventure':'rgba(37, 0, 204, 1)', 'Animation':'rgba(204, 0, 139, 1)',
              'Comedy':'rgba(48, 204, 0, 1)', 'Comic':'rgba(226, 145, 24, 1)', 'Drama':'rgba(204, 71, 0, 1)', 
              'Family':'rgba(116, 0, 204, 1)', 'Fantasy':'rgba(204, 146, 0, 1)', 'Sci-Fi':'rgba(63, 0, 158, 1)', 
             'Thriller':'rgba(147, 21, 21, 1)', 'Biography':'rgba(230, 230, 230, 1)', 'Crime':'rgba(220, 220, 220, 1)',
             'History':'rgba(200, 200, 200, 1)', 'Horror':'rgba(180, 180, 180, 1)', 'Musical':'rgba(160, 160, 160, 1)', 
              'Mystery':'rgba(140, 140, 140, 1)', 'Romance':'rgba(120, 120, 120, 1)', 'War':'rgba(100, 100, 100, 1)',
              'Sport':'rgba(80, 80, 80, 1)' }


trace = [go.Bar(x=list(df_genre_ratio.loc[df_genre_ratio.genre==genre,country_lis].values.flatten()), 
                y=country_lis, name=genre, orientation = 'h', marker=dict(color=colors[genre])) for genre in genre_lis]
data = trace
layout = go.Layout(title='Contributions of Movie Genres to Total Gross($) by Market', titlefont=dict(size=20),
                   yaxis=dict(title='Market', showline=True, titlefont=dict(size=16), tickfont=dict(size=15)),
                   xaxis=dict(title='Accumulated Ratio of Total Gross($)', showline=True, 
                              titlefont=dict(size=16), tickfont=dict(size=15),tickformat=".0%"), 
                   showlegend=False, barmode='stack', margin=dict(l=120))


annotations = []

for country in country_lis:
    for i in range(0, len(df_genre_ratio)):
        ratio = df_genre_ratio[country][i]
        genre = df_genre_ratio['genre'][i]
        if ratio>0.08:
            x = df_genre_ratio[country][0:i+1].sum()-0.048
            text = df_genre_ratio['genre'][i]
            annotations.append(dict(x=x, y=country, text=text,
                                  font=dict(family='Calibri', size=15,
                                  color='rgba(245, 246, 249, 1)'),
                                  showarrow=False))
layout['annotations'] = annotations
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
# Restructure data to row as country, genre as column
key_genre_lis = ['Adventure','Animation','Fantasy']
df_country_ratio = pd.DataFrame()

for i in range(0,len(key_genre_lis)):
    country_ratio_lis = []
    genre = key_genre_lis[i]
    country_ratio_lis.append(genre)
    for j in range(0, len(country_lis)):
        if len(df_country_genre_ratio.loc[
            (df_country_genre_ratio.genre==key_genre_lis[i])& (df_country_genre_ratio.country==country_lis[j])])>0:
            ratio = df_country_genre_ratio.loc[
                (df_country_genre_ratio.genre==key_genre_lis[i])& (
                    df_country_genre_ratio.country==country_lis[j]), 'ratio'].values[0]
        else:
            ratio = 0
        country_ratio_lis.append(ratio)
    country_ratio_s = pd.Series(country_ratio_lis, name=False)
    df_country_ratio = pd.concat((df_country_ratio, country_ratio_s), axis=1)
    

df_country_ratio.columns = df_country_ratio.iloc[0,:]
df_country_ratio = df_country_ratio.drop(0).reset_index()
df_country_ratio['country'] = pd.Series(country_lis)
trace = [go.Bar(x=list(df_country_ratio.loc[df_country_ratio.country==country,key_genre_lis].values.flatten()), 
                y=key_genre_lis, name=country, orientation = 'h') for country in country_lis]
data = trace
layout = go.Layout(title="Contributions of Key Genres to Totle Box Office Revenue by Market",
                   titlefont=dict(size=20),
                   yaxis=dict(showline=True, tickfont=dict(size=20)),
                   xaxis=dict(title='Percentage of Box Office Revenue($)', titlefont=dict(size=20),
                              showline=False, tickfont=dict(size=20),tickformat=".0%"), 
                   showlegend=True,
                   legend=dict(font=dict(size=20)), margin=dict(l=100))


fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
df_rank['animation'] = np.where(df_rank.genre_1=='Animation', 1, 
                                np.where(df_rank.genre_2=='Animation', 1, 
                                         np.where(df_rank.genre_3=='Animation', 1, 
                                                  np.where(df_rank.genre_4=='Animation', 1, 
                                                           np.where(df_rank.genre_5=='Animation', 1, 
                                                                    np.where(df_rank.genre_6=='Animation', 1, 
                                                                             np.where(df_rank.genre_7=='Animation', 1,
                                                                                      np.where(df_rank.genre_8=='Animation', 1, 0 ) )))))))
df_animation = df_rank[['title','country','animation']].groupby(['country','animation'], as_index=False).count()
df_animation.rename(columns={'title':'ani_proportion'}, inplace=True)
df_animation['ani_proportion'] = df_animation['ani_proportion']/20
df_animation['non_ani_proportion'] = 1-df_animation.ani_proportion
df_animation = df_animation.drop('animation', axis=1)
df_animation = df_animation.iloc[[1,3,5],:].transpose()
df_animation.columns = df_animation.iloc[0,:]
df_animation.drop('country', axis=0, inplace=True)
x = list(df_animation.loc[df_animation.index=='ani_proportion',country_lis].values.flatten())
y = country_lis

color_japan = 'rgba(204, 0, 139, 1)'
color_others = 'rgba(140, 140, 140, 1)'
text_color_japan = 'rgba(204, 0, 0, 1)'
text_color_others = 'rgba(140, 140, 140, 1)'

colors  = [color_japan if y[i]=='Japan' else color_others for i in range(0,len(y))]
text_size = [30 if y[i]=='Japan' else 20 for i in range(0,len(y))]
text_color  = [text_color_japan if y[i]=='Japan' else text_color_others for i in range(0,len(y))]


trace = [go.Bar(x=x, y=y, orientation = 'h', text=[str(round((ratio)*100))+'%' for ratio in x], textposition='outside',
                textfont=dict(size=text_size, color=text_color), marker=dict(color=colors))] 

data = trace
layout = go.Layout(title='Year 2017 Percentage of Animation Movies of Top 20 movies among Top 3 Markets',
                   font=dict(size=13),
                   yaxis=dict(showline=True, tickfont=dict(size=15)),
                   xaxis=dict(title='Percentage of Top 20 Movies', showline=False, 
                              titlefont=dict(size=15), ticks='',showticklabels=False, range=(0,0.5)),
                   margin=dict(l=200)
                  )
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
# replace punctuations 
replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
df_rank.plots_summary = df_rank.plots_summary.apply(lambda txt: txt.translate(replace_punctuation))

# lower case
df_rank.plots_summary = df_rank.plots_summary.apply(lambda txt: txt.lower())
# plots summary by country
plots_summary_us = df_rank.loc[df_rank.country=='US/Canada'].plots_summary.sum(axis=0)
plots_summary_china = df_rank.loc[df_rank.country=='China'].plots_summary.sum(axis=0)
plots_summary_japan = df_rank.loc[df_rank.country=='Japan'].plots_summary.sum(axis=0)
plots_summary_uk = df_rank.loc[df_rank.country=='Britain'].plots_summary.sum(axis=0)
plots_summary_india = df_rank.loc[df_rank.country=='India'].plots_summary.sum(axis=0)
stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", 
              "again", "against", "all", "almost", "alone", "along", "already", 
              "also","although","always","am","among", "amongst", "amoungst", "amount",  
              "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", 
              "are", "around", "as",  "at", "back","be","became", "because","become","becomes", 
              "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", 
              "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", 
              "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", 
              "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", 
              "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", 
              "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", 
              "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", 
              "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
              "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", 
              "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", 
              "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
              "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", 
              "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
              "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", 
              "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", 
              "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", 
              "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", 
              "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", 
              "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", 
              "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", 
              "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", 
              "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", 
              "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", 
              "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
              "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
              "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the",'yes',
              'character','reference','feng']
wc = WordCloud(background_color="white", max_words=6,
               stopwords=stop_words, width=1280, height=628)
wc.generate(plots_summary_us)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
wc.generate(plots_summary_china)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
wc.generate(plots_summary_japan)
plt.figure(figsize=(10,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
df_market_full = pd.read_excel('../input/World_movie_market_2017.xlsx', header=None, names=['country','box_office'])
fig = {
  "data": [
    {
      "values": list(df_market_full['box_office'].values.flatten()),
      "labels": list(df_market_full['country'].values.flatten()),
      "hoverinfo":"label+percent",
        'textinfo':'label+percent',
        'textposition':'outside',
      "hole": .7,
      "type": "pie",
        'sort':False,
        'showlegend':True
    },
    ],
  "layout": {
        "title":"2017 Global Movie Markets Share",

    }
}
    

offline.iplot(fig)
trace = [go.Bar(x=list(df_genre_ratio.loc[df_genre_ratio.genre==genre,country_lis].values.flatten()), 
                y=country_lis, name=genre, orientation = 'h') for genre in genre_lis]
data = trace
layout = go.Layout(title='Preference on Movie Genre by Market', font=dict(size=17),
                   yaxis=dict(title='Market', showline=True, titlefont=dict(size=16), tickfont=dict(size=15)),
                   xaxis=dict(title='Accumulated Ratio of Total Gross($)', showline=True, 
                              titlefont=dict(size=16), tickfont=dict(size=15), tickformat=".0%"), 
                   legend=dict(font=dict(size=10)),barmode='stack', margin=dict(l=120))


annotations = []

for country in country_lis:
    for i in range(0, len(df_genre_ratio)):
        ratio = df_genre_ratio[country][i]
        genre = df_genre_ratio['genre'][i]
        if ratio>0.07:
            x = df_genre_ratio[country][0:i+1].sum()-0.05
            text = df_genre_ratio['genre'][i]
            annotations.append(dict(x=x, y=country, text=text,
                                  font=dict(family='Calibri', size=17,
                                  color='rgba(245, 246, 249, 1)'),
                                  showarrow=False))
layout['annotations'] = annotations
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig)
