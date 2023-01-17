!pip install pycountry-convert
# installed for maps
!pip install folium
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import re
import nltk

import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

from wordcloud import WordCloud, STOPWORDS
import pycountry
import folium

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split

import re
from collections import Counter

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dil = pd.read_csv("/kaggle/input/the-social-dilemma-tweets/TheSocialDilemma.csv", parse_dates=['date','user_created'])
print("Shape of df: ",dil.shape)
print("Info of df: ",dil.info())
print("Describe df: ",dil.describe())
# Renamed the column names
dil.rename(columns={'user_friends':'friends',"user_followers":"followers","user_location":"location","user_verified":"verified","user_description":"description","user_favourites":"favourites"}, inplace=True)

# Cleaning location
dil['location'] = dil['location'].astype('str').str.split(".").str[0]
dil['location'] = dil['location'].str.replace(r'[^a-zA-Z,]', " ").str.strip()
dil['location'] = dil['location'].fillna('nan')

# Extracting day, hour, weekday
dil['month_day'] = dil.date.dt.day
dil['hour'] = dil.date.dt.hour
dil['week_day'] = dil.date.dt.weekday
dil['week_day'] = dil['week_day'].map({0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday',6:'Sunday'})

# calculating the age of user as on tweet day
dil['account_age'] = (dil['date'] - dil['user_created']).astype('str')

f = lambda x: x.split(" ")[0]
dil["account_age"] = dil["account_age"].apply(f)
dil["account_age"] = dil["account_age"].astype('int')

# calculating the age of user from today to the tweet day
dil['days_passed'] = (pd.datetime.now() - dil['date']).astype('str')

f = lambda x: x.split(" ")[0]
dil["days_passed"] = dil["days_passed"].apply(f)
dil["days_passed"] = dil["days_passed"].astype('int')

# Give nan hashtags 'no_tag' values
dil['hashtags'] = dil['hashtags'].fillna("no_tag")

# Checking columns with nulls
null_check = dil.isnull().sum() 
print(null_check[null_check > 0])
# Extract city and country for location
def separateCountry(loc):
    t=[]          
    if loc != 'nan':
        tokens = loc.split(",")
        if len(tokens) == 0:           
            return "no_country"
        elif len(tokens) == 2:           
            return tokens[1].strip()
        elif len(tokens) > 2:
            #print(">2 country.. ",tokens)            
            return tokens[-1].strip()
        else:           
            t = tokens
            tokens = ['no_city', t[0].strip()]            
            return tokens[1].strip()       
    else:
        tokens = ['no_city','no_country']       
        return tokens[1]  

def separateCity(loc):
    t=[]  
    if loc != 'nan':
        tokens = loc.split(",")
        if len(tokens) == 2:            
            return tokens[0].strip()
        elif len(tokens) > 2:
            t = tokens
            #print(">2 city.. ",tokens)
            tokens = [' '.join(t[0:-1]), t[-1].strip()]            
            return tokens[0].strip()
        else:            
            t = tokens
            tokens = ['no_city', t[0].strip()]            
            return tokens[0].strip()       
    else:
        tokens = ['no_city','no_country']        
        return tokens[0].strip()  

print("User Location: ",dil['location'].nunique(), dil['location'].unique())


dil['country'] = dil['location'].apply(separateCountry)
dil['city'] = dil['location'].apply(separateCity)
dil.head()
# Very noisy location data..hence manually imputing
# Correcting Atul Khatri's city, country
idx1 = dil[dil['user_name'] == 'Atul Khatri'].index
dil.loc[idx1,'city']='Mumbai'
dil.loc[idx1,'country']='India'

# Correcting Sreedhar Pillai's city, country
idx2 = dil[dil['user_name'] == 'Sreedhar Pillai'].index
dil.loc[idx2,'city']='Chennai'
dil.loc[idx2,'country']='India'

# Correcting Shiv Aroor's city, country
idx3 = dil[dil['user_name'] == 'Shiv Aroor'].index
dil.loc[idx3,'city']='Delhi'
dil.loc[idx3,'country']='India'

# Correcting Rahul Bose's city, country
idx3 = dil[dil['user_name'] == 'Rahul Bose'].index
dil.loc[idx3,'city']='Mumbai'
dil.loc[idx3,'country']='India'

# Correcting Rotten Tomatoes's city, country
idx4 = dil[dil['user_name'] == 'Rotten Tomatoes'].index
dil.loc[idx4,'city']='Los Angeles'
dil.loc[idx4,'country']='USA'

# Correcting diddy's city, country
idx5 = dil[dil['user_name'] == 'Diddy'].index
dil.loc[idx5,'city']='CA'
dil.loc[idx5,'country']='USA'

# Correcting E_L_James's city, country
idx6 = dil[dil['user_name'] == 'E_L_James'].index
dil.loc[idx6,'city']='West London'
dil.loc[idx6,'country']='England'

# Correcting tyler oakley's city, country
idx7 = dil[dil['user_name'] == 'tyler oakley'].index
dil.loc[idx7,'city']='NYC'
dil.loc[idx7,'country']='USA'

# Correcting Arianna Huffington's city, country
idx1 = dil[dil['user_name'] == 'Arianna Huffington'].index
dil.loc[idx1,'city']='AZ'
dil.loc[idx1,'country']='USA'

# Correcting BobSaget's city, country
idx7 = dil[dil['user_name'] == 'bob saget'].index
dil.loc[idx7,'city']='NJ'
dil.loc[idx7,'country']='USA'

# Correcting Sophie C's city, country
idx7 = dil[dil['user_name'] == 'Sophie C'].index
dil.loc[idx7,'city'] = 'Hyderabad'
dil.loc[idx7,'country']='India'

# Correcting DuckDuckGo's city, country
idx7 = dil[dil['user_name'] == 'DuckDuckGo'].index
dil.loc[idx7,'city'] = 'Pennsylvania'
dil.loc[idx7,'country']='USA'

# Correcting DuckDuckGo's city, country
idx7 = dil[dil['user_name'] == 'VOGUE India'].index
dil.loc[idx7,'city'] = 'Bengaluru'

# Correcting DNA's city, country
idx7 = dil[dil['user_name'] == 'DNA'].index
dil.loc[idx7,'city'] = 'Pune'


def correct_city_country(df):    
    if df.lower() == 'california' or df.lower() == 'usa' or df.lower() == 'new york' or df.lower() == 'los angeles' or df.lower() == 'texas' or df.lower() == 'mi' or df.lower() == 'oh' or df.lower() == 'va' or df.lower() == 'pa' or df.lower() == 'az' or df.lower() == 'or' or df.lower() == 'co' or df.lower() == 'fl' or df.lower() == 'ma'or df.lower() == 'dc' or df.lower() == 'nc' or df.lower() == 'il'  or df.lower() == 'united states' or df.lower() == 'tn' or df.lower() == 'brooklyn' or df.lower() == 'pittsburgh' or df.lower() == 'in' or df.lower() == 'wa' or df.lower() == 'oklahoma city' or df.lower() == 'ny' or df.lower() == 'tx' or df.lower() == 'ca' or df.lower() == 'ga':        
        df = 'United States'
    elif df.lower() == 'new south wales' or df.lower() == 'victoria':        
        df = 'Australia'
    elif df.lower() == 'british columbia' or df.lower() == 'ontario':        
        df = 'Canada'
    elif df.lower() == 'barcelona'  or df.lower() == 'comunidad de madrid':
        df = 'Spain'
    elif df.lower() == 'hamirpur' or df.lower() == 'delhi' or df.lower() == 'bengaluru' or df.lower() == 'new delhi' or df.lower() == 'mumbai'or df.lower() == 'rajasthan':
        df = 'India'
    elif df.lower() == 'london' or df.lower() == 'kent'  or df.lower() == 'united kingdom':
        df = 'UK'
    elif df.lower() == '':
        df = 'no_country'
    return df
    
dil['country'] = dil['country'].apply(lambda x : correct_city_country(x))

def get_country_code(co):
    mapping = {country.name: country.alpha_2 for country in pycountry.countries}    
    return mapping.get(co)
    
dil['code'] = dil['country'].map(get_country_code)
source_df = dil.groupby(['source']).agg('count').reset_index().rename(columns={'user_name':'count'})
source_df = source_df.sort_values(by=['count'],ascending=0)
source_df = source_df.drop(['month_day','favourites','location','user_created','description','followers','friends','verified','days_passed','hour','date','hashtags','text','is_retweet','week_day','account_age','city','country'],axis=1)

top_source_df = source_df[:6]
fig = px.bar(top_source_df, x='source', y='count', hover_data=['source', 'count'], height=400)
fig.show()
week_day_df = dil.groupby(['week_day']).agg('count').reset_index().rename(columns={'week_day':"week_day",'user_name':'count'})
week_day_df = week_day_df.sort_values(by=['count'],ascending=0)
week_day_df = week_day_df.drop(['code','Sentiment','month_day','favourites','location','user_created','description','followers','friends','verified','days_passed','hour','date','hashtags','text','is_retweet','source','account_age','city','country'],axis=1)
week_day_df.style.background_gradient(cmap='jet_r', subset=pd.IndexSlice[:, ['count']])
month_df = dil.groupby(['month_day']).agg('count').reset_index().rename(columns={"user_name":"count"})
month_df = month_df.drop(['code','Sentiment','favourites','location','user_created','description','followers','friends','verified','week_day','days_passed','hour','date','hashtags','text','is_retweet','source','account_age','city','country'],axis=1)

month_df.style.background_gradient(cmap='rainbow_r', subset=pd.IndexSlice[:, ['count']])
fig=go.Figure(go.Scatter(x=month_df['month_day'],
                                y=month_df['count'],
                               mode='markers+lines',
                               name="Submissions",
                               marker_color='dodgerblue'))

fig.update_layout(title_text='Tweets per Day',template="plotly_dark",title_x=0.5)
fig.show()
followers_df = dil.groupby(['followers','friends','favourites','city','country','account_age','Sentiment'])['user_name'].agg(sum).reset_index()
followers_df = followers_df.sort_values(by="followers",ascending=False)
followers_df[:30]

top_followers_df = followers_df[:20]
top_followers_df.style.background_gradient(cmap='viridis')
top30_followers_df = followers_df[:30]
fig = px.bar(top30_followers_df, x="user_name", y="account_age", color="Sentiment", title="Top 30 Influencers and their sentiments", 
             labels={"user_name": "User Name", "account_age": "account_age"},)
fig.show()
country_df = dil['country'].value_counts().to_frame().reset_index().rename(columns={'index':'country','country':'count'})

fig = go.Figure(go.Bar(
    x=country_df['country'][:10],y=country_df['count'][:10],
    marker={'color': country_df['count'][:10], 
    'colorscale': 'greens'},  
    text=country_df['count'][:10],
    textposition = "outside",
))
fig.update_layout(title_text='Top Countries with most tweets',xaxis_title="Countries",
                  yaxis_title="Number of Tweets",template="plotly_dark",height=700,title_x=0.5)

fig.show()
# Using Folium Maps

top_followers_df['Lat'] = [36.8, 45.7, 42.7,42.7,19.1 , 34.1, 34.5,19.1,19.1,19.1, 17.3,40.1,34.2 , 18.5, 18.1,19.1,17.1 , 41,12.9 , 22.6]
top_followers_df['Long'] = [-110.4, -84,-70,-64,72.8 , -111.2,-111.1 , 72.8,72.8,72.8, 78.4, -74.4,-118.2, 77.6, 73.8 ,71.8,70.8, -77, 77.5, 88.4]


world_map = folium.Map(location=[10,0], tiles="cartodbpositron", zoom_start=2,max_zoom=6,min_zoom=2)
for i in range(0,len(top_followers_df)):
    
    folium.Circle(
        location=[top_followers_df.iloc[i]['Lat'], top_followers_df.iloc[i]['Long']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+top_followers_df.iloc[i]['country']+"</h5>"+                    
                    "<div style='text-align:center;'>"+str((top_followers_df.iloc[i]['city']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+ "</ul>"
        , radius=(int((np.log(200+1.00001)))+0.2)*50000,
        color='#ff6600', fill_color='#ff8533', fill=True).add_to(world_map)

world_map
fig = plt.figure(figsize = (10, 5))

user_verified_df = dil['verified'].value_counts().to_frame().reset_index()
user_verified_df.columns = ['verified','counts']
user_verified_df['verified'] = user_verified_df['verified'].map({False:0,True:1})

fig = px.bar(user_verified_df, x='verified', y='counts',width=600, height=400)
fig.show()
ht = dict()
def create_hashtag_dict(hashtags):
    
    hashtags = hashtags.replace('[', "").strip()
    hashtags = hashtags.replace(']', "").strip()
    hashtags = hashtags.replace("'", "").strip()
    tags_list = hashtags.split(",")
    
    length = int(len(tags_list))
    
    for l in range(length):
        key = tags_list[l].strip()
        
        if key in ht.keys(): 
            ht[key] += 1
        else:
            ht[key] = 1  
    return ht  
               
hash_dict = dil['hashtags'].map(create_hashtag_dict)

sorted_hash_dict = {k: v for k, v in sorted(hash_dict[0].items(), key=lambda item: item[1], reverse=True)}
del sorted_hash_dict['no_tag'] 


tags_df = pd.DataFrame.from_dict(sorted_hash_dict, orient='index' ).reset_index().rename(columns={'index':"hashtags",0:"count"})

top_tags_df = tags_df[:20]
fig = px.bar(top_tags_df, x='hashtags', y='count',  hover_data=['hashtags', 'count'], color='count',height=400, title="Hash Tags Trending")
fig.show()
dil['text'] = dil['text'].apply(lambda x: x.lower())

stopword = nltk.corpus.stopwords.words('english')
stopword.extend(['thank','damn','always','might','well','smfh','li','yall','u','r','nt','ok','i', 'must','please','knew','go','brb','m', 'even','much','yes','hi','wow','the', 'frm','ah','us','of', 'on','also','us','okey','one', 'you', 'me', 'my', 'haa', 'erm','hey','okay', 'in', 'with', 'and', 'we', 'don','day', 'amp','re'])

tags_array = []
hashs_array = []
urls_array = []

def separate_url_tag(txt):    
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
    txt = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',txt)
    txt = re.sub(r'\bamp\b|\bthi\b|\bha\b',' ',txt)
    if urls :
        urls_array.append(" ".join(urls))
    
    tags = re.findall(r"@(\w+)", txt)
    txt = re.sub(r"@(\w+)", '',txt)
    if tags :
        tags_array.append(" ".join(tags))
    
    hashs = re.findall(r"#(\w+)", txt)
    txt = re.sub(r"#(\w+)", '',txt)
    hashs_array.append(" ".join(hashs))
    
    txt = re.sub('\d+', '',txt)
    txt = re.findall('\w+', txt)
    
    txt = [word for word in txt if word not in stopword]
    txt = " ".join(txt)
    return txt

dil['cleaned_text'] = dil['text'].map(separate_url_tag)
t_dict = {}
def create_tags_dict(tags_list):    
    length = int(len(tags_list))    
    for l in range(length):
        key = tags_list[l]
        
        if key in t_dict.keys(): 
            t_dict[key] += 1
        else:
            t_dict[key] = 1  
    return t_dict  

tags_dict = create_tags_dict(tags_array)
tags_dict

sorted_tags_dict = {k: v for k, v in sorted(tags_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_tags_dict

tag_df = pd.DataFrame.from_dict(sorted_tags_dict, orient='index' ).reset_index().rename(columns={'index':"tags",0:"count"})#,columns=['tags', 'count'])

top_tag_df = tag_df[:20]
fig = px.bar(top_tag_df, x='tags', y='count',  hover_data=['tags', 'count'], color='count',height=400, title="Mentions used")
fig.show()
l_dict = {}
def create_links_dict(tags_list):    
    length = int(len(tags_list))    
    for l in range(length):
        key = tags_list[l]
        
        if key in l_dict.keys(): 
            l_dict[key] += 1
        else:
            l_dict[key] = 1  
    return l_dict  

links_dict = create_links_dict(urls_array)
links_dict

sorted_links_dict = {k: v for k, v in sorted(links_dict.items(), key=lambda item: item[1], reverse=True)}
sorted_links_dict

links_df = pd.DataFrame.from_dict(sorted_links_dict, orient='index' ).reset_index().rename(columns={'index':"links",0:"count"})#,columns=['tags', 'count'])

top_links_df = links_df[:20]
fig = px.bar(top_links_df, x='links', y='count',  hover_data=['links', 'count'], color='count',height=400, title="Links used")
fig.show()
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

def stemming_lemmatizing(text):
    text = [ps.stem(word) for word in text]
    text = [wn.lemmatize(word) for word in text]
    return text


# Generating Word Clouds
stopwords = set(STOPWORDS)
stopwords.update(["tweet", "please"])
wc = WordCloud(width=1400, height=800, min_word_length=4, stopwords= stopwords, max_words=200).generate("".join(dil['cleaned_text']) )
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title('Most Used long Words in tweets',fontsize=35)
plt.show()
def ngram_df(corpus,nrange,n=None):
    vec = CountVectorizer(stop_words = 'english',ngram_range=nrange).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df
unigram_df=ngram_df(dil['cleaned_text'],(1,1),20)
bigram_df=ngram_df(dil['cleaned_text'],(2,2),20)
trigram_df=ngram_df(dil['cleaned_text'],(3,3),20)
fig = make_subplots(
    rows=3, cols=1,subplot_titles=("Unigram","Bigram",'Trigram'),
    specs=[[{"type": "scatter"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]
          ])

fig.add_trace(go.Bar(
    y=unigram_df['text'][::-1],
    x=unigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=unigram_df['count'],
    textposition = "outside",
    orientation="h",
    name="Months",
),row=1,col=1)

fig.add_trace(go.Bar(
    y=bigram_df['text'][::-1],
    x=bigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=bigram_df['count'],
     name="Days",
    textposition = "outside",
    orientation="h",
),row=2,col=1)

fig.add_trace(go.Bar(
    y=trigram_df['text'][::-1],
    x=trigram_df['count'][::-1],
    marker={'color': "blue"},  
    text=trigram_df['count'],
     name="Days",
    orientation="h",
    textposition = "outside",
),row=3,col=1)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text='Top N Grams',xaxis_title=" ",yaxis_title=" ",
                  showlegend=False,title_x=0.5,height=1200, template='plotly_dark')
fig.show()
text_df = dil.copy()
text_df['text_length'] = text_df['cleaned_text'].map(lambda x : len(x))

fig = go.Figure(data=go.Violin(y=text_df['text_length'], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='teal', opacity=0.7, x0='Tweet Text Length'))

fig.update_layout(yaxis_zeroline=False,title="Distribution of Text length",template='ggplot2')
fig.show()
print("Average length of Positive Sentiment tweets : {}".format(round(text_df[text_df['Sentiment']== 'Positive']['text_length'].mean(),2)))
print("Average length of Neutral Sentiment tweets : {}".format(round(text_df[text_df['Sentiment']== 'Neutral']['text_length'].mean(),2)))
print("Average length of Negative Sentiment tweets : {}".format(round(text_df[text_df['Sentiment']=='Negative']['text_length'].mean(),2)))
fig = go.Figure()

fig.add_trace(go.Violin(y=text_df[text_df['Sentiment']== 'Positive']['text_length'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='limegreen', opacity=0.6,name="Positive", x0='Positive')
             )

fig.add_trace(go.Violin(y=text_df[text_df['Sentiment']== 'Neutral']['text_length'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='skyblue', opacity=0.6,name="Neutral", x0='Neutral')
             )

fig.add_trace(go.Violin(y=text_df[text_df['Sentiment']== 'Negative']['text_length'], box_visible=False, line_color='black',
                               meanline_visible=True, fillcolor='red', opacity=0.6,name="Negative", x0='Negative')
             )

fig.update_traces(box_visible=False, meanline_visible=True)
fig.update_layout(title_text="Violin - Tweet Length",title_x=0.5)

fig.show()
sentiment_country_pos_df=text_df[text_df['Sentiment']=='Positive']['country'].value_counts().reset_index().rename(columns={'index':'country','country':'count'})
top15_pos_sentiment = sentiment_country_pos_df[:15]

# data is very noisy, so imputed some values
top15_pos_sentiment.insert(loc=2,column= "code", value=['USA','USA','IND','GBR','GBR','CAN','AUS','ZAF','IRL','PHL','GBR','KEN','PAK','DEU','IND']) 
fig = go.Figure(data=go.Choropleth(
    locations = top15_pos_sentiment['code'],
    z = top15_pos_sentiment['count'],   text = top15_pos_sentiment['country'],
    colorscale = 'reds', autocolorscale=False,  reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.8,     colorbar_title = '# of Tweets', ))

fig.update_layout(
    title_text='Positive Tweets over the world',title_x=0.5,
    geo=dict(showframe=True, showcoastlines=False, projection_type='equirectangular',    ) )

fig.show()
sentiment_country_pos_df = text_df[text_df['Sentiment']=='Positive']['cleaned_text'].reset_index()#.rename(columns={'country':'count'})
sentiment_country_pos_df

# Generating Word Clouds
stopwords = set(STOPWORDS)
stopwords.update(["tweet", "please"])
wc = WordCloud(width=1600, height=800, min_word_length=4, stopwords= stopwords, max_words=200).generate("".join(sentiment_country_pos_df['cleaned_text']) )
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title('Common Words in Positive tweets',fontsize=35)
plt.show()
sentiment_country_neg_df=text_df[text_df['Sentiment']=='Negative']['country'].value_counts().reset_index().rename(columns={'index':'country','country':'count'})
top15_neg_sentiment = sentiment_country_neg_df[:15]
top15_neg_sentiment

# data is very noisy, so imputed some values
top15_neg_sentiment.insert(loc=2,column= "code", value=['USA','USA','IND','GBR','GBR','CAN','ZAF','AUS','IRL','GBR','PHL','IND','IDN','ESP','USA']) 
top15_neg_sentiment
fig = go.Figure(data=go.Choropleth(
    locations = top15_neg_sentiment['code'],
    z = top15_neg_sentiment['count'],  text = top15_neg_sentiment['country'],
    colorscale = 'viridis',   autocolorscale=False,     reversescale=False,    marker_line_color='darkgray',
    marker_line_width=0.8,    colorbar_title = '# of Tweets',
))

fig.update_layout(
    title_text='Negative Tweets over the world',title_x=0.5,
    geo=dict(       showframe=True,         showcoastlines=False,        projection_type='equirectangular',
    ) )

fig.show()
sentiment_country_neg_df = text_df[text_df['Sentiment']=='Negative']['cleaned_text'].reset_index()
sentiment_country_neg_df

# Generating Word Clouds
stopwords = set(STOPWORDS)
stopwords.update(["tweet", "please"])
wc = WordCloud(width=1600, height=800, min_word_length=4, stopwords= stopwords, max_words=200).generate("".join(sentiment_country_pos_df['cleaned_text']) )
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title('Common Words in Negative tweets',fontsize=35)
plt.show()