!pip install comet_ml

!pip install seaborn

!pip install wordcloud

!pip install emoji

!pip install pyspellchecker

!pip install ftfy
# Package for creating an experiment in Comet

import comet_ml

from comet_ml import Experiment



# Setting the API key (saved as environment variable)

experiment = Experiment(api_key="upOwchWrd7H1e6VEnWKW7PSvz", project_name="classification-predict", workspace="team-rm1")
# Packages for data analysis

import numpy as np

import pandas as pd

from time import time



# Packages for visualisations

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from PIL import Image



# Packages for preprocessing

import re

from nltk import word_tokenize

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.probability import FreqDist

import emoji

from ftfy import fix_text

from spellchecker import SpellChecker 

from nltk.stem.snowball import SnowballStemmer

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import itertools

from sklearn.model_selection import train_test_split

from sklearn.utils import resample



# Packages for training models

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



# Packages for hyperparameter optimisation

from sklearn.model_selection import GridSearchCV



# Packages for evaluating model accuracy

from sklearn.metrics import f1_score 

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.preprocessing import label_binarize



# Packages for saving models

import pickle
# Import training dataset

df_train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

# Import testing dataset

df_test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')



# Set 'tweetid' as index

df_train.set_index('tweetid',inplace = True)

df_test.set_index('tweetid',inplace = True)



# Creat copy for EDA purposes

df_train_eda=df_train.copy()
df_train.info()
df_test.info()
# Function to extract sentiment

def sentiment_score(text):

    """ A function that determines the sentiment of a text string.



        Parameters

        ----------

        text: Text string.



        Returns

        -------

        sentiment:  String indicating the sentiment of the input string.

    """

    

    sid = SentimentIntensityAnalyzer()

    s = sid.polarity_scores(text)['compound']

    if s<-0.05:

        sentiment='negative'

    elif s>0.05:

        sentiment='positive'

    else:

        sentiment='neutral'

    

    return sentiment
# Extract all unique news related handles into a list

df_temp = df_train.copy()

df_temp.sort_index(inplace=True)

n_temp = [re.findall(r'@[\w]+',df_temp['message'].iloc[i]) for i,x in enumerate(df_temp['sentiment']) if x==2]

news = [x for x in n_temp if x!=[]]



# Only keep the unique values inside the list

news = sorted(list(set(itertools.chain.from_iterable(news))))
print(f'First 5 entries: {news[:5]} \nLast 5 entries: {news[-5:]}')
# Import dictionary of expanded hashtags

with open('../input/resources/hashtags.pkl', 'rb') as file:

    hashtags = pickle.load(file)
# Substitute hastags with separated words

def expand_hashtags(df,column_name):

    """ A funtion that expands the hashtag words into separate words.



        Parameters

        ----------

        df:          Dataframe containing the text column to be transformed.

        column_name: Name of the column containing the text data.



        Returns

        -------

        df:  Dataframe containg the updated text column

        

        Example

        -------

        #iamgreat returns 'i am great'

    """

    

    df[column_name] = df[column_name].str.lower()

    df[column_name] = df[column_name].apply(lambda x: re.sub(r"[#]",'',x))

    for word in hashtags.keys():

            df[column_name] = df[column_name].apply(lambda x: re.sub(word,hashtags[word]+' ',x))

    return df
df_train = expand_hashtags(df_train,'message')
# Dictionary of contracted words

contractions = {

"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"wasn't": "was not",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we'll":"we will",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

}
# Replace contracted words with full word

df_train['message'] = [' '.join([contractions[w.lower()] if w.lower() in contractions.keys() else w for w in raw.split()]) 

                       for raw in df_train['message']]
# Lower case all words to remove noise from Capital words. Capital words may be seen as different from lower case words

df_train['message'] = df_train['message'].str.lower()

df_train['message'] = df_train['message'].apply(lambda x: fix_text(x))

# Removing urls

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r'https\S+','url',x))

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r'www\S+', 'url',x))

# Replace emojis with their word meaning

df_train['message'] = df_train['message'].apply(lambda x: emoji.demojize(x))

# Replace shortened words with full words

short = {' BD ': ' Big Deal ',

 ' abt ':' about ',

 ' ab ': ' about ',

 ' fav ': ' favourite ',

 ' fab ': ' fabulous ',

 ' smh ': ' shaking my head ',

 ' u ': ' you ',

 ' c ': ' see ',

 ' anon ': ' anonymous ',

 ' ac ': ' aircon ',

 ' a/c ': ' aircon ',

 ' yo ':' year old ',

 ' n ':' and ',

 ' nd ':' and ',

 ' 2 ': ' to ',

 ' w ': ' with ',

 ' w/o ': ' without ',

 ' r ': ' are ',

 ' rip ':' rest in peace ',

 ' 4 ' : ' for ',

' BF ': ' Boyfriend ',

' BRB ': ' Be Right Back ',

' BTW ': ' By The Way ',

' GF ': ' Girlfriend ',

' HBD ': ' Happy Birthday ',

' JK ': ' Just Kidding ',

' K ':' Okay ',

' LMK ': ' Let Me Know ',

' LOL ': ' Laugh Out Loud ',

' HA ':' laugh ',

' MYOB ': ' Mind Your Own Business ',

' NBD ': ' No Big Deal ',

' NVM ': ' Nevermind ',

' Obv ':' Obviously ',

' Obvi ':' Obviously ',

' OMG ': ' Oh My God ',

' Pls ': ' Please ',

' Plz ': ' Please ',

' Q ': ' Question ', 

' QQ ': ' Quick Question ',

' RLY ': ' Really ',

' SRLSY ': ' Seriously ',

' TMI ': ' Too Much Information ',

' TY ': ' Thank You, ',

' TYVM ': ' Thank You Very Much ',

' YW ': ' You are Welcome ',

' FOMO ': ' Fear Of Missing Out ',

' FTFY ': ' Fixed This For You ',

' FTW ': ' For The Win ',

' FYA ': ' For Your Amusement ',

' FYE ': ' For Your Entertainment ',

' GTI ': ' Going Through It ',

' HTH ': ' Here to Help ',

' IRL ': ' In Real Life ',

' ICYMI ': ' In Case You Missed It ',

' ICYWW ': ' In Case You Were Wondering ',

' NBC ': ' Nobody Cares Though ',

' NTW ': ' Not To Worry ',

' OTD ': ' Of The Day ',

' OOTD ': ' Outfit Of The Day ',

' QOTD ': ' Quote of the Day ',

' FOTD ': ' Find Of the Day ',

' POIDH ': ' Pictures Or It Did ntt Happen ',

' YOLO ': ' You Only Live Once ',

' AFAIK ': ' As Far As I Know ',

' DGYF ': ' Dang Girl You Fine ',

' FWIW ': ' For What It is Worth ',

' IDC ': ' I Do not Care ',

' IDK ': ' I Do not Know ',

' IIRC ': ' If I Remember Correctly ',

' IMHO ': ' In My Honest Opinion ',

' IMO ': ' In My Opinion ',

' Jelly ': ' Jealous ',

' Jellz ': ' Jealous ',

' JSYK ': ' Just So You Know ',

' LMAO ': ' Laughing My Ass Off ',

' LMFAO ': ' Laughing My Fucking Ass Off ',

' NTS ': ' Note to Self ',

' ROFL ': ' Rolling On the Floor Laughing ',

' ROFLMAO ': ' Rolling On the Floor Laughing My Ass Off ',

' SMH ': ' Shaking My Head ',

' TBH ': ' To Be Honest ',

' TL;DR ':  ' Too Long; Did not Read ',

' TLDR ':  ' Too Long; Did not Read ',

' YGTR ': ' You Got That Right ',

' AYKMWTS ': ' Are You Kidding Me With This Shit ',

' BAMF ': ' Bad Ass Mother Fucker ',

' FFS ': ' For Fuck Sake ',

' FML ': ' Fuck My Life ',

' HYFR ': ' Hell Yeah Fucking Right ',

' IDGAF ': ' I Do not Give A Fuck ',

' NFW ': ' No Fucking Way ',

' PITA ': ' Pain In The Ass ',

' POS ': ' Piece of Shit ',

' SOL ': ' Shit Outta Luck ',

' STFU ': ' Shut the Fuck Up ',

' TF ': ' The Fuck ',

' WTF ': ' What The Fuck ',

' BFN ': ' Bye For Now ',

' CU ': ' See You ',

' IC ': ' I see ',

' CYL ': ' See You Later ',

' GTG ': ' Got to Go ',

' OMW ': ' On My Way ',

' RN ': ' Right Now ',

' TTYL ': ' Talk To You Later ',

' TYT ': ' Take Your time ',

' CC ': ' Carbon Copy ',

' CX ': ' Correction ',

' DM ': ' Direct Message ',

' FB ': ' Facebook ',

' FBF ': ' Flash-Back Friday ',

' FF ': ' Follow Friday ',

' HT ': ' Tipping my hat ',

' H/T ': ' Tipping my hat ',

' IG ': ' Instagram ',

' Insta ': ' Instagram ',

' MT ':' Modified Tweet ',

' OH ': ' Overheard ',

' PRT ': ' Partial Retweet ',

' RT ': ' Retweet ',

'rt ' : ' retweet ',

' SO ':' Shout Out ',

' S/O ': ' Shout Out ',

' TBT ': ' Throw-Back Thursday ',

' AWOL ': ' Away While Online ',

' BFF ': ' Best Friend Forever ',

' NSFW ': ' Not Safe For Work ',

' OG ': ' Original Gangster ',

' PSA ': ' Public Service Announcement ',

' PDA ': ' Public Display of Affection '}



short = dict((key.lower(), value.lower()) for key,value in short.items())
# Replacing shortened words with full words

for word in short.keys():

    df_train['message'] = df_train['message'].apply(lambda x: re.sub(word,short[word],x))

# Remove twitter non news related handles and @ symbol

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r'@', '', ' '.join([y for y in x.split() if y not in 

                                                                                     [z for z in re.findall(r'@[\w]*',x) 

                                                                                      if z not in news]])))
# Add sentiment

df_train['message'] = df_train['message'].apply(lambda x: x + ' ' + sentiment_score(x))  
# Remove punctuation

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r"[^A-Za-z ]*",'',x))

# Remove vowels repeated at least 3 times ex. Coooool > Cool

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r'([aeiou])\1+', r'\1\1', x))

# Replace sequence of 'h' and 'a', as well as 'lol' with 'laugh'

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r' ha([ha]) *', r'laugh', x))

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r' he([he]) *', r'laugh', x))

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r' lol([ol]) *', r'laugh', x))

df_train['message'] = df_train['message'].apply(lambda x: re.sub(r' lo([o])*l ', r'laugh', x))
def cleanup(raw):

    """ A function that 'cleans' tweet data. The text gets modified by:

        - being lower cased, 

        - removing urls, 

        - removing bad unicode,

        - replacing emojis with words,

        - removing twitter non news related handles,

        - removing punctuation,

        - removing vowels repeated at least 3 times,

        - replacing sequences of 'h' and 'a', as well as 'lol' with 'laugh',

        - adding sentiment



        Parameters

        ----------

        raw: Text string.



        Returns

        -------

        raw:  Modified clean string

    """

    

    # Convert to lowercase

    raw = raw.lower()

    

    # Fix strange characters

    raw = fix_text(raw)

    

    # Removing urls

    raw = re.sub(r'https\S+','url',raw)

    raw = re.sub(r'www\S+', 'url',raw)

    

    # Replace emojis with their word meaning

    raw = emoji.demojize(raw)



    # Remove twitter non news related handles

    raw = ' '.join([y for y in raw.split() if y not in [x for x in re.findall(r'@[\w]*',raw) if x not in news]])

    

    # Add sentiment

    raw = raw + ' ' + sentiment_score(raw)

    

    # Remove punctuation

    raw = re.sub(r"[^A-Za-z ]*",'',raw)

    

    # Remove vowels repeated at least 3 times ex. Coooool > Cool

    raw = re.sub(r'([aeiou])\1+', r'\1\1', raw)

    

    # Replace sequence of 'h' and 'a', as well as 'lol' with 'laugh'

    raw = re.sub(r' ha([ha]) *', r'laugh', raw)

    raw = re.sub(r' he([he]) *', r'laugh', raw)

    raw = re.sub(r' lol([ol]) *', r'laugh', raw)

    raw = re.sub(r' lo([o])*l ', r'laugh', raw)

    

    return raw

# Seperate hashtags

df_test = expand_hashtags(df_test,'message')
# Replace contracted words with full word

df_test['message'] = [' '.join([contractions[w.lower()] if w.lower() in contractions.keys() else w for w in raw.split()]) 

                      for raw in df_test['message']]
# Replacing shortened words with full words

for word in short.keys():

    df_test['message'] = df_test['message'].apply(lambda x: re.sub(word,short[word],x))

# Apply cleaning function

df_test['message'] = df_test['message'].apply(lambda x: cleanup(x))

spell = SpellChecker() 

# check for misspelled words

misspelled = df_train['message'].apply(lambda x: spell.unknown(x))

misspelled.isnull().mean()
# Checking for empty strings

blanks = [i for i,lb,tweet in df_train_eda.itertuples() if type(tweet) == str if tweet.isspace()]

blanks
# Checking for duplicates in tweets

df_train_eda[df_train_eda.duplicated(subset='message') == True].count()/len(df_train)*100
print('Number of tweets per sentiment class')

df_train_eda['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts()
# Plot the proportion of tweets per class

plot1 = plt.figure(figsize=(15,5))

names = ['Pro','News','Neutral','Anti']

perc = df_train_eda['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts()

perc.name = ''

perc.plot(kind='pie', labels=names, autopct='%1.1f%%')

plt.title('Proportion of tweets in each class',fontsize = 16)

plt.figtext(0.12, 0.1, 'figure 1: Percentage of tweets that are classified as either Anti, Pro, Neutral and News',

            horizontalalignment='left',fontsize = 14,style='italic')

plt.legend(df_train['sentiment'].replace({-1: 'Anti: Does not believe in man-made climate change',

                                          0:'Neutral: Neither believes nor refutes man-made climate change',

                                          1:'Pro:Believe in man-made climate change',

                                          2:'News: Factual News about climate change'}).value_counts().index,

           bbox_to_anchor=(2.3,0.7), loc="right")



plt.show()
# Create resampling function

def resampling(df, class1, class2):

    """ A function takes in a dataframe, a class to be resampled, and a class 

        thats observations are to be matched with.



        Parameters

        ----------

        df:     Dataframe to be resampled.

        class1: Integer of the class that is to be resampled.

        class2: Integer of the class whose length is used to resample class1.



        Returns

        -------

        df_resampled:  Resampled dataframe.

    """

    

    df_class1= df[df.sentiment==class1]

    df_class2 = df[df.sentiment==class2]

    df_new= df[df.sentiment!=class1]

    resampled = resample(df_class1, replace=True, n_samples=len(df_class2.sentiment), random_state=27)

    df_resampled = pd.concat([resampled, df_new])    

    return df_resampled
# Create a resampled dataset from our clean dataset

df_resample = resampling(df_train, -1, 2)
plot2 = plt.figure(figsize=(15,5))

names = ['Pro','News','Neutral','Anti']

perc = df_resample['sentiment'].replace({-1: 'Anti',0:'Neutral',1:'Pro',2:'News'}).value_counts()

perc.name = ''

perc.plot(kind='pie', labels=names, autopct='%1.1f%%')

plt.title('Proportion of tweets in each class: Resampled dataset',fontsize = 16)

plt.figtext(0.12, 0.1, 'figure 2: Percentage of tweets that are classified as either Anti, Pro, Neutral and News (Resampled)',

            horizontalalignment='left',fontsize = 14,style='italic')

plt.legend(df_train['sentiment'].replace({-1: 'Anti: Does not believe in man-made climate change',

                                          0:'Neutral: Neither believes nor refutes man-made climate change',

                                          1:'Pro:Believe in man-made climate change',

                                          2:'News: Factual News about climate change'}).value_counts().index,

           bbox_to_anchor=(2.3,0.7), loc="right")



plt.show()
# Create a dataframe containing only tweets from class 'Pro'

df_pro = df_train.loc[df_train['sentiment'] ==1, ['message']]



# Create a dataframe containing only tweets from class 'Anti'

df_anti = df_train.loc[df_train['sentiment'] ==-1, ['message']]



# Create a dataframe containing only tweets from class 'Neutral'

df_neutral = df_train.loc[df_train['sentiment'] ==0, ['message']]



# Create a dataframe containing only tweets from class 'News'

df_news=df_train.loc[df_train['sentiment']==2, ['message']]
# Create masks for the wordclouds of each class

pro_mask = np.array(Image.open('../input/resources/pro.jpg'))

neut_mask = np.array(Image.open('../input/resources/neutral.jpg'))

anti_mask = np.array(Image.open('../input/resources/anti.jpg'))

news_mask = np.array(Image.open('../input/resources/news.jpg'))
p= (' '.join(df_pro['message']))



wordcloud = WordCloud(width = 1000, height = 500,max_words=100,background_color="white",colormap="gist_earth",

                      mask=pro_mask,contour_width=30,contour_color='white').generate(p)

plot3 = plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.figtext(.5,.9,'Common words in Pro class: "Believes in man-made climate change"\n',fontsize=14, ha='center')



plt.show()
n= (' '.join(df_neutral['message']))



wordcloud = WordCloud(width = 1000, height = 500,max_words=100,background_color="white",colormap="gist_earth",

                      mask=neut_mask,contour_width=30,contour_color='white').generate(n)

plot4 = plt.figure(figsize=(20,12))

plt.imshow(wordcloud)

plt.axis('off')

plt.figtext(.515,.9,'Common words in Neutral class: "Neither supports nor refutes belief of man-made climate change\n"',fontsize=14, ha='center')



plt.show()
a= (' '.join(df_anti['message']))



wordcloud = WordCloud(width = 1000, height = 500,max_words=100,background_color="white",colormap="gist_earth",

                      mask=anti_mask,contour_width=30,contour_color='white').generate(a)

plot5 = plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.figtext(.515,.9,'Common words in Anti class: "Does not believe in man-made climate change"\n',fontsize=14, ha='center')



plt.show()
nw= (' '.join(df_news['message']))



wordcloud = WordCloud(width = 1000, height = 500,max_words=100,background_color="white",colormap="gist_earth",

                      mask=news_mask,contour_width=30,contour_color='white').generate(nw)

plot6 = plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.figtext(.515,.9,'Common words in News class: "Factual news about climate change"\n',fontsize=14, ha='center')



plt.show()
# Creat an extra variable containing the number of words in a tweet

df_train_eda['count'] = df_train_eda['message'].apply(lambda x: len(x.split()))
# Plot the distribution of the number of words per class

plot7 = plt.figure(figsize=(12,6))

plt.title('Kernel distribution of number of words per class',fontsize = 14)

sns.kdeplot(df_train_eda['count'][df_train_eda['sentiment']==1], shade=True, color='g',legend=False)

sns.kdeplot(df_train_eda['count'][df_train_eda['sentiment']==0], shade=True, color='b',legend=False)

sns.kdeplot(df_train_eda['count'][df_train_eda['sentiment']==-1], shade=True, color='r',legend=False)

sns.kdeplot(df_train_eda['count'][df_train_eda['sentiment']==2], shade=True, color='orange',legend=False)

plt.xlabel('number of words per class', fontsize = 10)

plt.xlabel('Probability density function', fontsize = 10)

plt.legend(title='Sentiment class', loc='upper right', labels=['Pro', 'Neutral', 'Anti', 'News'])

plt.figtext(0.12, 0, 'figure 3: Probability distribution function for the number of words in each class',

            horizontalalignment='left',fontsize = 14,style='italic')



plt.show()
# Extract lists of urls and find the length of those lists

df_train_eda['url'] = df_train_eda['message'].apply(lambda x: len(re.findall(r'https\S+|www\S+',x)))

print('Number of urls per sentiment class')

df_train_eda.groupby('sentiment').sum()['url']
# Plot number of urls per class

plot8 = plt.figure(figsize=(10,5))

plt.title('Number of urls per class',fontsize = 16)

df_train_eda.groupby('sentiment').sum()['url'].plot(kind='bar')

plt.figtext(0.12, -0.1, 'figure 4 : Number of urls per class', horizontalalignment='left',fontsize = 14,style='italic')

plt.show()
# Extract lists of #hashtags and find the length of those lists

df_train_eda['hashtags'] = df_train_eda['message'].apply(lambda x: len(re.findall(r'[#]',x)))

print('Number of hashtags per sentiment class')

df_train_eda.groupby('sentiment').sum()['hashtags']
# Plot number of urls per class

plot9 = plt.figure(figsize=(10,5))

plt.title('Number of hashtags per class',fontsize = 16)

df_train_eda.groupby('sentiment').sum()['hashtags'].plot(kind='bar')

plt.figtext(0.12, 0, 'figure 5: Number of hashtags per class', horizontalalignment='left',fontsize = 14,style='italic')

plt.show()
sid = SentimentIntensityAnalyzer()



# Create a copy of our train dataset to add sentiment scores

df_sent = df_train_eda.copy()

df_sent['compound']  =  df_sent['message'].apply(lambda x: sid.polarity_scores(x)['compound'])

df_sent['comp_score'] = df_sent['compound'].apply(lambda c: 'pos' if c >0 else 'neg' if c<0 else 'neu')



# Extract all news and separate them from positive, neuteral and negative

df_news = df_sent[df_sent['sentiment']==2] 

df_analyse = df_sent[df_sent['sentiment'] != 2]

df_analyse.head()
# 5 most positive tweets

print('\033[1m 5 random tweets with the highest positive sentiment: \033[0m \n')

tweets = df_analyse.loc[df_analyse['compound'] > 0.8, ['message']].sample(5).values

for c in tweets:

    print(c[0])

    print('\n')
print('\033[1m 5 random tweets with the neutral sentiment: \033[0m \n')

tweets = df_analyse.loc[df_analyse['compound'] == 0, ['message']].sample(5).values

for c in tweets:

    print(c[0])

    print('\n')
print('\033[1m 5 random tweets with the highest negative sentiment: \033[0m \n')

tweets = df_analyse.loc[df_analyse['compound'] < -0.8, ['message']].sample(5).values

for c in tweets:

    print(c[0])

    print('\n')
print('\033[1m 5 random tweets from news: \033[0m \n')

tweets = df_news['message'].sample(5).values

for c in tweets:

    print(c)

    print('\n')
#proportion of the negative positive and neutral sentiments.

df_analyse['comp_score'].replace({1:'positive',0:'neutral',-1:'negative'}).value_counts()/len(df_analyse)
# Plot the overall distribution of sentiment scores

plot10, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 4))

plt.figtext(.51,.95, 'Overall distribution of sentiment scores\n', fontsize=20, ha='center',fontweight='bold')



ax1.hist(df_analyse['compound'], bins=15, edgecolor='k',color='lightblue')

plt.figtext(0.23, 0.06, 'sentiment score\n', horizontalalignment='left',fontsize = 12)

plot10.text(0.00001, 0.5, 'number of tweets', va='center', rotation='vertical',fontsize=12)

plt.figtext(0.10, 0.0001, 'figure 6: positive, negative and neutral sentiment', horizontalalignment='left',fontsize = 14,style='italic')



bins = np.linspace(-1, 1, 30)

ax2.hist([df_analyse['compound'][df_analyse['compound'] > 0], df_analyse['compound'][df_analyse['compound'] < 0]], bins, label=['Positive sentiment', 'Negative sentiment'])

plt.xlabel('sentiment score\n',fontsize=12)

ax2.legend(loc='upper right')

plt.figtext(0.90, 0.0001, 'figure 7: positive and negative sentiment', horizontalalignment='right',fontsize = 14,style='italic')



plt.tight_layout()

plt.show()
# Plot the distribution of sentiment scores per class

df_sent['sentiment'] = df_sent['sentiment'].replace({2:'News',1:'Pro',0:'Neutral',-1:'Anti'})

plot11 = plt.figure(figsize=(10,8))

sns.boxplot(x='sentiment', y='compound' , data= df_sent)

plt.title('Distribution of the sentiment scores per class',fontsize = 14)

plt.xlabel('sentiment score', fontsize = 10)

plt.figtext(0.12, 0.00000001, 'figure 8: Distribution of positive, negative and neutral sentiment scores', horizontalalignment='left',

            fontsize = 14,style='italic')



plt.show()
y = df_resample['sentiment']

X = df_resample['message']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Write class that has object that tokenizes text data AND stems the tokens

class StemAndTokenize:

    def __init__(self):

        self.ss = SnowballStemmer('english')

    def __call__(self, doc):

        return [self.ss.stem(t) for t in word_tokenize(doc)]
# Stopwords included

SW_vec_uni = TfidfVectorizer(tokenizer=StemAndTokenize())

SW_X_uni = SW_vec_uni.fit_transform(X_train)



# Stopwords excluded

noSW_vec_uni = TfidfVectorizer(stop_words='english', tokenizer=StemAndTokenize())

noSW_X_uni = noSW_vec_uni.fit_transform(X_train)
# Stopwords included

SW_uni = LogisticRegression()

SW_uni.fit(SW_X_uni,y_train)



# Stopwords excluded

noSW_uni = LogisticRegression()

noSW_uni.fit(noSW_X_uni,y_train)
# Stopwords included

SW_uni_pred = SW_uni.predict(SW_vec_uni.transform(X_test))

print('====== Stopwords included ======')

print(f'Accuracy: {accuracy_score(y_test, SW_uni_pred)} \nF1-score: {f1_score(y_test, SW_uni_pred, average="weighted")}')

print()

print()



# Stopwords excluded

noSW_uni_pred = noSW_uni.predict(noSW_vec_uni.transform(X_test))

print('====== Stopwords excluded ======')

print(f'Accuracy: {accuracy_score(y_test, noSW_uni_pred)} \nF1-score: {f1_score(y_test, noSW_uni_pred, average="weighted")}')
# Bigrams

SW_vec_bi = TfidfVectorizer(tokenizer=StemAndTokenize(), ngram_range=(2, 2))

SW_X_bi = SW_vec_bi.fit_transform(X_train)



# Trigrams

SW_vec_tri = TfidfVectorizer(tokenizer=StemAndTokenize(), ngram_range=(3, 3))

SW_X_tri = SW_vec_tri.fit_transform(X_train)
# Bigrams

SW_bi = LogisticRegression()

SW_bi.fit(SW_X_bi,y_train)



# Trigrams

SW_tri = LogisticRegression()

SW_tri.fit(SW_X_tri,y_train)
# Unigrams

SW_uni_pred = SW_uni.predict(SW_vec_uni.transform(X_test))

print('========== Unigrams ==========')

print(f'Accuracy: {accuracy_score(y_test, SW_uni_pred)} \nF1-score: {f1_score(y_test, SW_uni_pred, average="weighted")}')

print()

print()

      

# Bigrams

SW_bi_pred = SW_bi.predict(SW_vec_bi.transform(X_test))

print('========== Bigrams ==========')

print(f'Accuracy: {accuracy_score(y_test, SW_bi_pred)} \nF1-score: {f1_score(y_test, SW_bi_pred, average="weighted")}')

print()

print()



# Trigrams

SW_tri_pred = SW_tri.predict(SW_vec_tri.transform(X_test))

print('========== Trigrams ==========')

print(f'Accuracy: {accuracy_score(y_test, SW_tri_pred)} \nF1-score: {f1_score(y_test, SW_tri_pred, average="weighted")}')
# parameters to be tested for Logistic Regression

param_grid_lr = {'C':[0.01, 0.1, 1, 5, 10, 20, 50]}
# Create pipeline for Logistic Regression:

lr = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('lr', GridSearchCV(LogisticRegression(),

                                   param_grid=param_grid_lr,

                                   cv=10,

                                   n_jobs=-1,

                                   scoring='f1_weighted'))

              ])
# parameters to be tested for Naïve Bayes

param_grid_nb = {'alpha':[0.001, 0.01, 0.1, 1, 5]}
# Create pipeline for Naïve Bayes:

nb = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('nb', GridSearchCV(MultinomialNB(),

                                   param_grid=param_grid_nb,

                                   cv=10,

                                   n_jobs=-1,

                                   scoring='f1_weighted'))

              ])
# parameters to be tested for SVM

param_grid_svm = {'C':[10, 30, 50, 100],

                  'kernel':['linear', 'rbf'],

                  'gamma':['scale','auto']}
# Create pipeline for SVM:

svm = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('svm', GridSearchCV(SVC(random_state=42, probability = True),

                                   param_grid=param_grid_svm,

                                   cv=10,

                                   n_jobs=-1,

                                   scoring='f1_weighted'))

              ])
# parameters to be tested for Random Forest

param_grid_rf = {'max_features':[0.5,'log2'],

                 'n_estimators':[50,75,100],

                 'max_depth':[15,None]}
# Create pipeline for Random Forest:

rf = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('rf', GridSearchCV(RandomForestClassifier(random_state=42,criterion='entropy'),

                                   param_grid=param_grid_rf,

                                   cv=10,

                                   n_jobs=-1,

                                   scoring='f1_weighted'))

              ])
# parameters to be tested for KNN

param_grid_knn = {'n_neighbors':[3,10,30,50,100,150]}
# Create pipeline for KNN:

knn = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('knn', GridSearchCV(KNeighborsClassifier(weights='distance'),

                                    param_grid=param_grid_knn,

                                    cv=10,

                                    n_jobs=-1,

                                    scoring='f1_weighted'))

              ])
# parameters to be tested for Neural Networks

param_grid_nn = {'activation':['tanh','relu']}
# Create pipeline for Neural Networks:

nn = Pipeline([('tfidf', TfidfVectorizer(tokenizer=StemAndTokenize())),

               ('nn', GridSearchCV(MLPClassifier(batch_size=100,random_state=42,verbose=True,early_stopping=True,tol=0.005),

                                   param_grid=param_grid_nn,

                                   cv=10,

                                   n_jobs=-1,

                                   scoring='f1_weighted'))

              ])
# Fitting the Logistic Regression model

t0_lr = time()

lr.fit(X_train, y_train)

train_time_lr = time() - t0_lr
best_param_lr = lr['lr'].best_params_

print(f'The best parameters: {best_param_lr}')

print("Training time:  %0.3fs" % train_time_lr)
# Fitting the Naïve Bayes model

t0_nb = time()

nb.fit(X_train, y_train)

train_time_nb = time() - t0_nb
best_param_nb = nb['nb'].best_params_

print(f'The best parameters: {best_param_nb}')

print("Training time:  %0.3fs" % train_time_nb)
# Fitting the SVM model

t0_svm = time()

svm.fit(X_train, y_train)

train_time_svm = time() - t0_svm
best_param_svm = svm['svm'].best_params_

print(f'The best parameters: {best_param_svm}')

print("Training time:  %0.3fs" % train_time_svm)
# Fitting the Random Forest model

t0_rf = time()

rf.fit(X_train, y_train)

train_time_rf = time() - t0_rf
best_param_rf = rf['rf'].best_params_

print(f'The best parameters: {best_param_rf}')

print("Training time:  %0.3fs" % train_time_rf)
# Fitting the KNN model

t0_knn = time()

knn.fit(X_train, y_train)

train_time_knn = time() - t0_knn
best_param_knn = knn['knn'].best_params_

print(f'The best parameters: {best_param_knn}')

print("Training time:  %0.3fs" % train_time_knn)
# Fitting the Neural Networks model

t0_nn = time()

nn.fit(X_train, y_train)

train_time_nn = time() - t0_nn
best_param_nn = nn['nn'].best_params_

print(f'The best parameters: {best_param_nn}')

print("Training time:  %0.3fs" % train_time_nn)
# Form a prediction set for the Logistic Regression model

pred_lr = lr.predict(X_test)
# Form a prediction set for the Naïve Bayes model

pred_nb = nb.predict(X_test)
# Form a prediction set for the Linear SVM model

pred_svm = svm.predict(X_test)
# Form a prediction set for the Random Forest model

pred_rf = rf.predict(X_test)
# Form a prediction set for the KNN model

pred_knn = knn.predict(X_test)
# Form a prediction set for the Neural Network model

pred_nn = nn.predict(X_test)
# Function to plot ROC curves

def ROC_curve(model,y_test,model_name,ax):

    ''' Function that plots a Receiver operating characteristic (ROC) curve for multiclass data

       

        Parameters

        ----------

        model:      Trained model.

        y_test:     Series of label values for testing.

        model_name: Text string of the name of the model used.

        ax:         The axis to which the ROC curve is plotted.



        Returns

        -------

        ROC curve for multiclass data.

        

        - based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    

    '''

    

    # Binarize label

    y_lr = label_binarize(y_test, classes=[-1,0,1,2])

    

    # Calculate confidence scores per class combination

    y_score = model.predict_proba(X_test)

    

    FPR = dict()

    TPR = dict()

    roc_auc = dict()

    

    for i in range(4):

        FPR[i], TPR[i], _ = roc_curve(y_lr[:, i], y_score[:, i])

        roc_auc[i] = auc(FPR[i], TPR[i])

        

    lab = ['Anti','Neutral','Pro','News']

    colors = ['red', 'blue', 'green','orange']

    for i, color in zip(range(4), colors):

        ax.plot(FPR[i], TPR[i], color=color, lw=3,

                 label=f'ROC curve of the "{lab[i]}" class (area = {round(roc_auc[i],2)})')

        

    ax.plot([0, 1], [0, 1], 'k--', lw=3)

    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('False Positive Rate (FPR)')

    ax.set_ylabel('True Positive Rate (TPR)')

    ax.set_title(f'Receiver operating characteristic (ROC) curve: {model_name}\n')

    ax.legend(loc="lower right")

labels = ['2: News', '1: Pro', '0: Neutral', '-1: Anti']
pd.DataFrame(data=confusion_matrix(y_test, pred_lr), index=labels, columns=labels)
pd.DataFrame(data=confusion_matrix(y_test, pred_nb), index=labels, columns=labels)
pd.DataFrame(data=confusion_matrix(y_test, pred_svm), index=labels, columns=labels)
pd.DataFrame(data=confusion_matrix(y_test, pred_rf), index=labels, columns=labels)
pd.DataFrame(data=confusion_matrix(y_test, pred_knn), index=labels, columns=labels)
pd.DataFrame(data=confusion_matrix(y_test, pred_nn), index=labels, columns=labels)
# Plot ROC curves from different models

plot12, axs = plt.subplots(3, 2, figsize=(15,20))

ROC_curve(lr,y_test,'Logistic Regression',axs[0,0])

ROC_curve(nb,y_test,'Naïve Bayes',axs[1,0])

ROC_curve(svm,y_test,'SVM',axs[2,0])

ROC_curve(rf,y_test,'Random Forest',axs[0,1])

ROC_curve(knn,y_test,'KNN',axs[1,1])

ROC_curve(nn,y_test,'Neural Networks',axs[2,1])



plt.tight_layout()

plt.show()
print('\033[1m Classification Report from Logistic Regression Model \033[0m \n')

print(classification_report(y_test, pred_lr, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
print('\033[1m Classification Report from Naïve Model \033[0m \n')

print(classification_report(y_test, pred_nb, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
print('\033[1m Classification Report from SVM (Support Vector Machine) Model \033[0m \n')

print(classification_report(y_test, pred_svm, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
print('\033[1m Classification Report from Random Forest Model \033[0m \n')

print(classification_report(y_test, pred_rf, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
print('\033[1m Classification Report from KNN(K Nearest Neighbours) Model \033[0m \n')

print(classification_report(y_test, pred_knn, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
print('\033[1m Classification Report from Neural Networks Model \033[0m \n')

print(classification_report(y_test, pred_nn, target_names=['2: News', '1: Pro', '0: Neutral', '-1: Anti']))
F1_dict = {'Model':['Logistic Regression','Naïve Bayes','Linear SVM','Random Forest','KNN','Neural Network'],

           'F1 score' :[f1_score(y_test, pred_lr, average='weighted'),

                        f1_score(y_test, pred_nb, average='weighted'),

                        f1_score(y_test, pred_svm, average='weighted'),

                        f1_score(y_test, pred_rf, average='weighted'),

                        f1_score(y_test, pred_knn, average='weighted'),

                        f1_score(y_test, pred_nn, average='weighted')], 

           'Train time (sec)':[train_time_lr,

                               train_time_nb,

                               train_time_svm,

                               train_time_rf,

                               train_time_knn,

                               train_time_nn]}





F1_score = pd.DataFrame(data=F1_dict, columns=['Model','F1 score','Train time (sec)'])

F1_score.set_index('Model',inplace = True)

F1_score.sort_values("F1 score", ascending = False, inplace=True)

F1_score
plot13, ax = plt.subplots(1, 2, figsize=(10, 5))

F1_score.plot(y=['F1 score'], kind='bar', ax=ax[0], xlim=[0,1.1], ylim=[0.0,0.92])

F1_score.plot(y='Train time (sec)', kind='bar', ax=ax[1])



plt.tight_layout()

plt.show()
# Logistic regression

with open('Logistic_regression.pkl','wb') as file:

    pickle.dump(lr,file)

    

# Naïve Bayes

with open('Naive_bayes.pkl','wb') as file:

    pickle.dump(nb,file)

    

# SVM

with open('SVM.pkl','wb') as file:

    pickle.dump(svm,file)

    

# Random Forest

with open('Random_forest.pkl','wb') as file:

    pickle.dump(rf,file)

    

# KNN

with open('KNN.pkl','wb') as file:

    pickle.dump(knn,file)

    

# Neural Network

with open('Neural_network.pkl','wb') as file:

    pickle.dump(nn,file)
# Logistic regression

with open('best_param_dict_lr.pkl','wb') as file:

    pickle.dump(best_param_lr,file)

    

# Naïve Bayes

with open('best_param_dict_nb.pkl','wb') as file:

    pickle.dump(best_param_nb,file)

    

# SVM

with open('best_param_dict_svm.pkl','wb') as file:

    pickle.dump(best_param_svm,file)

    

# Random Forest

with open('best_param_dict_rf.pkl','wb') as file:

    pickle.dump(best_param_rf,file)

    

# KNN

with open('best_param_dict_knn.pkl','wb') as file:

    pickle.dump(best_param_knn,file)

    

# Neural Network

with open('best_param_dict_nn.pkl','wb') as file:

    pickle.dump(best_param_nn,file)
with open('prop_tweet.pkl','wb') as file:

    pickle.dump(plot1,file)

    

with open('prop_tweet_resample.pkl','wb') as file:

    pickle.dump(plot2,file)

    

with open('WC_pro.pkl','wb') as file:

    pickle.dump(plot3,file)

    

with open('WC_neut.pkl','wb') as file:

    pickle.dump(plot4,file)

    

with open('WC_anti.pkl','wb') as file:

    pickle.dump(plot5,file)

    

with open('WC_news.pkl','wb') as file:

    pickle.dump(plot6,file)

    

with open('dist_wordc.pkl','wb') as file:

    pickle.dump(plot7,file)

    

with open('urls_class.pkl','wb') as file:

    pickle.dump(plot8,file)

    

with open('hash_class.pkl','wb') as file:

    pickle.dump(plot9,file)



with open('oa_sent_dist.pkl','wb') as file:

    pickle.dump(plot10,file)



with open('sent_dist.pkl','wb') as file:

    pickle.dump(plot11,file)

    

with open('ROC.pkl','wb') as file:

    pickle.dump(plot12,file)

    

with open('acc_vs_time.pkl','wb') as file:

    pickle.dump(plot13,file)
# SVM

f1_svm = f1_score(y_test, pred_svm, average='weighted')

recall_svm = recall_score(y_test, pred_svm, average='weighted')

precision_svm = precision_score(y_test, pred_svm, average='weighted')

accuracy_svm = accuracy_score(y_test, pred_svm)

confusion_mat_svm = confusion_matrix(y_test, pred_svm)
# (FINAL MODEL after gridsearch best parms) -> SVM

paramssvm={"random_state":42,

           "test_size":0.2,

           "model_type":"SVM"

          }.update(best_param_svm)



metricssvm = {"f1":f1_svm,

              "recall":recall_svm,

              "precision":precision_svm,

              "accuracy":accuracy_svm}



experiment.log_parameters(paramssvm)

experiment.log_metrics(metricssvm)



experiment.log_confusion_matrix(labels=['2: News', '1: Pro', '0: Neutral', '-1: Anti'],matrix=confusion_mat_svm)



experiment.log_figure(figure=plot1,figure_name='Wordcloud of the most common words in the class "Pro"')

experiment.log_figure(figure=plot2,figure_name='Wordcloud of the most common words in the class "Neutral"')

experiment.log_figure(figure=plot3,figure_name='Wordcloud of the most common words in the class "Anti"')

experiment.log_figure(figure=plot4,figure_name='Wordcloud of the most common words in the class "News"')

experiment.log_figure(figure=plot5,figure_name='Overall distribution of sentiment scores')

experiment.log_figure(figure=plot6,figure_name='Distribution of the sentiment scores per class')

experiment.log_figure(figure=plot7,figure_name='Number of tweets per class')

experiment.log_figure(figure=plot8,figure_name='Proportion of tweets in each class')

experiment.log_figure(figure=plot9,figure_name='Proportion of tweets in each class: Resampled dataset')

experiment.log_figure(figure=plot10,figure_name='Kernel distribution of number of words per class')

experiment.log_figure(figure=plot11,figure_name='Receiver operating characteristic (ROC) curves for every model')

experiment.log_figure(figure=plot12,figure_name='Model accuracy vs. Training time')



experiment.log_model(name='SVM model', file_or_folder='SVM.pkl')
# Make predictions

predictions = svm.predict(df_test['message']).reshape(-1, 1)

Id = np.array(df_test.index).reshape(-1, 1)

names = ['tweetid', 'sentiment']



# Create output dataframe

out = pd.DataFrame(np.append(Id,predictions, axis=1), columns=names)
# Output to csv file

out.to_csv('output.csv',index=False)
experiment.end()
experiment.display()