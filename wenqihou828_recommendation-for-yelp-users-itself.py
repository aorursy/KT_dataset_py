import json

import pandas as pd
frames_tip = []

for chunk in pd.read_json('../input/yelp_academic_dataset_tip.json', lines=True, chunksize = 10000):

    frames_tip.append(chunk)

tip=pd.concat(frames_tip)
tip.columns
tip.head()
frames_checkin = []

for chunk in pd.read_json('../input/yelp_academic_dataset_checkin.json', lines=True, chunksize = 10000):

    frames_checkin.append(chunk)

checkin=pd.concat(frames_checkin)
checkin.columns
checkin.shape
checkin.head()
frames_review = []

for chunk in pd.read_json('../input/yelp_academic_dataset_review.json', lines=True, chunksize = 20000):

    frames_review.append(chunk)

review=pd.concat(frames_review)
review.shape
frames = []

for chunk in pd.read_json('../input/yelp_academic_dataset_user.json', lines=True, chunksize = 10000):

    frames.append(chunk)

user = pd.concat(frames)
user.shape
user.head()
frames_business = []

for chunk in pd.read_json('../input/yelp_academic_dataset_business.json', lines=True, chunksize = 10000):

    frames_business.append(chunk)

business = pd.concat(frames_business)

business.head()
business['city'].value_counts().head()
business_vegas=business[business['city']=='Las Vegas']
business_vegas=business_vegas.reset_index(drop=True)
import re

business_vegas['restaurant']=business_vegas['categories'].str.contains('Restaurants',flags=re.IGNORECASE)
business_vegas_restaurant=business_vegas[business_vegas['restaurant']==True]
business_vegas_restaurant.head()
business_vegas_restaurant.reset_index(drop=True).head()
business_vegas_restaurant.shape
business_vegas_restaurant.to_pickle('restaurant in vegas.pickle')

review=review.drop('text',axis=1)
review_in_vegas=review.loc[review['business_id'].isin(business_vegas_restaurant['business_id'].unique())]
review_in_vegas.reset_index(drop=True).head()
review_in_vegas.to_pickle('vegas_review.pickle')
user.columns
user_in_vegas=user.loc[user['user_id'].isin(review_in_vegas['user_id'].unique())]
user_in_vegas.to_pickle('vegas_users.pickle')
tip.columns
tip_in_vegas=tip.loc[tip['user_id'].isin(review_in_vegas['user_id'].unique())].reset_index(drop=True)
tip_in_vegas.to_pickle('tip_in_vegas.pickle')
check_in_vegas=checkin.loc[checkin['business_id'].isin(business_vegas_restaurant['business_id'].unique())].reset_index(drop=True)
check_in_vegas.to_pickle('checkin_vegas.pickle')
import pandas as pd

rest = pd.read_pickle('restaurant in vegas.pickle')
import numpy as np

rest.fillna(value=pd.np.nan, inplace=True)
Rest = rest.reset_index(drop=True)

Rest.index +=1

Rest.head()
Rest.columns
Rest_final = Rest[['name', 'business_id', 'address', 'categories', 'postal_code','attributes','hours','latitude','longitude','review_count','stars']]
categories=', '.join(list(Rest_final['categories'].unique()))

categories=categories.split(', ')

categories[:5]
from collections import Counter, defaultdict

c = Counter(categories)

c.most_common(60)
cuisine = 'American|Chinese|Italian|Japanese|Mexican|Asian Fusion|Thai|Korean|Mediterranean'

Rest_final['cuisine']=Rest_final['categories'].str.findall(cuisine)
Rest_final['cuisine']=Rest_final['cuisine'].map(lambda x: list(x))

Rest_final['cuisine']=Rest_final['cuisine'].map(lambda x: ['Others'] if x==[] else x)
Rest_final['cuisine'].head(20)
Rest_final['cuisine']=Rest_final['cuisine'].map(lambda x: list(dict.fromkeys(x)))

Rest_final['cuisine']=Rest_final['cuisine'].map(', '.join) # convert list of string to string

Rest_final['cuisine'].head(20)
Rest_final['cuisine'].unique()
Rest_final['cuisine'].iloc[np.where(Rest_final['cuisine'].str.contains('Asian Fusion'))]='Asian Fusion'
Rest_final['cuisine'].unique()
Rest_final.isnull().sum()
Rest_final['attributes'].apply(pd.Series).head()

# Split the attributes dictionary into all its values
R = Rest_final['attributes'].apply(pd.Series)

list(R.columns)
Rest_new = pd.concat([Rest_final.drop(['attributes'], axis=1), Rest_final['attributes'].apply(pd.Series)], axis=1)

Rest_new.head()
Rest_new = Rest_new[['name', 'business_id', 'address', 'cuisine', 'postal_code','hours','latitude','longitude',

                   'review_count','stars','OutdoorSeating','BusinessAcceptsCreditCards','RestaurantsDelivery',

                   'RestaurantsReservations','WiFi','Alcohol','categories']]
Rest_new.fillna(value=pd.np.nan, inplace=True)

Rest_new['WiFi'].unique()
a=Rest_new['WiFi'].map(lambda x: 'No' if x in np.array(["u'no'", "'no'",'None']) else x)

a=a.map(lambda x: 'Free' if x in np.array(["'free'", "u'free'"]) else x)

a.unique()
a=a.map(lambda x: 'Paid' if x in np.array(["'paid'", "u'paid'"]) else x)

a.unique()
Rest_new['WiFi']=a
Rest_new['Alcohol'].unique()
Alc = Rest_new['Alcohol'].map(lambda x: 'Full_Bar' if x in np.array(["u'full_bar'", "'full_bar'"]) else x)

Alc.unique()
Alc = Alc.map(lambda x: 'Beer&Wine' if x in np.array(["u'beer_and_wine'", "'beer_and_wine'"]) else x)

Alc.unique()
Alc = Alc.map(lambda x: 'No' if x in np.array(["u'none'", "'none'",'None']) else x)

Alc.unique()
Rest_new['Alcohol']= Alc

Rest_new.head()
print(Rest_new['hours'][Rest_new['hours'].notnull()].map(lambda x: x.values()).map(len).sort_values().value_counts())
def merge(x,y):

    result = []

    try:

        for i in x:

            index = x.index(i)

            result.append(i)

            result.append(y[index])

        return result

    except TypeError:

        result = [np.NaN, np.NaN]
Rest_new['business_days']=Rest_new['hours'][Rest_new['hours'].notnull()].map(lambda x:list(x.keys()))

Rest_new['business_hours']=Rest_new['hours'][Rest_new['hours'].notnull()].map(lambda x:list(x.values()))

Rest_new['hours_day'] = Rest_new.apply(lambda row: merge(row['business_days'], row['business_hours']), axis=1)
Rest_new_hours = Rest_new[:]

Rest_new_hours.head(10)
Rest_new_hours['hours_day'][Rest_new_hours['hours_day'].notnull()]=Rest_new_hours['hours_day'][Rest_new['hours_day'].notnull()].map(lambda x: ''.join(x))

Rest_new_hours.head()
Rest_new_hours['Monday_Open']=Rest_new_hours['hours_day'].str.extract('[M][o][n][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Tuesday_Open']=Rest_new_hours['hours_day'].str.extract('[T][u][e][s][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Wednesday_Open']=Rest_new_hours['hours_day'].str.extract('[W][e][d][n][e][s][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Thursday_Open']=Rest_new_hours['hours_day'].str.extract('[T][h][u][r][s][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Friday_Open']=Rest_new_hours['hours_day'].str.extract('[F][r][i][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Saturday_Open']=Rest_new_hours['hours_day'].str.extract('[S][a][t][u][r][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Sunday_Open']=Rest_new_hours['hours_day'].str.extract('[S][u][n][d][a][y](\d*[:]\d*)[-]\d*[:]\d*')

Rest_new_hours['Monday_Close']=Rest_new_hours['hours_day'].str.extract('[M][o][n][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Tuesday_Close']=Rest_new_hours['hours_day'].str.extract('[T][u][e][s][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Wednesday_Close']=Rest_new_hours['hours_day'].str.extract('[[W][e][d][n][e][s][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Thursday_Close']=Rest_new_hours['hours_day'].str.extract('[T][h][u][r][s][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Friday_Close']=Rest_new_hours['hours_day'].str.extract('[F][r][i][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Saturday_Close']=Rest_new_hours['hours_day'].str.extract('[S][a][t][u][r][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')

Rest_new_hours['Sunday_Close']=Rest_new_hours['hours_day'].str.extract('[S][u][n][d][a][y]\d*[:]\d*[-](\d*[:]\d*)')
Rest_new_hours.head(5)
Rest_new_hours.drop(['hours_day','business_days','business_hours'],axis=1,inplace=True)

Rest_new_hours.columns
def str2time(val):

    try:

        return dt.datetime.strptime(val, '%H:%M').time()

    except:

        return pd.NaT
import datetime as dt

Rest_new_hours.iloc[:,17:31]=Rest_new_hours.iloc[:,17:31].astype(str)

Rest_new_hours.iloc[:,17:31]=Rest_new_hours.iloc[:,17:31].applymap(lambda x: str2time(x))

Rest_new_hours.iloc[:,17:31].head()
Rest_new_hours.loc[3801]
Rest_new_hours.drop('hours',axis=1,inplace=True)

Rest_new_hours.head()
import pickle

import pandas as pd

import numpy as np
pickle_review = open("vegas_review.pickle","rb")

review = pickle.load(pickle_review)

review.head()
Review = review.reset_index(drop=True)

Review.index +=1

Review.head()
Review = Review[['business_id', 'user_id', 'review_id', 'date', 'cool','funny','useful','stars']]

Review.head()
pickle_users = open("vegas_users.pickle","rb")

users = pickle.load(pickle_users)
#dropping org index 

users = users.reset_index(drop=True)

users.index +=1
titles = ['user_id','name','average_stars','yelping_since','review_count','elite','fans','useful','cool','funny','friends']

users =users.reindex(columns=titles)



#rename columns

users = users.rename(columns={'name':'user_name','review_count':'review'})   
#converting timestamp to date 

users['yelping_since'] = pd.to_datetime(users['yelping_since'])

users['yelping_since'] = users['yelping_since'].dt.date
import re

users['elite'] = users['elite'].apply(lambda x: re.findall('20\d\d',x))
users['elite'] = users['elite'].apply(lambda x: len(x))
users['friends'].str.split(',')

users['friends'] = users['friends'].apply(lambda x: len(x))
users = users.rename(columns={'elite':'years_of_elite'})

users.head()
pickle_tip = open("tip_in_vegas.pickle","rb")

tip = pickle.load(pickle_tip)

tip = tip.set_index(keys='business_id')
#load in restaurant pickle file in order to get the restaurant names

pickle_restaurant = open("restaurant in vegas.pickle","rb")

restaurant = pickle.load(pickle_restaurant)

restaurant_new = restaurant[['name','business_id']]

restaurant_new = restaurant_new.set_index(keys='business_id')
tip_new = tip.join(restaurant_new,how='inner')
tip_new['date'] = pd.to_datetime(tip_new['date'])

tip_new['date'] = tip_new['date'].dt.date
titles = ['name','date','text','user_id']

tip_new = tip_new.reindex(columns=titles)
tip_new = tip_new.rename(columns={'name':'restaurant_name','text':'user_tips','date':'tips_date'})
tip_new.head()
tip1 = tip_new.reset_index()

tip1.head()
import requests, re

import pandas as pd

import seaborn as sns

import nltk
import string, itertools

from collections import Counter, defaultdict

from nltk.text import Text

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer, WordNetLemmatizer

from wordcloud import WordCloud
# Select user_tips for reviewing the sentiment

pid = 10

report = 'user_tips'

print(tip1[['restaurant_name','tips_date']].loc[pid])

s = tip1.at[pid, report]

s
report = 'user_tips'

s = tip1[report]

s.head()
tip1['word_count'] = tip1['user_tips'].apply(lambda x: len(str(x).split(" ")))

tip2 = tip1[['business_id','restaurant_name','user_tips','word_count']]
tip2['char_count'] = tip2['user_tips'].str.len() ## this also includes spaces

tip2[['business_id','restaurant_name','user_tips','word_count','char_count']].head()
stop = stopwords.words('english')



tip2['stopwords'] = tip2['user_tips'].apply(lambda x: len([x for x in x.split() if x in stop]))

tip2[['user_tips','word_count','char_count','stopwords']].head()
tip2['user_tips'] = tip2['user_tips'].apply(lambda x: " ".join(x.lower() for x in x.split()))

tip2['user_tips'].head()
tip2['user_tips'] = tip2['user_tips'].str.replace('[^\w\s]','')

tip2['user_tips'].head()
tip2['user_tips'] = tip2['user_tips'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

tip2['user_tips'].head()
from textblob import TextBlob

tip2['user_tips'][:500].apply(lambda x: str(TextBlob(x).correct())).head(10)
frequency = pd.Series(' '.join(tip2['user_tips']).split()).value_counts()[-100:]

frequency.head(20)
frequency = list(frequency.index)

tip2['user_tips'] = tip2['user_tips'].apply(lambda x: " ".join(x for x in x.split() if x not in frequency))

tip3 = tip2.copy()

tip3['user_tips'].head()
TextBlob(tip3['user_tips'][10]).words
from textblob import Word

tip3['user_tips'] = tip3['user_tips'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

tip3['user_tips'].head()
tip3['user_tips'][:5].apply(lambda x: TextBlob(x).sentiment)
tip3.head()
tip3['sentiment'] = tip3['user_tips'].apply(lambda x: TextBlob(x).sentiment[0] )

tip3[['business_id','restaurant_name','user_tips','sentiment']].head()
tip3.head()
sentiment=tip3.groupby(by='business_id')['sentiment'].mean().sort_values(ascending = False)

sentiment.head()
Rest_new_hours.set_index('business_id',inplace=True)

Rest_new_hours['sentiment']=sentiment
Rest_new_hours.head()
Review['review_year'] = Review['date'].dt.year

##group by business_id and stars

new = Review.groupby(['business_id','review_year']).mean()

new.head()
year_stars=new['stars'].unstack()
df=pd.concat([Rest_new_hours,year_stars],axis=1)

df = df.set_index(keys=['name','address']).sort_index(level=[0,1])

print(df.columns)

df.head()
%pylab inline

import pandas as pd

import seaborn as sns
plt.figure(figsize=(25,10))

cuisines=df[df['cuisine']!='Others']

ax=sns.pointplot(x='cuisine', y='sentiment',join=True,ci=None, estimator=mean,data=cuisines)

plt.legend(['Sentiment Score'],prop={'size' : 20},bbox_to_anchor=(0.2,0.2));

ax2 = ax.twinx()

sns.pointplot(x='cuisine', y='stars',join=True,ci=None, estimator=mean,data=cuisines,color='red',ax=ax2)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90);

plt.legend(['Average Stars'],bbox_to_anchor=(0.18,0.3),prop={'size': 20});

def findmyresturant(WiFi=None,OutdoorSeating=None,RestaurantsDelivery=None,BusinessAcceptsCreditCards=None,RestaurantsReservations=None,Alcohol=None,cuisine=None):

    if WiFi!=None:

        df_1 = df[df['WiFi'].notnull()]

        df_2 = df_1[df_1['WiFi']!=False]

        df_wifi = df_2[df_2['WiFi']!='No']

    else:

        df_wifi = df

    if OutdoorSeating != None:

        df_3 = df_wifi[df_wifi['OutdoorSeating'].notnull()]

        df_outdoor = df_3[df_3['OutdoorSeating']!=False]

    else:

        df_outdoor = df_wifi

    if RestaurantsDelivery != None:

        df_4 = df_outdoor[df_outdoor['RestaurantsDelivery'].notnull()]

        df_delivery = df_4[df_4['RestaurantsDelivery']!=False]

    else:

        df_delivery = df_outdoor

    if RestaurantsReservations != None:

        df_5 = df_delivery[df_delivery['RestaurantsReservations'].notnull()]

        df_reserve = df_5[df_5['RestaurantsReservations']!=False]

    else:

        df_reserve = df_delivery

    if Alcohol != None:

        df_6 = df_delivery[df_delivery['Alcohol'].notnull()]

        df_alcohol = df_6[df_6['Alcohol']!='No']

    else: 

        df_alcohol=df_reserve

    if BusinessAcceptsCreditCards !=None:

        df_cards = df_alcohol[df_alcohol['BusinessAcceptsCreditCards']==True]

    else:

        df_cards = df_alcohol

    if cuisine != None:

        df_cuisine = df_cards[df_cards['cuisine'].str.contains(cuisine)]

    else:

        df_cuisine = df_cards

        

    df_cuisine=df_cuisine.sort_values(['stars','review_count'],ascending=False)

    new_df= df_cuisine[['cuisine','stars','postal_code','review_count','OutdoorSeating','BusinessAcceptsCreditCards',

 'RestaurantsDelivery','RestaurantsReservations','WiFi','Alcohol',2016,2017,2018]]

    %pylab inline

    import pandas as pd

    import seaborn as sns

    plot=new_df.reset_index().iloc[:5]

    g = sns.FacetGrid(plot, row='name', sharex=True, sharey=True, height=3,aspect=1.5)

    g=g.map(plt.scatter, 2018, 'review_count',marker='s');

    g=g.map(plt.scatter,'stars','review_count',color='red');

    plt.legend(('2018 Stars','Average Stars'),bbox_to_anchor=(1,5))

    return new_df.head(10)
findmyresturant(WiFi='yes',Alcohol='yes',cuisine='Chinese')
findmyresturant(WiFi='yes',OutdoorSeating='yes',cuisine='American',RestaurantsDelivery='yes')
df['review_count']=df['review_count'].fillna(0.0)

df_close_hour = df[df['Friday_Close'].notnull()].reset_index()

df_close_hour['Friday_Close'] = df_close_hour['Friday_Close'].map(lambda t: dt.datetime(year=2018, month=12, day=30, hour=t.hour, minute=t.minute))

df_close_hour.set_index('Friday_Close',inplace=True)
df_close_hour['review_count'].resample('180Min').mean().plot(figsize=(10,5));

plt.xlabel ('Friday Close Time');

plt.ylabel ('Reivew Count');
df_close_hour['sentiment'].resample('180Min').mean().plot(figsize=(10,5));

plt.xlabel ('Friday Close Time');

plt.ylabel ('Sentiment Score');
df[df['Saturday_Close']==datetime.time(15, 30)][['cuisine','stars','categories']]
users13 = users.loc[users['years_of_elite']==13]

users_elite = users13.sort_values('review',ascending=False).iloc[0:30]

users_elite.set_index('user_id',inplace=True)
tip_new1 = tip_new.set_index('user_id')

users_elite1 = users_elite.join(tip_new1,how='inner',on='user_id')
elite_top5 = users_elite1['restaurant_name'].value_counts().iloc[0:10]

elite_top5 = pd.DataFrame(elite_top5)

elite_top5.rename(columns={'restaurant_name':'common_review'},inplace=True)
plt.figure(figsize(15,8));

elite_top5.plot(kind='bar',color='g',legend=False);

plt.title('Most Reviewed by Elite Members');

plt.xlabel('Restaurant');

plt.ylabel('Number of Common Reviews');
tip_new2 = tip_new.set_index('user_id')

user_tip = users.join(tip_new2, how='inner',on='user_id')
Review['review_year'] = Review['date'].dt.year
new = pd.merge(Rest_new_hours, Review, on='business_id', how='outer')
new1 =new[['name','review_year','stars_y']]

new1 = new1.groupby(['name','review_year']).mean().reset_index()
##def a function for checking rate trending for a restaurant 



def rate_trend(name):       

    a =new1.loc[new1['name']==name]

    plt.figure(figsize=(10,8))

    sns.pointplot(x=a['review_year'],y=a['stars_y'],data=a,join=True,color='m')

    plt.yticks(np.arange(0, 6, step=1))

    plt.ylabel('Average Stars')  

    plt.xlabel('Year')

    plt.title('Rating Trend: '+str(name) )

    
rate_trend('Kabuto')
chosen_words = ['awesome', 'service', 'cheap', 'expensive', 'price', 'yummy', 'delicious', 'again', 'great',

                'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'place', 'strip', 'casino', 'ambience',

                'night', 'open', 'bar', 'nice', 'friendly', 'hostile', 'excellent','awful', 'wow', 'hate','staff']

print(chosen_words)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(vocabulary=chosen_words, lowercase=False)



selected_word_count = vectorizer.fit_transform(tip1['user_tips'].values)

print(vectorizer.get_feature_names())
word_count_array = selected_word_count.toarray()

word_count_array.shape
word_count_array

word_count_array.sum(axis=0)
Yelp_words = pd.DataFrame(index=vectorizer.get_feature_names(),

                    data=word_count_array.sum(axis=0)).rename(columns={0: 'Value Count'})
Yelp_words.plot(kind='bar', stacked=False, figsize=[10,8], colormap='Greens_r');
cloud = WordCloud(width=1200, height= 1080,max_words= 1000).generate(' '.join(tip1['user_tips'].astype(str)))

plt.figure(figsize=(15, 25))

plt.imshow(cloud)

plt.axis('off');