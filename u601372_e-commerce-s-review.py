# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/Amazon_Instant_Video_5.json', lines=True)
Apps_android_df = pd.read_json('../input/Apps_for_Android_5.json', lines=True)
Auto_df = pd.read_json('../input/Automotive_5.json', lines=True)
baby_df = pd.read_json('../input/Baby_5.json', lines=True)
Beauty_df = pd.read_json('../input/Beauty_5.json', lines=True)
CDs_df = pd.read_json('../input/CDs_and_Vinyl_5.json', lines=True)
CellPhone_df = pd.read_json('../input/Cell_Phones_and_Accessories_5.json', lines=True)
Clothing_df = pd.read_json('../input/Clothing_Shoes_and_Jewelry_5.json', lines=True)
Digital_df = pd.read_json('../input/Digital_Music_5.json', lines=True)
Elect_df = pd.read_json('../input/Electronics_5.json', lines=True)
Grocery_df = pd.read_json('../input/Grocery_and_Gourmet_Food_5.json', lines=True)
Health_df = pd.read_json('../input/Health_and_Personal_Care_5.json', lines=True)
Home_df = pd.read_json('../input/Home_and_Kitchen_5.json', lines=True)
Kindle_df = pd.read_json('../input/Kindle_Store_5.json', lines=True)
Musical_df = pd.read_json('../input/Musical_Instruments_5.json', lines=True)
Office_df = pd.read_json('../input/Office_Products_5.json', lines=True)
Patio_df = pd.read_json('../input/Patio_Lawn_and_Garden_5.json', lines=True)
Pet_df = pd.read_json('../input/Pet_Supplies_5.json', lines=True)
Sports_df = pd.read_json('../input/Sports_and_Outdoors_5.json', lines=True)
Tools_df = pd.read_json('../input/Tools_and_Home_Improvement_5.json', lines=True)
Toys_df = pd.read_json('../input/Toys_and_Games_5.json', lines=True)
Videos_df = pd.read_json('../input/Video_Games_5.json', lines=True)
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

features_positive = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in negative_fileids]

threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))

features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
print ("\nNumber of training :", len(features_train))
print ("Number of testing:", len(features_test))

classifier = NaiveBayesClassifier.train(features_train)
print ("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))
newdf = df[df.columns[3]]
newdf.head()
input_reviews = [
        newdf.iloc[10],
        newdf.iloc[356]
   ]

print ("\nPredictions:")
for review in input_reviews:
    print ("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print ("Predicted sentiment:", pred_sentiment) 
    print ("Probability:", round(probdist.prob(pred_sentiment), 2))
def sentence_Scoring(review):
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    if pred_sentiment == "Negative":
        return -1
    else:
        return 1
df['Polarity'] = df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Apps_android_df['Polarity'] = Apps_android_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Auto_df['Polarity'] = Auto_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
baby_df['Polarity'] = baby_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Beauty_df['Polarity'] = Beauty_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
CDs_df['Polarity'] = CDs_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
CellPhone_df['Polarity'] = CellPhone_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Clothing_df['Polarity'] = Clothing_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Digital_df['Polarity'] = Digital_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Elect_df['Polarity'] = Elect_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Grocery_df['Polarity'] = Grocery_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Health_df['Polarity'] = Health_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Home_df['Polarity'] = Home_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Kindle_df['Polarity'] = Kindle_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Musical_df['Polarity'] = Musical_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Office_df['Polarity'] = Office_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Patio_df['Polarity'] = Patio_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Pet_df['Polarity'] = Pet_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Sports_df['Polarity'] = Sports_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Tools_df['Polarity'] = Tools_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Toys_df['Polarity'] = Toys_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
Videos_df['Polarity'] = Videos_df.apply(lambda row : sentence_Scoring(row['reviewText']),axis = 1)
def trustCaculate(df):
    Sentiment_score = df.groupby('Polarity').size()
    negative_score = Sentiment_score.loc[-1]
    positive_score = Sentiment_score.loc[1]
    maximum_rating = df['overall'].max()
    trust = (positive_score/(positive_score+negative_score))*maximum_rating
    return trust
trust_list = []
trust_list.append(trustCaculate(df))
trust_list.append(trustCaculate(Apps_android_df))
trust_list.append(trustCaculate(Auto_df)) 
trust_list.append(trustCaculate(baby_df)) 
trust_list.append(trustCaculate(Beauty_df)) 
trust_list.append(trustCaculate(CDs_df)) 
trust_list.append(trustCaculate(CellPhone_df)) 
trust_list.append(trustCaculate(Clothing_df))
trust_list.append(trustCaculate(Digital_df))
trust_list.append(trustCaculate(Elect_df)) 
trust_list.append(trustCaculate(Grocery_df)) 
trust_list.append(trustCaculate(Health_df))
trust_list.append(trustCaculate(Home_df))
trust_list.append(trustCaculate(Kindle_df)) 
trust_list.append(trustCaculate(Musical_df)) 
trust_list.append(trustCaculate(Office_df))
trust_list.append(trustCaculate(Patio_df))
trust_list.append(trustCaculate(Pet_df)) 
trust_list.append(trustCaculate(Sports_df)) 
trust_list.append(trustCaculate(Tools_df))
trust_list.append(trustCaculate(Toys_df)) 
trust_list.append(trustCaculate(Videos_df)) 
Categories_list = ['Amazon Instant Video','Apps for Android','Automotive','Baby','Beauty','CDs and Vinyl','Cell Phones and Accessories','Clothing, Shoes and Jewelry','Digital Music','Electronics','Grocery and Gourmet Food','Health and Personal Care','Home and Kitchen','Kindle Store','Musical Instruments','Office Products','Patio, Lawn and Garden','Pet Supplies','Sports and Outdoors','Tools and Home Improvement','Toys and Games','Video Games']
Column = ['Categories','Trust value (max: 5)']
trust_dict = {'Categories': Categories_list,'Trust value (max: 5)':trust_list}
Complete_df = pd.DataFrame(trust_dict, columns=Column)
Complete_df
import matplotlib.pyplot as plt
ax = plt.subplot()
ax.set_ylabel('Trust value')
ax.set_xlabel('Categories')
Complete_df.index = Categories_list
Complete_df.plot(kind='bar',figsize=(18,7), ax = ax, title = "Trust score for each product category of amazon")
Complete_df['Trust value (max: 5)'].mean()
