import os
import pandas as pd
import numpy as np
import re
#NLTK
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

#Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#Spell Correction
!pip install autocorrect
from autocorrect import Speller

#Tokenization
!pip install wordninja
import wordninja

#Necessary Libraries for plotting charts
import matplotlib.pyplot as plt
#plotly
!pip install plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

!pip install contractions
# contractions is a library for converting words like "I'm" to "I am"
import contractions

# Necessary Libraries to find similarity
import gensim 

import pandas as pd
df=pd.read_csv('../input/preprocessed-review-file/Review_text_preprocessed_file.csv')
df.head()
#Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#Necessary Libraries for plotting charts
import matplotlib.pyplot as plt
#plotly
!pip install plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
# replace null values with empty string
def remove_null(text):
    text = ''
    return text

#  repalcing null values in pros and cons columns
for i in df[df['Pros_Modified_Text'].isnull()]['Pros_Modified_Text'].index:
    df.loc[i,'Pros_Modified_Text'] = remove_null(df.loc[i,'Pros_Modified_Text'])

for i in df[df['Cons_Modified_Text'].isnull()]['Cons_Modified_Text'].index:
    df.loc[i,'Cons_Modified_Text'] = remove_null(df.loc[i,'Cons_Modified_Text'])
##### Function to calculate trigrams from a corpus using count vectorizer
def get_trigrams(corpus):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq
##### Function to calculate bigrams from a corpus using count vectorizer
def get_bigrams(corpus):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq
def bar_plot_top_n_trigrams(CompanyName,ReviewType,n):
    common_words = get_trigrams(df[df['CompanyName']==CompanyName][ReviewType])[:n]
    df1 = pd.DataFrame(common_words, columns = ['word', 'count'])
    fig = px.bar(df1, x='word', y='count')
    fig.update_layout(title_text='Top 20 trigrams by count in '+ReviewType[0:4]+' Review for '+CompanyName, template="plotly_white")
    return fig
bar_plot_top_n_trigrams('Amazon','Pros_Modified_Text',20)
bar_plot_top_n_trigrams('Amazon','Cons_Modified_Text',20)
def bar_plot_top_n_bigrams(CompanyName,ReviewType,n):
    common_words = get_bigrams(df[df['CompanyName']==CompanyName][ReviewType])[:n]
    df1 = pd.DataFrame(common_words, columns = ['word', 'count'])
    fig = px.bar(df1, x='word', y='count')
    fig.update_layout(title_text='Top 20 bigrams by count in '+ReviewType[0:4]+' Review for '+CompanyName, template="plotly_white")
    return fig
bar_plot_top_n_bigrams('Amazon','Pros_Modified_Text',20)
bar_plot_top_n_bigrams('Amazon','Cons_Modified_Text',20)
####Creating a dataframe of bigrams by company and Review Type
def bigramdataframe(df,companyname,reviewtype):
    data=get_bigrams(df[df['CompanyName']==companyname][reviewtype])
    temp = pd.DataFrame(data, columns =['Bigram', 'Frequency']) 
    temp['CompanyName']=companyname
    temp['ReviewType']=reviewtype[0:4]
    return temp
#Creating a dataframe of bigrams with frequency for all the companies and by review type
company_list=list(df['CompanyName'].unique())
Review_Type=['Pros_Modified_Text','Cons_Modified_Text']
bigrams= pd.DataFrame(columns =['Bigram', 'Frequency','CompanyName','ReviewType'])
for i in company_list:
    for j in Review_Type:
        bigrams=pd.concat([bigrams,bigramdataframe(df,i,j)])    
bigrams.head()
####Creating a dataframe of trigrams by company and Review Type
def trigramdataframe(df,companyname,reviewtype):
    data=get_trigrams(df[df['CompanyName']==companyname][reviewtype])
    temp = pd.DataFrame(data, columns =['Trigram', 'Frequency']) 
    temp['CompanyName']=companyname
    temp['ReviewType']=reviewtype[0:4]
    return temp
trigrams= pd.DataFrame(columns =['Trigram', 'Frequency','CompanyName','ReviewType'])
for i in company_list:
    for j in Review_Type:
        trigrams=pd.concat([trigrams,trigramdataframe(df,i,j)])  
trigrams.head()
#Function to calculate overall average rating based on keyword i.e.,bigram or trigram
import re
def avgsentiscorebasedonkeyword(Companyname,keyword,ReviewColumnName):
    avg=df[(df[ReviewColumnName].str.contains(keyword,flags=re.IGNORECASE, regex=True)) & (df['CompanyName']==Companyname)]['Review_polarity'].mean()
    return avg
def avg_rating_column(df,ngramtype):
    ratings_list=[]
    for index, row in df.iterrows():
        Companyname=row['CompanyName']
        keyword=row[ngramtype]
        if str(row['ReviewType'])=='Pros':
            ReviewColumnName='Pros_Modified_Text'
        elif str(row['ReviewType'])=='Cons':
            ReviewColumnName='Cons_Modified_Text'
        else:
            pass
        ratings_list.append(avgsentiscorebasedonkeyword(Companyname,keyword,ReviewColumnName))
    return(ratings_list)
#Group dataframe by company name,review type and take top 100 rows based on frequency in each category
bigrams = bigrams.groupby(["CompanyName","ReviewType"]).apply(lambda x: x.sort_values(["Frequency"], ascending = False)).reset_index(drop=True)
trigrams= trigrams.groupby(["CompanyName","ReviewType"]).apply(lambda x: x.sort_values(["Frequency"], ascending = False)).reset_index(drop=True)
df['Pros_Modified_Text'] = df['Pros_Modified_Text'].map(str)
df['Cons_Modified_Text'] = df['Cons_Modified_Text'].map(str)
# select top N rows within each category
bigrams_top100=bigrams.groupby(["CompanyName","ReviewType"]).head(100)
trigrams_top100=trigrams.groupby(["CompanyName","ReviewType"]).head(100)
#Reindexing
bigrams_top100.index = range(len(bigrams_top100.index))
trigrams_top100.index = range(len(trigrams_top100.index))
bigrams_top100['AverageRating']=avg_rating_column(bigrams_top100,'Bigram')
trigrams_top100['AverageRating']=avg_rating_column(trigrams_top100,'Trigram')
bigrams_top100.head()
bigrams_top100 = bigrams_top100[pd.notnull(bigrams_top100['ReviewType'])]
trigrams_top100=trigrams_top100[pd.notnull(trigrams_top100['ReviewType'])]
bigrams_top100_pros=bigrams_top100[bigrams_top100['ReviewType']=='Pros']
bigrams_top100_cons=bigrams_top100[bigrams_top100['ReviewType']=='Cons']
trigrams_top100_pros=trigrams_top100[trigrams_top100['ReviewType']=='Pros']
trigrams_top100_cons=trigrams_top100[trigrams_top100['ReviewType']=='Cons']
#Function to calculate cosine similarity
def get_phrase_similarity(p1,p2,model): 
    tokens_1=[t for t in p1.split() if t in model.wv.vocab]
    tokens_2=[t for t in p2.split() if t in model.wv.vocab]
    #compute cosine similarity using word embedings 
    cosine=0
    if (len(tokens_1) > 0 and len(tokens_2)>0):
        cosine=model.wv.n_similarity(tokens_1,tokens_2)
    return cosine
#Training Pros review text using CBOW,char n-gram and skipgram
pros_document_list=[]
for i in range(len(df['Pros_Modified_Text'])):
    pros_document_list.append(df['Pros_Modified_Text'][i].lower().split())
    #Train a CBOW Model
model_cbow = gensim.models.Word2Vec (pros_document_list, size=150, window=10, min_count=0, workers=10)
model_cbow.train(pros_document_list,total_examples=len(pros_document_list),epochs=10)
#Train a char n-gram model (subword information) with fastText
from gensim.models.fasttext import FastText
model_subword = FastText(pros_document_list, size=150, window=10, min_count=0, workers=10, min_n=3, max_n=6)  # instantiate
model_subword.train(pros_document_list,total_examples=len(pros_document_list),epochs=10)
#Train a SkipGram model
model_skipgram = gensim.models.Word2Vec (pros_document_list, size=150, window=10, min_count=0, workers=10, sg=1)
model_skipgram.train(pros_document_list,total_examples=len(pros_document_list),epochs=10)
#print(model_cbow.wv.vocab)
#Function to compute rating after taking similarities into consideration
def function4(df,list1,companyname,ngramType):
    rating_list=[]
    for i in list1:
        numerator=0
        denominator=0
        for j in i:
            numerator=numerator+float((df[(df['CompanyName']==companyname)&(df[ngramType]==j)]['AverageRating'])*(df[(df['CompanyName']==companyname)&(df[ngramType]==j)]['Frequency']))
            denominator=denominator+float((df[(df['CompanyName']==companyname)&(df[ngramType]==j)]['Frequency']))
        rating_list.append(float(numerator/denominator))
    return rating_list
def function3(df,companyname,ngramtype):
    model_cbow_list=[]
    model_skipgram_list=[]
    model_subword_list=[]
    for idx1, row1 in df[df['CompanyName']==companyname].iterrows():
        cbow_list=[]
        skipgram_list=[]
        subword_list=[]
        for idx2, row2 in df[df['CompanyName']==companyname].iterrows():
            cos_m1_score=get_phrase_similarity(row1[ngramtype],row2[ngramtype],model_cbow)
            cos_m2_score=get_phrase_similarity(row1[ngramtype],row2[ngramtype],model_skipgram)
            cos_m3_score=get_phrase_similarity(row1[ngramtype],row2[ngramtype],model_subword)
            cbow_list.append((row2[ngramtype],cos_m1_score))
            skipgram_list.append((row2[ngramtype],cos_m2_score))
            subword_list.append((row2[ngramtype],cos_m3_score))
        cbow_list=sorted(cbow_list, key = lambda x: x[1],reverse=True)
        skipgram_list=sorted(skipgram_list, key = lambda x: x[1],reverse=True)
        subword_list=sorted(subword_list, key = lambda x: x[1],reverse=True)
        model_cbow_list.append([word for word,idx in cbow_list][0:5])
        model_skipgram_list.append([word for word,idx in skipgram_list][0:5])
        model_subword_list.append([word for word,idx in subword_list][0:5])
    result=df[df['CompanyName']==companyname]
    result['cbow']=model_cbow_list
    result['skipgram']=model_skipgram_list
    result['subword']=model_subword_list
    result['cbow_rating']=function4(result,model_cbow_list,companyname,ngramtype)
    result['skipgram_rating']=function4(result,model_skipgram_list,companyname,ngramtype)
    result['subword_rating']=function4(result,model_subword_list,companyname,ngramtype)
    return result
Final_result_Bigrams=pd.concat([function3(bigrams_top100_pros,'Amazon','Bigram'),function3(bigrams_top100_cons,'Amazon','Bigram'),
                               function3(bigrams_top100_pros,'IBM','Bigram'),function3(bigrams_top100_cons,'IBM','Bigram'),
                               function3(bigrams_top100_pros,'Google','Bigram'),function3(bigrams_top100_cons,'Google','Bigram'),
                               function3(bigrams_top100_pros,'Microsoft','Bigram'),function3(bigrams_top100_cons,'Microsoft','Bigram')],ignore_index=True)

Final_result_Bigrams.to_csv('Final_result_Bigrams.csv')

Final_result_Trigrams=pd.concat([function3(trigrams_top100_pros,'Amazon','Trigram'),function3(trigrams_top100_cons,'Amazon','Trigram'),
                               function3(trigrams_top100_pros,'IBM','Trigram'),function3(trigrams_top100_cons,'IBM','Trigram'),
                               function3(trigrams_top100_pros,'Google','Trigram'),function3(trigrams_top100_cons,'Google','Trigram'),
                               function3(trigrams_top100_pros,'Microsoft','Trigram'),function3(trigrams_top100_cons,'Microsoft','Trigram')],ignore_index=True)
Final_result_Trigrams.to_csv('Final_result_Trigrams.csv')