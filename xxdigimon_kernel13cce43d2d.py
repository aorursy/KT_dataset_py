import numpy as np
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim.summarization import bm25
import nltk
import string
from tqdm import tqdm
input_path='/kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/5_materials/'
for dirname, _, filenames in os.walk(input_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
stool_df=pd.read_csv(input_path+'What do we know about viral shedding in stool_.csv')
print('cols')
print(stool_df.columns)
print('dataframe')
print(stool_df)
t_data=stool_df['Material']
print(t_data)
t_data=stool_df['Study Link']
print(t_data)
t_data=stool_df['Journal']
print(t_data)
t_data=stool_df['Study']
print(t_data)
t_data=stool_df['Conclusion']
print(t_data)
t_data=stool_df['Measure of Evidence']
print(t_data)
def preprocessing(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]
    tokens = [word for word in tokens if len(word) >= 3]
    tokens = [word.lower() for word in tokens]
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text= ' '.join(tokens)

    return preprocessed_text

covid_list=['covid-19','sars‐cov‐2']
def lemma(text):
    tokens=[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    token_set=[]
    for words in tokens:
        if words in covid_list:
            token_set.extend(covid_list)
        else:
            words=wn.synsets(words)
            lemma_list=[word.lemma_names() for word in words]
            print(lemma_list)
            for l in lemma_list:
                for w in l:
                    if w not in token_set:
                        token_set.append(w)
    return token_set


def clean_frame(df):
    df.duplicated('Study Link')
    df=df.drop_duplicates('Study Link')
    return df

    
def analysis(df):
    df=clean_frame(df)
    df_t=df[["Study","Conclusion"]].applymap(preprocessing)
    df_t['text']=df_t['Study']+df_t['Conclusion']
    df[["Study","Conclusion"]]=df_t[["Study","Conclusion"]]
    df['text']=df_t['text']
    df['score']=np.zeros(df.shape[0])
    text_values=df_t['text']
    return df,text_values

corpus=[]
records={}
for dirname, _, filenames in os.walk(input_path):
    for idx,filename in enumerate(filenames):
        df=pd.read_csv(os.path.join(dirname, filename))
        df,text=analysis(df)
        for id_t,t in tqdm(enumerate(text)):
            records[len(corpus)]=(idx,id_t)
            corpus.append(t.split())
bm25Model = bm25.BM25(corpus)
string='Human immune response to COVID-19'
string=preprocessing(string)
lemmas=lemma(string)
print(lemmas)
scores=bm25Model.get_scores(lemmas)
print(scores)
max_i=scores.index(max(scores))
print(max_i,records[max_i])
df=pd.read_csv(input_path+'What do we know about viral shedding in stool_.csv')
print(df.loc[0])
