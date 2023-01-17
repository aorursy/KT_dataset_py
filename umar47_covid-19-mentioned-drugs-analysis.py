import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from tqdm.notebook import tqdm
pd.set_option('display.max_colwidth', -1)
import os
dir_list = [
    '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv',
    '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset',
    '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license',
    '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset'
]
results_list = list()
for target_dir in dir_list:
    
    print(target_dir)
    
    for json_fp in tqdm(glob(target_dir + '/*.json')):

        with open(json_fp) as json_file:
            target_json = json.load(json_file)

        data_dict = dict()
        data_dict['doc_id'] = target_json['paper_id']
        data_dict['title'] = target_json['metadata']['title']

        abstract_section = str()
        for element in target_json['abstract']:
            abstract_section += element['text'] + ' '
        data_dict['abstract'] = abstract_section

        full_text_section = str()
        for element in target_json['body_text']:
            full_text_section += element['text'] + ' '
        data_dict['full_text'] = full_text_section
        
        results_list.append(data_dict)
        
    
df_results = pd.DataFrame(results_list)
df_results.head()        
df_results.info()
dfdrugs=pd.read_csv('/kaggle/input/drug-data/drugsComTest_raw.csv')
dfdrugs.info()
freq = pd.DataFrame(' '.join(df_results['full_text']).split(), columns=['drugName']).drop_duplicates()
freq.head()
result = pd.merge(freq, dfdrugs, on=['drugName'])
result.head()
result.sample(10)
result.drugName.value_counts()
result=result.where(result['rating']>7.0).dropna()
result.where(result['condition'].str.contains('Cough')).dropna().sample(5)
#result.where(result['condition'].str.contains('Headache')).dropna()
#result.where(result['condition'].str.contains('Cluster Headaches')).dropna()
articles=df_results['full_text'].values
for text in articles:
    for sentences in text.split('.'):
        if 'Benzonatate' in sentences:
            print(sentences)        
for text in articles:
    for sentences in text.split('.'):
        if 'Codeine' in sentences:
            print(sentences)
dfdrugs2=pd.read_csv('../input/usp-drug-classification/usp_drug_classification.csv')
dfdrugs2['drugName']=dfdrugs2['drug_example']
dfdrugs2.head()
result2 = pd.merge(freq, dfdrugs2, on=['drugName'])
result2.head()
result2['usp_category'].value_counts()[:10]
antivirals=list(result2.drugName.where(result2['usp_category']=='Antivirals').dropna().unique())
antivirals[:5]
for text in articles:
    for sentences in text.split('.'):
        if 'entecavir' in sentences:
            print(sentences) 
CA=list(result2.drugName.where(result2['usp_category']=='Cardiovascular Agents').dropna().unique())
Cardiovascular_Agents =[]
for text in articles:
    for sentences in text.split('.'):
        if any(word in sentences for word in CA):
            Cardiovascular_Agents .append(sentences)
            #print(sentences) 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(Cardiovascular_Agents))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
antivirals_all=[]
for text in articles:
    for sentences in text.split('.'):
        if any(word in sentences for word in antivirals):
            antivirals_all.append(sentences)
            #print(sentences) 
wordcloud = WordCloud(stopwords =stopwords, width=1000, height=500).generate("+".join(antivirals_all))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
df = pd.DataFrame(antivirals_all, columns=['sentence']) 
def matcher(x):
    for i in antivirals:
        if i.lower() in x.lower():
            return i
    else:
        return np.nan
    
df['Match'] = df['sentence'].apply(matcher)    
df.sample(5)
df['Match'].value_counts()[:10]
df['sentence'] = df['sentence'].astype(str)
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['sentence'] = df['sentence'].str.replace('[^\w\s]','')

stop = stopwords.words('english')
df['sentence'] = df['sentence'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
st = PorterStemmer()
df['sentence'] = df['sentence'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

def senti(x):
    return TextBlob(x).sentiment  
 
df['senti_score'] = df['sentence'].apply(senti)
df.sample(5)
polarity=[]
subjectivity=[]
for i, j in df.senti_score:
    polarity.append(i)
    subjectivity.append(j)
df['subjectivity']=subjectivity
df['polarity']=polarity
df.where(df['polarity']==1).dropna().head(5)
vote_data=df[['Match', 'polarity']]
items=vote_data['Match']#item's column
votes=vote_data['polarity']#vote's column
num_of_votes=len(items)
    
m=min(votes)
avg_votes_for_item=vote_data.groupby('Match')['polarity'].mean()#mean of each item's vote
mean_vote=np.mean(votes)#mean of all votes
pol=pd.DataFrame(((num_of_votes/(num_of_votes+m))*avg_votes_for_item)+((m/(num_of_votes+m))*mean_vote))
pol.head()
vote_data=df[['Match', 'subjectivity']]
items=vote_data['Match']#item's column
votes=vote_data['subjectivity']#vote's column
num_of_votes=len(items)
    
m=min(votes)
avg_votes_for_item=vote_data.groupby('Match')['subjectivity'].mean()#mean of each item's vote
mean_vote=np.mean(votes)#mean of all votes
sub=pd.DataFrame(((num_of_votes/(num_of_votes+m))*avg_votes_for_item)+((m/(num_of_votes+m))*mean_vote))
sub.head()
on_weighted_score=pd.concat([pol, sub.reindex(pol.index)], axis=1).sort_values(by=['polarity', 'subjectivity'], ascending=False)
value_count=pd.DataFrame(df['Match'].value_counts())
bests=pd.concat([value_count, on_weighted_score.reindex(value_count.index)], axis=1).sort_values(by=['polarity', 'subjectivity'], ascending=False)
bests.head(5)