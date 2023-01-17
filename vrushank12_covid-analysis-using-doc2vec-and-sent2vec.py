class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):
         return f'{self.paper_id}: \n{self.abstract[:400]}...\n{self.body_text[:400]}...'

def doi_url(d): return f'http://{d}' if d.startswith('doi.org') else f'http://doi.org/{d}'

def load_data_from_kaggle():
    file_counter = 0
    outer_loop = True

    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            file_counter += 1
            print(filename)
            if file_counter > 10:
                outer_loop = False
                break
        else:
            continue
        break
            
        

    # Any results you write to the current directory are saved as output.

    root_path = '/kaggle/input/CORD-19-research-challenge/'
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    filename = f'{root_path}metadata.csv'
    print(f'Metadata Filename: {filename}')
    len(all_json)

    # load in metadata
    #
    meta_data = pd.read_csv(filename, dtype={
            'doi': str,
            'pubmed_id': str,
            'Microsoft Academic Paper ID': str
            })

    print(len(meta_data))
    meta_data.head(2)

    # working with relevent fields
    #
    meta_data = meta_data[['sha','title','doi','abstract','publish_time','authors','journal']]

    meta_data.abstract = meta_data.abstract.fillna(meta_data.title)
    print(f'record count: {len(meta_data)}')
    meta_data.head(2)

    # remove older years
    meta_data['publish_time_year'] = meta_data['publish_time'].str[:4]
    meta_data = meta_data[meta_data['publish_time_year'] >= '2012']
    meta_data = meta_data.reset_index(drop=True) 
    
    # removal of null and duplicate
    #
    duplicate_paper = ~(meta_data.title.isnull() | meta_data.abstract.isnull() | meta_data.doi.isnull()) & (meta_data.duplicated(subset=['title', 'abstract']))
    meta_data = meta_data[~duplicate_paper].reset_index(drop=True)
    len(meta_data)
    meta_data.doi = meta_data.doi.fillna('').apply(doi_url)
    first_row = FileReader(all_json[0])
    print(first_row)

    # Load the data into dataframe
    #
    dict_ = {'paper_id': [], 'body_text': []}
    counter = 0
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
    
        try:
            content = FileReader(entry)
            dict_['paper_id'].append(content.paper_id)
            dict_['body_text'].append(content.body_text)
            
        except:
            counter+= 1       

    dataframe = pd.DataFrame(dict_, columns=['paper_id', 'body_text'])
    print(f'Total records rejected due to wrong structure: {counter}')
    dataframe.head()

    # perform join between metadata and json files
    #
    left = meta_data[['sha','title','doi','abstract','publish_time','authors','journal']]
    right = dataframe[['paper_id','body_text']]
    dataset = pd.merge(left, right, left_on='sha', right_on='paper_id', how='left')
    print(f'dataset ->: {len(dataframe)}')
    print(f'left ->: {len(left)}')
    print(f'right ->: {len(right)}')
    print(f'final ->: {len(dataset)}')

 
    gc.collect() 
    
    return dataset
import csv
import gc
import glob
import heapq
import json
import pickle
import os
import re
import string
import sys

#
# Libraries licensed under BSD
#
import numpy as np
import pandas as pd
from IPython.display import HTML
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
DATA_PATH = '/kaggle/input/CORD-19-research-challenge'
W2V_PATH = '/kaggle/input/covid19-w2v/'

df_orig_data = load_data_from_kaggle()
df_covid=df_orig_data
df_covid['body_text']=pd.Series(df_covid['body_text'], dtype="str")
df_covid['abstract']=pd.Series(df_covid['abstract'], dtype="str")
df_covid['title']=pd.Series(df_covid['title'], dtype="str")
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from langdetect import detect
# Utilize English only documents
def detect_lang(text):
    try:
        portion=text[0:400]
        lang=detect(portion)
    except Exception:
        lang=None
  
    return lang 
    

df_covid.drop_duplicates(['body_text'], inplace=True)

df_covid.dropna(inplace=True)


df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
df_covid.head()
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
plt.hist(np.clip(df_covid['body_word_count'], 0, 50000), bins=100, density=True)
plt.xlabel('Total words')
plt.show()
df_covid['column'] = np.where(df_covid['publish_time'] < '2019-12-31', 'Before_Covid', 'After_Covid')
df_covid['column'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.ylabel("Count of Papers", labelpad=14)
plt.title("Count of papers before covid vs after covid", y=1.02);
df_covid['column_1'] = np.where(df_covid['abstract'].isnull() & df_covid['title'].notnull() , 'Abstract Not Present', 'Abstract Present')
df_covid['column_1'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.ylabel("Count of Papers", labelpad=14)
plt.title("Total count of abstracts if title is not Null", y=1.02);
import plotly.express as px
value_counts = df_covid['journal'].value_counts()
value_counts_df = pd.DataFrame(value_counts)
value_counts_df['journal_name'] = value_counts_df.index
value_counts_df['count'] = value_counts_df['journal']
fig = px.bar(value_counts_df[0:10], 
             x="count", 
             y="journal_name",
             title='Most Common Journals in CORD-19 Dataset',
             orientation='h')
fig.show()
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
def splitDataFrameIntoSmaller(df, chunkSize = 1000): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf
covid19_synonyms = ['covid','covid-19','covid19','sarscov2',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b']
text_split=splitDataFrameIntoSmaller(df_covid)
for i in range(0,37) :
    text_split[i]['flagCol'] = np.where(text_split[i].body_text.str.contains('|'.join(covid19_synonyms)),1,0)
    text_split[i] = text_split[i][text_split[i]['flagCol']==1]
final_text=pd.concat(text_split, axis=0)
final_text=final_text.dropna(subset=['flagCol','body_text'])
final_text.drop( final_text[ final_text['body_word_count'] < 500 ].index , inplace=True)
final_text.describe(include='all')
final_text.to_csv('filtered_data_1.csv',index=False)
from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
import spacy
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))


# pos_words and extend words are some common words to be removed from body_text

pos_words = ['highest','among','either','seven','six','plus','strongest','worst','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
,'greatest','every','better','per','across','throughout','except','fewer','trillion','fewest','latest','least','manifest','unlike','eight','since','toward','largest','despite','via','finest','besides','easiest','must','million','oldest','behind','outside','smaller','nest','longest','whatever','stronger','worse','two','another','billion','best','near','nine','around','nearest','wechat','lowest','smallest','along','higher','three','older','greater','neither','inside','newest','lower','may','although','though','earlier','upon','five','ca','larger','us','whether','beyond','onto','might','one','out','unless','four','whose','can','fastest','without','ecobooth','broadest','easier','within','like', 'could','biggest','bigger','would','thereby','yet','timely','thus','also','avoid','know','usually','time','year','go','welcome','even','date',
             'used', 'following', 'go', 'instead', 'fundamentally', 'first', 'second', 'alone',
               'everything', 'end', 'also', 'year', 'made', 'many', 'towards', 'truly', 'last','introduction', 'abstract', 'section', 'edition', 'chapter','and', 'the', 'is', 'any', 'to', 'by', 'of', 'on','or', 'with', 'which', 'was','be','we', 'are', 'so',
                    'for', 'it', 'in', 'they', 'were', 'as','at','such', 'no', 'that', 'there', 'then', 'those',
                    'not', 'all', 'this','their','our', 'between', 'have', 'than', 'has', 'but', 'why', 'only', 'into',
                    'during', 'some', 'an', 'more', 'had', 'when', 'from', 'its', "it's", 'been', 'can', 'further',
                    'above', 'before', 'these', 'who', 'under', 'over', 'each', 'because', 'them', 'where', 'both',
                     'just', 'do', 'once', 'through', 'up', 'down', 'other', 'here', 'if', 'out', 'while', 'same',
                    'after', 'did', 'being', 'about', 'how', 'few', 'most', 'off', 'should', 'until', 'will', 'now',
                    'he', 'her', 'what', 'does', 'itself', 'against', 'below', 'themselves','having', 'his', 'am', 'whom',
                    'she', 'nor', 'his', 'hers', 'too', 'own', 'ma', 'him', 'theirs', 'again', 'doing', 'ourselves',
                     're', 'me', 'ours', 'ie', 'you', 'your', 'herself', 'my', 'et', 'al', 'may', 'due', 'de',
                     'one','two', 'three', 'four', 'five','six','seven','eight','nine','ten', 'however',
                     'i', 'ii', 'iii','iv','v', 'vii', 'viii', 'ix', 'x', 'xi', 'xii','xiii', 'xiv' 
               'often', 'called', 'new', 'date', 'fully', 'thus', 'new', 'include', 'http', 
               'www','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et',
               'al', 'author', 'figure','rights', 'reserved', 'permission', 'used', 'using', 'biorxiv',
               'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI','-PRON-']
extend_words =['used', 'following', 'go', 'instead', 'fundamentally', 'first', 'second', 'alone', 'everything', 'end', 'also', 'year', 'made', 'many', 'towards', 'truly', 'last', 'often', 'called', 'new', 'date', 'fully', 'thus', 'new', 'include', 'http', 'www','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure','rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI','-PRON-']

pos_words.extend(extend_words)
pos_words
stop_words = stop_words.union(pos_words)

def text_preprocess(text):
    lemma = nltk.wordnet.WordNetLemmatizer()
    
    #Convert to lower
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    
    #Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split()]
    
    lemmatized = []
    #Lemmatize non-stop words and save
    other_words = ['virus','study','viral','human','infection'] # common words to remove specific to these articles
    for word in text:
        if word not in stop_words:
            x = lemma.lemmatize(word)
            if x not in other_words:
                lemmatized.append(x)
   
    result = " ".join(lemmatized)
    return result

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
final_text['language'] = final_text['abstract'].apply(lambda x: detect_lang(x))
final_text = final_text[final_text['language'] == 'en'] 
final_text['body_text_processed'] = final_text['body_text'].apply(text_preprocess)
final_text['body_text_processed_1'] = final_text['body_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
import gensim
tokenized_doc = []
for d in final_text['body_text_processed']:
    tokenized_doc.append(word_tokenize(d.lower()))
tagged_data = [gensim.models.doc2vec.TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]

model_docs = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=4,sample=0.0008, epochs=50,dm=0,dbow_words=0)
# Build the Volabulary
model_docs.build_vocab(tagged_data)
# Train the Doc2Vec model
model_docs.train(tagged_data, total_examples=model_docs.corpus_count, epochs=model_docs.epochs)
test_data = ("artificial intelligence and deep learning treatment")
test_data=word_tokenize(text_preprocess(test_data))
ivec = model_docs.infer_vector(test_data, steps=100,alpha=0.001)
similar=model_docs.docvecs.most_similar(positive=[ivec], topn=10)
def Extract(lst): 
    return list(list(zip(*lst))[0])
top_articles_1=final_text.iloc[Extract(similar)]
top_articles_1.iloc[:,[12,0,2]]

from gensim.similarities.index import AnnoyIndexer
annoy_index = AnnoyIndexer(model_docs, 500)
approximate_neighbors = model_docs.docvecs.most_similar(positive=[model_docs.infer_vector(test_data)],topn=20
, indexer=annoy_index)
#1) open code of summarization
top_articles=final_text.iloc[Extract(similar)].iloc[:2,2].str.cat(sep=', ')
import re
text_first=top_articles
text_first = re.sub(r'\[[0-9]*\]', ' ', text_first)
text_first = re.sub(r'\s+', ' ', text_first)
formatted_article_text = re.sub('[^a-zA-Z]', ' ', text_first)
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
sentence_list = nltk.sent_tokenize(text_first)
stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(formatted_article_text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)
# 2) use of neural network technique
from gensim.summarization.summarizer import summarize
print(summarize(top_articles,0.09))
task_ques_list = ['deep learning treatment',\
                  'ECMO death',\
                  'Resources long term care facilities.', \
                  'Mobilization of surge medical staff to address shortages in overwhelmed communities', \
                  'Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies', \
                  'Outcome mechanical ventilation', \
                  'Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.', \
                  'Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.', \
                  'Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.', \
                  'Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.', \
                  'Guidance on the simple things people can do at home to take care of sick people and manage disease', \
                  'Oral medications that might potentially work', \
                  'Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcome', \
                  'Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials', \
                  'Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials', \
                  'Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)']

def Extract(lst): 
    return list(list(zip(*lst))[0])
def Extract_1(lst): 
    return list(list(zip(*lst))[1])
from gensim.summarization.summarizer import summarize
final_docs = pd.DataFrame(columns=[ 'Similarity score','Question', 'Title','Summarize', 'Authors', 'Published_Date', 'Link'])
for i, info in enumerate(task_ques_list):
    test_data=word_tokenize(text_preprocess(info))
    ivec = model_docs.infer_vector(test_data, steps=100,alpha=0.001)
    similar=model_docs.docvecs.most_similar(positive=[ivec], topn=5)
    df =  final_text.iloc[Extract(similar)]
    df['Body_text_summarize'] = df['body_text'].apply(summarize)
    abstracts = df['abstract']
    Summarization=df['Body_text_summarize']
    titles = df['title']
    similar_1=Extract(similar)
    similar_2=Extract_1(similar)
    for l in range(len(similar)):
        final_docs = final_docs.append({ 'Similarity score': similar_2[l] ,'Summarize': Summarization.iloc[l], 'Question': info[:100], 'Title': titles.iloc[l], \
                                        'Authors': df['authors'].iloc[l], 'Published_Date': df['publish_time'].iloc[l], \
                                        'Link': df['doi'].iloc[l] },ignore_index=True)
        
final_docs
# Function to take an URL string and text string and generate a href for embedding
#
def href(val1,val2):
    return '<a href="{}" target="_blank">{}</a>'.format(val1,val2)
#
# Function to add column width for a particular column within the HTML string
#
def setColWidth(html, col, width):
    html = re.sub('<th>'+ col,  '<th width="'+ width+ '%">'+col, html)
    return html

# Function to replace additional authors with 'et al.'
def etal(val):
    if isinstance(val, float): 
        return_val = " "
    else:
        if ';' in val:
            if ',' in val:
                return_val = re.sub(',.*', ' et al.', val)
            else:
                return_val = re.sub(';.*', ' et al.', val)
        else:
            return_val = " "
    return return_val

def setCaption(html, caption):
    html = re.sub('mb-0">', 'mb-0"><caption>' + caption + '</caption>', html)
    return html

def format_answer(val):
    val = val.replace("\n","")
    val = val.replace("https:","")
    val = val.replace("http:","")
    val = val.replace("www.","")
    return val

  
#
# Function to generate HTML string
#
def createHtmlTable(df,prefix):
    # CSS string to justify text to the left
    css_str ='<style>.dataframe thead tr:only-child th {text-align: left;}.dataframe thead th {text-align: left;}.dataframe tbody tr th:only-of-type {vertical-align: middle;}.dataframe tbody tr th {vertical-align: top;}.dataframe tbody tr td {text-align: left;}.dataframe caption {display: table-caption;text-align: left;font-size: 12px;color: black;font-weight: bold;}</style>'
    
    # Create a new Title column and combines the URL link to allow the user to open the source document in another tab
    df['Title'] = df.apply(lambda row : href (row['Link'], row['Title']), axis=1)
    df['Authors'] = df.apply(lambda row : etal (row['Authors']), axis=1)  
    df['Summarize'] = df.apply(lambda row : format_answer (row['Summarize']), axis=1)
    
    # Generate HTML table string    
    html_str = df[[ 'Title', 'Authors', 'Published_Date',  'Summarize' ]].to_html(render_links=True, index=False,  classes="table table-bordered table-striped mb-0")
    html_str = html_str + '<hr>'
    
    # Set table caption
    html_str = setCaption (html_str, 'Top Published Documents')
    
    # Perform a few adjustments on the HTML string to make it even better
    html_str = re.sub('&lt;', '<', html_str)
    html_str = re.sub('&gt;', '>', html_str)
    html_str = setColWidth(html_str, 'Title', '31')
    html_str = setColWidth(html_str, 'Authors', '13')
    html_str = setColWidth(html_str, 'Published_Date', '11')
    html_str = setColWidth(html_str, 'Summarize', '45')  
    
    
    # Return the final HTML table string for display
    return css_str + prefix + html_str

#
#Function to generate HTML Q&A + table that can be displayed
#
pd.set_option('mode.chained_assignment', None)
def create_html_per_question(df, ques_list):
    i=0
    for ques in task_ques_list:
        i=i+1
        new_section = 1
        prefix = '<h4>#' + str(i) + ': ' + ques + '</h4>'
        df = (final_docs.loc[final_docs['Question'] == ques])
        for index, data in df.iterrows():
            if data['Question'] == ques and new_section == 1:
                answer_w2v = data['Question']
                prefix = prefix + '<p>' + answer_w2v + '</p>'
                new_section = 0
                small_table = final_docs.loc[final_docs["Question"] == ques]
                display(HTML(createHtmlTable(small_table,prefix)))
create_html_per_question(final_docs, task_ques_list)
#running word2vec to create sentence vectors
from gensim.models import Word2Vec
model_ww = Word2Vec( min_count=5,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,)
model_ww.build_vocab(tokenized_doc)
model_ww.train(tokenized_doc, total_examples=model_ww.corpus_count,epochs=10)
model_ww.wv.most_similar(positive=["risk"])
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.summarization.summarizer import summarize
final_docs_1 = pd.DataFrame(columns=[ 'Similarity score','Question', 'Title','Summarize', 'Authors', 'Published_Date', 'Link'])
for i, info in enumerate(task_ques_list):
    test_data=word_tokenize(text_preprocess(info))
    ivec = model_docs.infer_vector(test_data, steps=100,alpha=0.001)
    similar=model_docs.docvecs.most_similar(positive=[ivec], topn=15)
    df =  final_text.iloc[Extract(similar)]
    df['Body_text_summarize'] = df['body_text'].apply(summarize)
    abstracts = df['abstract']
    Summarization=df['Body_text_summarize']
    titles = df['title']
    similar_1=Extract(similar)
    similar_2=Extract_1(similar)
    for l in range(len(similar)):
        final_docs_1 = final_docs_1.append({ 'Similarity score': similar_2[l] ,'Summarize': Summarization.iloc[l], 'Question': info[:100], 'Title': titles.iloc[l], \
                                        'Authors': df['authors'].iloc[l], 'Published_Date': df['publish_time'].iloc[l], \
                                        'Link': df['doi'].iloc[l] },ignore_index=True)
from collections import Counter

cnt = Counter({k:v.count for k, v in model_ww.wv.vocab.items()})

from itertools import chain
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from functools import partial
from tqdm import tqdm_notebook
all_sents = chain(*map(sent_tokenize,final_docs_1['Summarize']))
all_sents = list(all_sents)
my_nums = list(filter(lambda s: len(s.split()) >= 5, all_sents))
def sent2vector(sent):
    words = word_tokenize(sent.lower())
    
    # Here we weight-average each word in sentence by 1/log(count[word])
    emb = [model_ww[w] for w in words if w in model_ww]
    weights = [1./cnt[w] for w in words if w in model_ww]
    
    if len(emb) == 0:
        return np.zeros(300, dtype=np.float32)
    else:
        return np.dot(weights, emb) / np.sum(weights)
sent_vectors = np.array(list(map(sent2vector, tqdm_notebook(my_nums))))
from sklearn.neighbors import KDTree
kdtree = KDTree(sent_vectors)
def search(sent, k=3):
    sent_vec = sent2vector(sent)
    closest_sent = kdtree.query(sent_vec[None], k)[1][0]
    
    return [all_sents[i] for i in closest_sent]

search("ecmo death",9)