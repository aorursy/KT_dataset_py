import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.options.display.float_format = '{:20,.2f}'.format

!jupyter nbextension enable --py --sys-prefix widgetsnbextension

import os

import ast

import scipy.spatial

from ipywidgets import interact

from ipywidgets import interact

import ipywidgets as widgets

import pandas as pd

from IPython.display import display

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

def world_cloud(wc,species='species'):

    mpl.rcParams['figure.figsize']=(12.0,18.0)    #(6.0,4.0)

    mpl.rcParams['font.size']=16                #10 

    mpl.rcParams['savefig.dpi']=100            #72 

    mpl.rcParams['figure.subplot.bottom']=.1 





    stopwords = set(STOPWORDS)





    wordcloud = WordCloud(

                              background_color='red',



                              max_words=100,

                              max_font_size=50, 

                              random_state=0

                             ).generate(str(wc))



    print(wordcloud)

    fig = plt.figure(1)

    plt.imshow(wordcloud)

    plt.title(species)

    plt.axis('off')

    plt.show()





from IPython.display import display, HTML

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

from string import punctuation

nltk.download('stopwords')

nltk.download('punkt')

stop_words = set(stopwords.words('english'))

from nltk import tokenize

def preprocess_sentence(text):

    text = text.replace('/', ' / ')

    text = text.replace('.-', ' .- ')

    text = text.replace('.', ' . ')

    text = text.replace('\'', ' \' ')

    text = text.lower()



    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]



    return ' '.join(tokens)
!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

#model = SentenceTransformer('bert-base-uncased')

base_comment=pd.read_csv("/kaggle/input/nyt-comments/CommentsMay2017.csv",low_memory=False)

base_comment.parentID.unique().shape
base_comment_sample=base_comment.sample(1000)

base_comment_sample['summary_preprocessed']=base_comment_sample['commentBody'].apply(lambda x:tokenize.sent_tokenize(x))
base_comment_sample.parentID.unique().shape
new_data_sent=base_comment_sample['summary_preprocessed'].apply(pd.Series).reset_index().melt(id_vars='index').dropna()[['index', 'value']].set_index('index')

new_data_sent.shape


new_data_sent=new_data_sent.merge(base_comment_sample,right_index=True,left_index=True,how='left')

new_data_sent['wrd_cnt']=new_data_sent['value'].str.split().str.len()

print("Total " + str(new_data_sent.shape[0]))

new_data_sent_strip=new_data_sent[new_data_sent['wrd_cnt']>6]

print("wrd cnt > 6 " + str(new_data_sent_strip.shape[0]))

new_data_sent_strip=new_data_sent_strip[new_data_sent_strip['wrd_cnt']<300]

print("wrd cnt < 300 " + str(new_data_sent_strip.shape[0]))

new_data_sent_strip=new_data_sent_strip.reset_index(drop=True)
corpus=new_data_sent_strip.value.values.tolist()

commentID_co=new_data_sent_strip.commentID.values.tolist()



len(corpus)
%%time

corpus_embeddings = model.encode(corpus,show_progress_bar=False)
new_data_sent_strip['emb']=corpus_embeddings
queries= ["Stunning visuals effects","This comment is toxic and bad","wonderful editorial","air pollution","politics"]
# query_embeddings = model.encode([queries[0]],show_progress_bar=False)[0]

# distances = scipy.spatial.distance.cdist([query_embeddings], new_data_sent_strip.emb.values, "cosine")[0]

# new_data_sent_strip['emb']=distances

# results = zip(range(len(distances)),commentID_co, distances)

# results = sorted(results, key=lambda x: x[2])





# for idx, commentID,distance in results[0:6]:

#     sc="(Score: %.4f)" % (1-distance)

#     display(HTML(corpus[idx].strip()  +sc))

#     display((new_data_sent_strip[new_data_sent_strip.commentID.isin([commentID])][['commentBody','commentID','parentID']]))

# new_data_sent_strip[new_data_sent_strip.commentID.isin([idx for idx, distance in results[:6]])]
def search_display(df,query,parent=None,model=model,corpus=corpus,closest_n=5):

    if parent is not None:

        df=df[df['parentID'].isin(parent)].copy()



    query_embeddings = model.encode([query],show_progress_bar=False)[0]

    distances = scipy.spatial.distance.cdist([query_embeddings], df.emb.values.tolist(), "cosine")[0]

    df['distances']=distances

    df=df.sort_values(by=['distances'],ascending=True)



    df=df.drop_duplicates('commentID')



    #display(df[['commentBody','commentID','parentID','value','distances']].head(closest_n))

    display(HTML(df[['commentBody','commentID','parentID','value','distances']].head(closest_n).style.set_properties(subset=['value'], \

            **{'font-weight': 'bold','font-size': '12pt','text-align':"left",'background-color': 'lightgrey','color': 'black'}).set_table_styles(\

                    [dict(selector='th', props=[('text-align', 'left'),('font-size', '12pt'),('background-color', 'pink'),('border-style','solid'),('border-width','1px')])]).hide_index().render()))

    return df[['commentBody','commentID','parentID','value','distances']].head(closest_n)

        



def search_utility(df,mode,parent=None,queries=["search"],model=model,corpus=corpus,closest_n=5):

    if mode.lower() =='input':

        while True:

            query = input('your question: ')

            if query.lower()  in ["exit",'break','quit']:

                break

            return search_display(df,query,parent,model,corpus,closest_n)

    elif mode.lower() =='list_search':

        for query in queries:

            display(HTML('your question: '+str(query)))

            return search_display(df,query,parent,model,corpus,closest_n)

            
json_res=[]

json_res_eval=[]

for quey in queries:

    ck2=search_utility(new_data_sent_strip,mode='list_search',queries=[quey],model=model,corpus=corpus,closest_n=5)

    json_res.append(ck2.to_json())

    json_res_eval.append(ast.literal_eval(ck2.to_json()))
json_res[0]
json_res_eval[0]


json_res=[]

json_res_eval=[]

for quey in queries:

    ck2=search_utility(new_data_sent_strip,mode='list_search',parent=[0],queries=queries,model=model,corpus=corpus,closest_n=5)

    json_res.append(ck2.to_json())

    json_res_eval.append(ast.literal_eval(ck2.to_json()))

    
#search_utility(mode='input',queries=queries,model=model,corpus=corpus,closest_n=5)
new_data_sent_strip['value_edit']=new_data_sent_strip['value'].apply(lambda x:preprocess_sentence(x))
wc=new_data_sent_strip['value_edit'].values.tolist()

wc=" ".join(wc)

world_cloud(wc,species='base')