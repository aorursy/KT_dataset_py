!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
!jupyter nbextension enable --py --sys-prefix widgetsnbextension
# Helper packages.
import os
import pandas as pd
pd.set_option('max_colwidth', 1000)
pd.set_option('max_rows', 100)
import numpy as np
np.set_printoptions(threshold=10000)
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import re
import json
from tqdm import tqdm
import textwrap
import importlib as imp
from scipy.spatial.distance import cdist

# Packages with tools for text processing.
# if you have not downloaded stopwords, run the following line
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
import scispacy
import spacy

# Packages for working with text data.
from sklearn.feature_extraction.text import CountVectorizer

# Packages for getting data ready for and building a LDA model
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

# Package for FastText
import fasttext

# Other plotting tools.
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud
from IPython.display import display
import ipywidgets as widgets

# Don't show warnings
import warnings
warnings.filterwarnings("ignore")

# Print current directory
os.getcwd()
# set toggle
source_column = 'text' # if abstract, change to 'abstract'
id_colname = 'cord_uid'
split_sentence_by = '(?<=\.) ?(?![0-9a-z])'
        
# set covid variables
cov_earliest_date = '2019-12-01'
cov_key_terms = ['covid\W19','covid19', 'covid', '2019\Wncov', '2019ncov', 'ncov\W2019','sars\Wcov\W2', 'sars\Wcov2', '新型冠状病毒']
cov_related_terms = '(novel|new)( beta| )coronavirus'

# path
# kaggle
input_data_path = '/kaggle/input/CORD-19-research-challenge/'
working_data_path = '/kaggle/input/cov19-pickles/'
# load pickled file: no need to run this section once this file is loaded
meta_full_text = pickle.load(open(working_data_path + 'all_papers.pkl', 'rb'))
# import metadata
metadata = pd.read_csv(input_data_path + 'metadata.csv', encoding='utf-8').replace({pd.np.nan: None})
print(metadata.shape)
metadata.isnull().sum(axis=0)
# create sha array to match full text 
def pdf_or_pmc(r):
    if r.has_pdf_parse: return 'pdf_json'
    if r.has_pmc_xml_parse: return 'pmc_json'
    return ''
metadata['sha_arr'] = metadata.apply(lambda r: r.sha.split(';') if r.sha is not None else [], axis=1)
metadata['full_text_file_path'] = metadata.apply(lambda r: np.unique(['/'.join([r.full_text_file, r.full_text_file, pdf_or_pmc(r), sha.strip()]) if r.has_pdf_parse or r.has_pmc_xml_parse else '' for sha in r.sha_arr]) if len(r.sha_arr) > 0 else [], axis=1)

# clean/format various publish time
metadata['publish_time'] = metadata['publish_time'].str.replace(' ([a-zA-Z]{3}-[a-zA-Z]{3})|(Spring)|(Summer)|(Autumn)|(Fall)|(Winter)','',regex=True).str.strip()

metadata['publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y-%m-%d', errors='coerce')
mask = metadata.publish_time_.isnull()

metadata.loc[mask, 'publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y %B', errors='coerce')
mask = metadata.publish_time_.isnull()

metadata.loc[mask, 'publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y %b', errors='coerce')
mask = metadata.publish_time_.isnull()

metadata.loc[mask, 'publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y %B %d', errors='coerce')
mask = metadata.publish_time_.isnull()

metadata.loc[mask, 'publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y %b %d', errors='coerce')
mask = metadata.publish_time_.isnull()

metadata.loc[mask, 'publish_time_'] = pd.to_datetime(metadata.publish_time, format='%Y', errors='coerce')
mask = metadata.publish_time_.isnull()

invalid_dates = metadata.loc[mask,:].shape[0]

print("In total, {} entries in metadata are not assigned to a valid date.".format(invalid_dates))

metadata.publish_time = metadata.publish_time_
metadata.drop(['publish_time_'], inplace=True, axis=1)
mask = metadata['full_text_file_path'].apply(lambda r: len(r)>1)
print("In total, {} entires in metadata have multiple sha. ".format(len(metadata.loc[mask,])))
# extract full text from JSON files
def get_paper_info(json_data):
    return ' '.join([t['text'] for t in json_data['body_text']])

full_text = []
for r in tqdm(metadata.to_dict(orient='records')):
    record = []
    for p in r['full_text_file_path']:
        with open(input_data_path + p + '.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            record.append(get_paper_info(data))
    full_text_ = '\n'.join(np.unique(record)) if len(record) > 0 else None
    full_text.append(full_text_)
metadata['full_text'] = full_text

# drop records with empty abstract AND empty full text
meta_full_text = metadata
meta_full_text[source_column]= np.where(meta_full_text['full_text'].isnull(), meta_full_text['abstract'], meta_full_text['full_text'])
meta_full_text = meta_full_text.dropna(subset = [source_column]).reset_index(drop=True)

# check duplicated text: most likely due to publications on different journals - in which case we keep the latest one
print('In total, {} of the rows have a duplicated {} column, and there are a total of {} duplicated {} entries.'.format(sum([len(g) for k, g in meta_full_text.groupby(source_column) if len(g) > 1]), source_column, len([1 for k, g in meta_full_text.groupby(source_column) if len(g) > 1]), source_column))
meta_full_text = meta_full_text.sort_values('publish_time', ascending=False).drop_duplicates(source_column)

# check duplicated cord_uid: most likely due to publications on different journals - in which case we keep the latest one
print('In total, {} of the rows have a duplicated {} column, and there are a total of {} duplicated {} entries.'.format(sum([len(g) for k, g in meta_full_text.groupby(id_colname) if len(g) > 1]), id_colname, len([1 for k, g in meta_full_text.groupby(id_colname) if len(g) > 1]), id_colname))
meta_full_text = meta_full_text.sort_values('publish_time', ascending=False).drop_duplicates(id_colname)

# drop redundant columns
meta_full_text.drop(['sha', 'pmcid', 'pubmed_id', 
                     'Microsoft Academic Paper ID', 'has_pdf_parse', 
                     'has_pmc_xml_parse', 'full_text_file', 'sha_arr',
                     'full_text_file_path', 'full_text'], inplace=True, axis=1)

print(meta_full_text.shape)
print(meta_full_text.columns)

# save to pickle
pickle.dump(meta_full_text, open(working_data_path + 'all_papers.pkl', 'wb'))
corpus = meta_full_text[source_column]
# get stop words
stop_words=stopwords.words('english')

# custom CORD19 stop words, mostly from Daniel Wolffram's submission "Topic Modeling: Finding Related articles"
cord_stopwords = ['doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights', 'reserved', 
                  'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier', 'PMC', 'CZI','-PRON-',
                  'abstract']

for word in tqdm(cord_stopwords):
    if (word not in stop_words):
        stop_words.append(word)
    else:
        continue
nlp_lg = spacy.load('en_core_sci_lg',disable=['tagger', 'parser', 'ner'])
nlp_lg.max_length = 2000000
# add out stop word list to the spacy model stop list
for w in tqdm(stop_words):
    nlp_lg.vocab[w].is_stop = True
# Function that removes substrings before it's tokenized and stemmed
def removeParenthesesNumbers(v):
    char_list_rm = ['[(]','[)]','[′·]']
    char_list_rm_spc = [' no[nt]-',' non', ' low-', ' high-']
    v = re.sub('|'.join(char_list_rm), '', v)
    v = re.sub('|'.join(char_list_rm_spc), ' ', v)
    return(v)
sentence_test = '($2196.8)/case (in)fidelity μg μg/ml a=b2 www.website.org α-gal 2-len a.'
def spacy_tokenizer(sentence):
    sentence = removeParenthesesNumbers(sentence)
    # define types of tokens that should be removed using regex
    token_rm = ['(www.\S+)','(-[1-9.])','([∼≈≥≤≦⩾⩽→μ]\S+)','(\S+=\S+)','(http\S+)']
    tokenized_list = [word.lemma_ for word in nlp_lg(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space)]
    tokenized_list = [word for word in tokenized_list if not re.search('|'.join(token_rm),word)]
    tokenized_list = [word for word in tokenized_list if len(re.findall('[a-zA-Z]',word))>1]
    tokenized_list = [word for word in tokenized_list if re.search('^[a-zA-Z0-9]',word)]
    return tokenized_list
spacy_tokenizer(sentence_test)
# X, valid_tokens: No need to run the next cell if loaded
X = pickle.load(open(working_data_path + 'TM_X.pkl', 'rb'))
valid_tokens = pickle.load(open(working_data_path + 'TM_valid_tokens.pkl', 'rb'))
# Initialize `CountVectorizer`. Remove common and sparse terms
vec = CountVectorizer(max_df = .8, min_df = .001, tokenizer = spacy_tokenizer)

# Transform the list of snippets into DTM.
X = vec.fit_transform(tqdm(corpus))

valid_tokens = vec.get_feature_names()
print(len(valid_tokens))
print(valid_tokens)

pickle.dump(X, open(working_data_path + 'TM_X.pkl', 'wb'))
pickle.dump(valid_tokens, open(working_data_path + 'TM_valid_tokens.pkl', 'wb'))
# Examine common and sparse terms that are removed; only run if you want to examine these terms
vec_all = CountVectorizer(tokenizer = spacy_tokenizer)

# Transform the list of snippets into DTM.
X_all = vec_all.fit_transform(tqdm(corpus))

valid_tokens_all = vec_all.get_feature_names()

com_sprs_trms = [word for word in valid_tokens_all if word not in valid_tokens]
print(len(com_sprs_trms))
print(com_sprs_trms)
# No need to run the next cell if loaded
texts = pickle.load(open(working_data_path + 'TM_texts.pkl', 'rb'))
arr = X.toarray()
texts = []
for i in tqdm(range(arr.shape[0])):
    text = []
    for j in range(arr.shape[1]):
        occurrence = arr[i,j]
        if occurrence > 0:
            text.extend([valid_tokens[j]] * occurrence)
    texts.append(text)
pickle.dump(texts, open(working_data_path + 'TM_texts.pkl', 'wb'))
np.random.seed(1)
dictionary = gensim.corpora.Dictionary(texts)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
# No need to run the next two cell if loaded
bow_corpus = pickle.load(open(working_data_path + 'TM_bow_corpus.pkl', 'rb'))
bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
print(bow_corpus[0])
pickle.dump(bow_corpus, open(working_data_path + 'TM_bow_corpus.pkl', 'wb'))
bow_doc_1 = bow_corpus[0]
print(corpus[corpus.index[0]])
for i in tqdm(range(len(bow_doc_1))):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_1[i][0], dictionary[bow_doc_1[i][0]],bow_doc_1[i][1]))
def compute_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):
        model = gensim.models.LdaMulticore(corpus = corpus, id2word = dictionary, num_topics = num_topics, random_state = 1)
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('Number of topics: {}, Coherence value: {}'.format(num_topics, coherencemodel.get_coherence()))

    return model_list, coherence_values
# No need to run the next cell if loaded
model_list = pickle.load(open(working_data_path + 'TM_model_list.pkl', 'rb'))
coherence_values = pickle.load(open(working_data_path + 'TM_coherence_values.pkl', 'rb'))
model_list, coherence_values = (compute_coherence_values(dictionary = dictionary, 
                                                         corpus = bow_corpus, # if we want to train the model using tfidf, then use corpus_tfidf
                                                         texts = texts, 
                                                         start = 10, limit = 20, step = 1))
pickle.dump(model_list, open(working_data_path + 'TM_model_list.pkl', 'wb'))
pickle.dump(coherence_values, open(working_data_path + 'TM_coherence_values.pkl', 'wb'))
limit=20; start=10; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
topic_num = 17
# No need to run the next cell if loaded
lda_model = pickle.load(open(working_data_path+'TM_lda_model.pkl','rb'))
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics = topic_num, id2word = dictionary, workers = 4, passes = 2)
print(lda_model)
pickle.dump(lda_model, open(working_data_path+'TM_lda_model.pkl','wb'))
# topic_list = open('models/'+text_prefix+'lda_model'+suffix+'.txt','a+')
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
#     topic_list.write('Topic: {} Word: {}\n'.format(idx, topic))
# topic_list.close()
from IPython.display import HTML
HTML(filename=working_data_path + 'TM_lda_vis.html')
# #Prepare LDA vis object by providing:
# vis = pyLDAvis.gensim.prepare(lda_model,   #<- model object
#                               bow_corpus, #<- corpus object
#                               dictionary)  #<- dictionary object
# pyLDAvis.display(vis)
# pyLDAvis.save_html(vis, working_data_path + 'TM_lda_vis.html')
# No need to run the next TWO cell if loaded
data_predictions = pickle.load(open(working_data_path + 'TM_lda_data_predictions.pkl','rb'))
# create an empty data frame
doc_topic_df = pd.DataFrame(columns = range(topic_num))
# record all topic probabilities
for i in tqdm(range(dictionary.num_docs)): 
    doc_topic_df = doc_topic_df.append(dict(lda_model.get_document_topics(bow_corpus[i])),ignore_index=True)
doc_topic_df['index'] = corpus.index
data_predictions = meta_full_text.merge(doc_topic_df, left_index=True, right_on=['index'])
data_predictions.columns = ['topic ' + str(column) if type(column)==int else column for column in data_predictions.columns]
data_predictions.columns
pickle.dump(data_predictions, open(working_data_path + 'TM_lda_data_predictions.pkl','wb'))
cols = ['#029386','#f97306','#ff796c','#cb416b','#fe01b1',
        '#fd411e','#be03fd','#1fa774','#04d9ff','#c9643b',
        '#7ebd01','#155084','#fd4659','#06b1c4','#8b88f8',
        '#029386','#f97306']
topics = lda_model.show_topics(num_words=20,num_topics=topic_num,formatted=False)
cloud = WordCloud(background_color='black',color_func=lambda *args, **kwargs: cols[i],prefer_horizontal=1.0, font_step=1, width=350,height=200)
# Make word clouds for all topics
fig, axes = plt.subplots(3, 6, figsize=(25,10), sharex=True, sharey=True)

for i, ax in tqdm(enumerate(axes.flatten())):
    if i < len(topics):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=50)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
    else:
        ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
# filter covid-related papers
def get_covid19(data):
    print('We define a paper to be Covid-19 related if')
    print('a. the paper contains key terms in [{}], or'.format('|'.join(cov_key_terms)))
    print('b. the paper contains related terms in [{}] and is published after {}, or'.format(cov_related_terms, cov_earliest_date))
    print('a. the paper is marked as WHO #Covidence and either contains related terms in [{}] or is published after {}.'.format(cov_related_terms, cov_earliest_date))

    cov_key_terms_mask = data[source_column].str.lower().str.contains('|'.join(cov_key_terms))
    cov_related_terms_mask = data[source_column].str.lower().str.contains(cov_related_terms)

    data['WHO_covidence'] = False
    data.loc[~data['WHO #Covidence'].isnull(), 'WHO_covidence'] = True

    data['contain_key_terms'] = False
    data.loc[cov_key_terms_mask,'contain_key_terms'] = True

    data['contain_related_terms'] = False
    data.loc[cov_related_terms_mask,'contain_related_terms'] = True

    data['after_earliest_date'] = False
    data.loc[data.publish_time>= cov_earliest_date,'after_earliest_date'] = True

    covid19 = data[data.contain_key_terms | (data.contain_related_terms & data.after_earliest_date) | (data.WHO_covidence & (data.contain_related_terms | data.after_earliest_date))]
    covid19.reset_index(drop=True, inplace=True)
    print("There are a total number of {} papers satisfying the above definition".format(len(covid19)))
    return covid19
covid19 = get_covid19(meta_full_text)
print(covid19.shape)
covid19[:1]
# first select parameters for fasttext

search_column = 'abstract'

search_text = 'raw_' + search_column
selected_text = 'raw_' + source_column
selected_epoch = 3
selected_m = 'cbow'

model_name_suffix = selected_m + '_' + selected_text + '_epoch' + str(selected_epoch)
search_name_suffix = selected_m + '_' + search_text + '_epoch' + str(selected_epoch)
# lookup paper information for each sentence

def sentence_to_paper(df, id_colname, text_colname, topic_colname_prefix, split_sentence_by):
    # link sentences to a paper: sents_in_paper
    sents_in_paper = dict()
    papers = [(paper[id_colname], paper[text_colname]) if paper[text_colname] is not None else (paper[id_colname], "") for paper in df.to_dict(orient='row')]
    sents = [(paper[0], re.split(split_sentence_by, paper[1])) for paper in papers]
    sent_order = 1
    for pair in np.concatenate([list(zip(id, sent)) for id, sent in [([sent[0]]*len(sent[1]),sent[1]) for sent in sents]]):
        sent = pair[1]
        if sent not in sents_in_paper:
            sents_in_paper[sent] = (pair[0], sent_order)
            sent_order += 1
            
    # lookup paper information: paper_lookup        
    paper_lookup = dict()
    for paper in df.to_dict(orient='records'):
        id = str(paper[id_colname])
        if id not in paper_lookup:
            paper[topic_colname_prefix] = dict((k, paper[k]) for k in paper.keys() if k.startswith(topic_colname_prefix))
            paper_lookup[id] = paper    
    
    return sents_in_paper, paper_lookup
sents_in_paper, paper_lookup = sentence_to_paper(covid19, id_colname=id_colname, text_colname=search_column, topic_colname_prefix='topic', split_sentence_by=split_sentence_by)
# pickle.dump(sents_in_paper, open(working_data_path + 'fasttext_model_' + search_column + '_sents_in_paper.pkl', 'wb'))
sents_in_paper = pickle.load(open(working_data_path + 'fasttext_model_' + search_column + '_sents_in_paper.pkl', 'rb'))
# No need to run next cell if loaded
# load model from memory
model = fasttext.load_model(working_data_path + 'fasttext_model_' + model_name_suffix)
emb_len = len(model.get_output_matrix()[0])
%%time
# run fasttext model

# create file with individual sentence on each line
file = open(working_data_path + 'fasttext_model_' + source_column + '_by_sentence.txt', 'w', encoding='utf-8')
for txt in filter(None, corpus.values):
    file.write('\n'.join(re.split(split_sentence_by, txt)))
file.close()

# run model
model = fasttext.train_unsupervised(working_data_path + source_column + '.txt', 
                                    model = selected_m, 
                                    epoch = selected_epoch)
emb_len = len(model.get_output_matrix()[0])

model.save_model(working_data_path + 'fasttext_model_' + model_name_suffix)
# pre-calculate sentence embeddings for all sentences from covid19 by running command below (multiprocessing)
# bash: python sentemb.py

X = pickle.load(open(working_data_path + 'fasttext_model_' + search_name_suffix + '_X.pkl', 'rb'))
# create search

from IPython.display import display, Markdown, Latex

class Quicksearch:
    def __init__(self, modl, emb_len, sentences, sentence_embeddings, paper_lookup):
        self.modl = modl
        self.emb_len = emb_len
        self.sentences = sentences
        self.sentence_embeddings = sentence_embeddings
        self.paper_lookup = paper_lookup
    def sentence_to_vec(self, sent):
        words = self.modl.words
        emb = self.modl.get_output_matrix()
        vectors = []

        for token in sent.split():
            if token not in vectors:
                if token in words:
                    vectors.append(list(map(float, emb[words.index(token)])))
        if len(vectors) == 0:
            vectors.append(np.zeros(self.emb_len))
        return np.mean(vectors, axis=0)
    def get_candidate(self, i):
        sentences = list(self.sentences.keys())
        return np.array([i,sentences[i]])
    def get_candidate_ranking(self, sent):
        get_candidate_vec = np.vectorize(self.get_candidate, signature='()->(m)', otypes=[tuple])
        y = np.array([self.sentence_to_vec(sent)])
        scores = cdist(self.sentence_embeddings,y,'cosine').ravel()
        ranked_sentences = get_candidate_vec(np.argsort(scores))
        return ranked_sentences, scores
    def term(self, init, placeholder, description):
        return widgets.Textarea(value=init, 
                                placeholder=placeholder, 
                                description=description, 
                                layout=widgets.Layout(width='90%', display='flex'))
    def sort(self, init, options, description):
        return widgets.Dropdown(options=options,
                                  value=init,
                                  description=description, 
                                  layout=widgets.Layout(width='90%', display='flex'))
    def top(self, init, maxx, description):
        return widgets.IntSlider(min=1, 
                                 max=maxx, 
                                 value=init, 
                                 description=description, 
                                 layout=widgets.Layout(width='90%', display='flex'))
    def search(self, term, sort_by, show_top):
        if term == '':
            print('')
        else:
            term = term.lower()
            sent_rank, paper_rank, final_result = [], dict(), []
            
            # get ranking for search results
            ranked_sentences, scores = self.get_candidate_ranking(term)
            
            # for each sentence, record content, rank, order in paper
            # for each paper, record highest ranked sentence
            for i, [rank, sentence] in enumerate(ranked_sentences):
                if i < show_top:
                    r = dict()
                    r['rank'] = i + 1
                    r['sentence'] = sentence
                    r['paper id'] = self.sentences[sentence][0]
                    r['sentence_order'] = self.sentences[sentence][1]
                    sent_rank.append(r)
                    
                    #record highest ranking sentence
                    if self.sentences[sentence][0] not in paper_rank: 
                        paper_rank[self.sentences[sentence][0]] = i + 1
    
            # for each paper, lookup information on that paper
            for key, group in pd.DataFrame(sent_rank).groupby('paper id'):
                r = dict()
                r['rank'] = paper_rank[key]
                r['publish_time'] = self.paper_lookup[key]['publish_time']
                r['title'] = self.paper_lookup[key]['title']
                r['journal'] = self.paper_lookup[key]['journal']
                r['url'] = self.paper_lookup[key]['url']
                r['topic'] = self.paper_lookup[key]['topic']
                r['sentences'] = [sent for sent, order in sorted(zip(group['sentence'].values, group['sentence_order'].values), key=lambda r: r[1])]
                final_result.append(r)
            final_result = pd.DataFrame(final_result)
            
            # print search results
            if_ascend = False if sort_by == 'publish_time' else True
            
            print('Search Results for ' + '"' + term.upper() + '"')
            
            for k,r in final_result.sort_values(by=[sort_by], ascending=if_ascend).iterrows():
                r['title'] = '\033[1m' + r['title'] + '\033[0m'
                r['url'] = "" if r['url'] is None else r['url']
                r['journal'] = 'Unknown Journal' if r['journal'] is None else r['journal']
                r['publish_time'] = '' if pd.isnull(r['publish_time']) else datetime.strftime(r['publish_time'], '%Y-%m-%d')
                r['sentences'] = '...'.join(r['sentences'])
                
                print('<ul style="list-style:none; padding-left:0px;">')
                print('<li><b><a href="' + r['url'] + '">' + r['title'] + '</a></b></li>')
                print('<li><i>' + r['journal'] + '</i></li>')
                print('<li>' + r['publish_time'] + '</li>')
                print('<li>"' + r['sentences'] + '"</li>')
                print('</ul>')
                print('\n')
# define quicksearch
quicksearch = Quicksearch(model, emb_len, sents_in_paper, X, paper_lookup)

# set up init options
init_show = 10
init_max = 100
init_sort = 'publish_time'
init_search = 'smoking, pre-existing pulmonary disease'
init_options = {'Most Recent': 'publish_time', 'Most Similar': 'rank'}

# set up widget
term = quicksearch.term(init=init_search, placeholder='', description='Search: ')
sort_by = quicksearch.sort(init=init_sort, options=init_options, description='Sort By: ')
show_top = quicksearch.top(init=init_show, maxx=init_max, description='Filter # of Sentences to Show: ')
show_top.style.handle_color='darkred'
term.style.description_width = '100px'
sort_by.style.description_width = '100px'
show_top.style.description_width = '180px'

search = widgets.interactive(quicksearch.search, 
                             {"manual": True},
                             term = term, 
                             sort_by = sort_by, 
                             show_top = show_top)