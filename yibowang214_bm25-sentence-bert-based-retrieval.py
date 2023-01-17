

# # Extract meaningful fields and clean the data.

# import json



# import os



# import re

# import string

# from nltk.tokenize import word_tokenize

# from nltk.corpus import stopwords



# import pandas as pd

# import math



# stop_words = stopwords.words('english')



# path = "/kaggle/input/CORD-19-research-challenge"





# # files = os.listdir(path)





# def clean(content):

#     # delete URL

#     results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)

#     content = results.sub("", content)

#     content = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%|-)*\b', '', content, flags=re.MULTILINE)



#     content = content.lower()

#     # delete stopwords

#     tokens = word_tokenize(content)

#     content = [i for i in tokens if not i in stop_words]

#     s = ' '

#     content = s.join(content)



#     # delete punctuations, but keep '-'

#     del_estr = string.punctuation

#     del_estr = list(del_estr)

#     del_estr.remove('-')

#     del_estr = ''.join(del_estr)

#     replace = " " * len(del_estr)

#     tran_tab = str.maketrans(del_estr, replace)

#     content = content.translate(tran_tab)



#     return content





# data = pd.read_csv(path + '/metadata.csv')

# print(data.head(5))

# print(len(data))

# print(data['pdf_json_files'][15])



# i = 0



# ppath = []

# for file in range(len(data)):

#     s = []

#     # file = 61298

#     if not isinstance(data['pdf_json_files'][file], float):

#         # print(data['pdf_json_files'][file])

#         pdf_path = data['pdf_json_files'][file].split('; ')

#         filepath = path + "/" + pdf_path[0]

#     elif not isinstance(data['pmc_json_files'][file], float):

#         # print(data['pmc_json_files'][file])

#         pmc_path = data['pmc_json_files'][file].split('; ')

#         filepath = path + "/" + pmc_path[0]

#     else:

#         continue



#     with open(filepath, 'r', encoding='utf-8') as f:

#         temp = json.loads(f.read())

#         # print(temp.keys())



#         contents = []

#         file_path = data['cord_uid'][file]

#         if file_path in ppath:

#             print(file_path)

#             continue

#         else:

#             contents.append(file_path)

#             contents.append('\n')

#             ppath.append(file_path)



#         # print(file_path)



#         metadata_dict = temp['metadata']

#         metadata = []

#         if 'title' in metadata_dict.keys():

#             metadata.append(metadata_dict['title'])



#         abstract = []

#         if 'abstract' in temp.keys():

#             abstract_list = temp['abstract']

#             for content in abstract_list:

#                 # print(content.keys())

#                 if 'text' in content.keys():

#                     abstract.append(content['text'])



#         body_text_list = temp['body_text']

#         body_text = []

#         for content in body_text_list:

#             # print(content.keys())

#             if 'text' in content.keys():

#                 body_text.append(content['text'])

#         #

#         # print(metadata)

#         # print("___________________")

#         # print(body_text)

#         # print("+++++++++++++++++++")



#         contents.append(filepath)

#         contents.append('\n')



#         link = []

#         if not isinstance(data['url'][file], float):

#             link = data['url'][file].replace(' ', '')

#             contents.append(link)

#             # print(link)

#         contents.append('\n')



#         journal = []

#         if not isinstance(data['journal'][file], float):

#             journal = data['journal'][file]

#             contents.append(journal)

#         contents.append('\n')



#         date = []

#         if not isinstance(data['publish_time'][file], float):

#             date = data['publish_time'][file]

#             contents.append(date)

#         contents.append('\n')



#         metadata = str(metadata)

#         metadata = clean(metadata)

#         contents.append(metadata)



#         abstract = str(abstract)

#         abstract = clean(abstract)

#         contents.append(abstract)



#         body_text = str(body_text)

#         body_text = clean(body_text)

#         contents.append(body_text)



#         # print(contents)

#         #

#         f1 = open(path+"/extract/%d.txt" % (i + 1), 'w', encoding='utf-8')

#         # print(contents)

#         contents = "".join(contents)

#         f1.write(contents)

#         # print(i, contents)

#         # print("!!!!!!!!!!!!!!!!!")

#         i += 1

# # Collect useful information of the top-1000 documents to build a csv file.

# import json

# import pandas as pd

# import csv



# query = []

# with open('/kaggle/input/cord19round2queries/queries2.txt', 'r', encoding='utf-8') as f3:

#     for line in f3:

#         # line = line.split(' ')

#         line = line.strip('\n')

#         query.append(line[2:])

# # print(query)



# docid = []

# docpath = []

# url = []

# journal = []

# date = []

# with open('/kaggle/input/cord19round2luceneresult/round2_BM25_1000.txt', 'r', encoding='utf-8') as f1:

#     # i = 0

#     # while i < len(f1):

#     #     print(f1[i])

#     for line in f1:

#         line = line.split(' ')

#         # print(line)

#         docid.append(line[4])

#         docpath.append(line[6])

#         url.append(line[7])

#         journal.append(line[8])

#         date.append(line[9])



# with open("/round2_BM25_1000_full.csv", "a") as csvfile:

#     writer_BM25 = csv.writer(csvfile)

#     writer_BM25.writerow(["query", "rank", "paper_id", "title", "abstract", "contents", 'url', 'journal', 'date'])



#     for i in range(9):

#         k = 0



#         query_curr = query[i]

#         rank_curr = k

#         for file in docpath[1000*i: 1000*i+1000]:

#             # print(file)

#             k += 1

#             insert = []

#             insert.append(query_curr)

#             insert.append(k)

#             with open(file, 'r', encoding='utf-8') as f:

#                 temp = json.loads(f.read())



#                 # paper_id_curr = temp['paper_id']

#                 # insert.append(paper_id_curr)

#                 insert.append(docid[1000*i+k-1])



#                 metadata_dict = temp['metadata']



#                 if 'title' in metadata_dict.keys():

#                     title = metadata_dict['title']

#                 insert.append(title)



#                 if 'abstract' in temp.keys():

#                     abstract_list = temp['abstract']

#                     abstract = []

#                     for content in abstract_list:

#                         if 'text' in content.keys():

#                             abstract.append(content['text'])

#                 insert.append(' '.join(abstract))



#                 if 'body_text' in temp.keys():

#                     contents_list = temp['body_text']

#                     contents = []

#                     for content in contents_list:

#                         if 'text' in content.keys():

#                             contents.append(content['text'])

#                 insert.append(' '.join(contents))



#                 if len(url[1000*i+k-1]) == 1:

#                     insert.append(' ')

#                 else:

#                     insert.append(url[1000*i+k-1][1:])

#                 if len(journal[1000*i+k-1]) == 1:

#                     insert.append(' ')

#                 else:

#                     insert.append(journal[1000*i+k-1][1:])

#                 if len(date[1000*i+k-1]) == 1:

#                     insert.append(' ')

#                 else:

#                     insert.append(date[1000*i+k-1][1:])



#             # print([insert])

#             writer_BM25.writerows([insert])
# Test GPU.



import tensorflow as tf



device_name = tf.test.gpu_device_name()



if device_name == '/device:GPU:0':

    print('Found GPU at: {}'.format(device_name))

else:

    raise SystemError('GPU device not found')



import torch



# If there's a GPU available...

if torch.cuda.is_available():    



    # Tell PyTorch to use the GPU.    

    device = torch.device("cuda")



    print('There are %d GPU(s) available.' % torch.cuda.device_count())



    print('We will use the GPU:', torch.cuda.get_device_name(0))



# If not...

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
# # Use bert to rerank.



# import sys



# !pip install -U sentence-transformers



# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')



# import pandas as pd

# import numpy as np

# np.set_printoptions(threshold=sys.maxsize)

# pd.set_option('display.width',None)



# df = pd.read_csv("/kaggle/input/cord19round2prepareforbert/round2_BM25_1000_full.csv")

# # print(df.head(5))

# df['title'] = df['title'].astype(str)

# df['encoding'] =""

# for rows,index in df.iterrows():

#     title = index['title']

#     print(title)

#     search_phrase_vector = model.encode([title])[0]

#     # print(search_phrase_vector)

#     df.at[rows,'encoding'] = search_phrase_vector

#     # print(df.loc[rows])

# # df.to_csv('bert_encodings.csv')



# query = []

# with open('/kaggle/input/covid19round2queries/queries2.txt', 'r', encoding='utf-8') as f1:

#     for line in f1:

#         # line = line.split(' ')

#         line = line.strip('\n')

#         query.append(line[2:]) 

        

        

# from sklearn.metrics.pairwise import cosine_similarity

# import csv



# with open("result_queries2_5_full.csv", "a") as csvfile:

#   writer = csv.writer(csvfile)

#   writer.writerow(["query", "rank", "paper_id", "title", "abstract", "contents", "url", "journal", "date", "value"])

#   for i in range(9):

#     query_en = model.encode([query[i]])[0]

#     query_en = query_en.reshape(-1,1024)

    

#     result_cur = []

#     for j in range(i*1000, i*1000+1000):

#         row = df.loc[j]

#         doc = row['encoding']

#         doc = doc.reshape(-1,1024)

#         value = cosine_similarity(doc, query_en)

#         query_cur = query[i]

#         paper_id = row['paper_id']

#         result_cur.append((paper_id, value))

#     result_cur = sorted(result_cur, key=lambda x:x[1], reverse=True)

#     print(result_cur[:100])



#     t = 0

#     for k in range(5):

#         for j in range(i*1000, i*1000+1000):

#             if df.loc[j]['paper_id'] == result_cur[k][0]:

#                 title_cur = df.loc[j]['title']

#                 abstract_cur = df.loc[j]['abstract']

#                 contents_cur = df.loc[j]['contents']

#                 encoding_cur = df.loc[j]['encoding']

#                 url_cur = df.loc[j]['url']

#                 journal_cur = df.loc[j]['journal']

#                 date_cur = df.loc[j]['date']

#                 break

#         t += 1

#         result = []

#         result.append(query_cur)

#         result.append(str(t))

#         result.append(result_cur[k][0])

#         result.append(title_cur)

#         result.append(abstract_cur)

#         result.append(contents_cur)

#         result.append(url_cur)

#         result.append(journal_cur)

#         result.append(date_cur)

#         result.append(str(result_cur[k][1][0][0]))

# #         print(result)

#         writer.writerows([result])

import pandas as pd



df = pd.read_csv('/kaggle/input/cord19round2bertresult/result_queries2_5_full.csv')

print(df.columns)

print(len(df))

# date; sentence; url; journal; method; result;    measure of evidence

# date; study   ; url; journal;       ; summerize; 
import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')



from nltk import word_tokenize, pos_tag

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer
import re

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import pandas as pd



stop_words = stopwords.words('english')



def clean(content):



    def get_wordnet_pos(tag):

        if tag.startswith('J'):

            return wordnet.ADJ

        elif tag.startswith('V'):

            return wordnet.VERB

        elif tag.startswith('N'):

            return wordnet.NOUN

        elif tag.startswith('R'):

            return wordnet.ADV

        else:

            return None



    if pd.isnull(content):

        return ''



    # delete URL

    results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)

    content = results.sub("", content)

    content = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%|-)*\b', '', content, flags=re.MULTILINE)



    content = content.lower()

    # delete stopwords

    tokens = word_tokenize(content)

    content = [i for i in tokens if not i in stop_words]

    # s = ' '

    # content = s.join(content)



    # lemmatization

    tagged_sent = pos_tag(content)

    wnl = WordNetLemmatizer()

    lemmas_sent = []

    for tag in tagged_sent:

        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN

        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    # print(lemmas_sent)

    s = ' '

    content = s.join(lemmas_sent)



    # delete punctuations, but keep '-'

    del_estr = string.punctuation

    del_estr = list(del_estr)

    del_estr.remove('-')

    del_estr = ''.join(del_estr)

    replace = " " * len(del_estr)

    tran_tab = str.maketrans(del_estr, replace)

    content = content.translate(tran_tab)



    return content
stopwords = nltk.corpus.stopwords.words('english')



study = []

for i in range(len(df)):

    title = df['title'][i]

    if not isinstance(title, float):

        study.append(title)

    else:

        print("=====")

        word_frequencies = {}

        text_original = ''

        if not pd.isnull(df['title'][i]):

            text_original = text_original + df['title'][i] + ' '

        if not pd.isnull(df['abstract'][i]):

            text_original = text_original + df['abstract'][i] + ' '

        if not pd.isnull(df['contents'][i]):

            text_original = text_original + df['contents'][i]

        text = clean(text_original)



        for word in nltk.word_tokenize(text):

            if word not in stopwords:

                if word not in word_frequencies.keys():

                    word_frequencies[word] = 1

                else:

                    word_frequencies[word] += 1



        maximum_frequncy = max(word_frequencies.values())



        for word in word_frequencies.keys():

            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)



        text_cur = df['contents'][i]

        text_cur = text_cur.replace('e.g.', 'eg')

        text_cur = text_cur.replace('et al.', 'et al')

        sentence_list = nltk.sent_tokenize(text_cur)[:10]



        sentence_scores = {}

        for sent in sentence_list:

            for word in nltk.word_tokenize(sent.lower()):

                if word in word_frequencies.keys():

                    if len(sent.split(' ')) < 30:

                        if sent not in sentence_scores.keys():

                            sentence_scores[sent] = word_frequencies[word]

                        else:

                            sentence_scores[sent] += word_frequencies[word]



        study_sentences = heapq.nlargest(1, sentence_scores, key=sentence_scores.get)



        studying = ' '.join(study_sentences)

        study.append(studying)

        # print(study)
df['study'] = study

print(df.columns)
import heapq



stopwords = nltk.corpus.stopwords.words('english')



methods = []

for i in range(len(df)):

    word_frequencies = {}

    text_original = ''

    if not pd.isnull(df['title'][i]):

        text_original = text_original + df['title'][i] + ' '

    if not pd.isnull(df['abstract'][i]):

        text_original = text_original + df['abstract'][i] + ' '

    if not pd.isnull(df['contents'][i]):

        text_original = text_original + df['contents'][i]

    text = clean(text_original)



    # print(i, text[:10])



    for word in nltk.word_tokenize(text):

        if word not in stopwords:

            if word not in word_frequencies.keys():

                word_frequencies[word] = 1

            else:

                word_frequencies[word] += 1



    maximum_frequncy = max(word_frequencies.values())



    for word in word_frequencies.keys():

        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)



    text_cur = ' '

    if not pd.isnull(df['title'][i]):

        text_cur = text_cur + df['title'][i] + ' '

    if not pd.isnull(df['abstract'][i]):

        text_cur = text_cur + df['abstract'][i] + ' '

    text_cur = text_cur.replace('e.g.', 'eg')

    text_cur = text_cur.replace('et al.', 'et al')

    sentence_list = nltk.sent_tokenize(text_cur)

    # print(sentence_list)



    sentence_scores = {}

    for sent in sentence_list:

        for word in nltk.word_tokenize(sent.lower()):

            if word in word_frequencies.keys():

                if len(sent.split(' ')) < 30:

                    if sent not in sentence_scores.keys():

                        sentence_scores[sent] = word_frequencies[word]

                    else:

                        sentence_scores[sent] += word_frequencies[word]



    method_sentence = heapq.nlargest(1, sentence_scores, key=sentence_scores.get)



    method = ' '.join(method_sentence)

    methods.append(method)
# print(methods)

print(len(methods))
!wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

!unzip stanford-corenlp-full-2018-02-27.zip

!cd stanford-corenlp-full-2018-02-27



!echo "Downloading CoreNLP..."

!wget "http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip" -O corenlp.zip

!unzip corenlp.zip

!mv ./stanford-corenlp-4.0.0 ./corenlp
!pip install stanza



import stanza





import os

os.environ["CORENLP_HOME"] = "./corenlp"



from stanza.server import CoreNLPClient



client = CoreNLPClient(annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'], memory='4G', endpoint='http://localhost:9001')

print(client)



client.start()

import time; time.sleep(10)



!pip install pycorenlp
import nltk

from pycorenlp import *

import collections



subjects = []



for sentence in methods:

    if len(sentence) == 0:

        subjects.append(' ')

        continue

    doc = client.annotate(sentence, properties={"annotators":"tokenize,ssplit,pos,depparse,natlog,openie",

                                                "outputFormat": "json",

                                                "triple.strict":"true"

                                                #  "openie.triple.strict":"true",

                                                # "openie.max_entailments_per_clause":"2"

                                                })

    result = [doc["sentences"][0]["openie"] for item in doc]



    length1 = 0

    length2 = 0

    subject = ''

    object_ = ''

    for i in result:

        # print(i)

        for rel in i:

            relationSent1=rel['subject']

            if len(relationSent1) > length1:

                length1 = len(relationSent1)

                subject = relationSent1

#             relationSent2=rel['object']

#             if len(relationSent2) > length2:

#                 length2 = len(relationSent2)

#                 object_ = relationSent2

#         if length2 > length1:

#             subjects.append(object_)

#         else:

#             subjects.append(subject)

        if len(subject)>10:

            subjects.append(subject)

        else:

            subjects.append(' ')

#     print(subjects)
subjects = pd.Series(subjects)

# print(subjects)

print(len(subjects))

df["method"] = subjects

# print(df.head(5))

print(df.columns)
import heapq



stopwords = nltk.corpus.stopwords.words('english')



summerize = []

for i in range(len(df)):

    word_frequencies = {}

    text_original = ''

    if not pd.isnull(df['title'][i]):

        text_original = text_original + df['title'][i] + ' '

    if not pd.isnull(df['abstract'][i]):

        text_original = text_original + df['abstract'][i] + ' '

    if not pd.isnull(df['contents'][i]):

        text_original = text_original + df['contents'][i]

    text = clean(text_original)



    # print(i, text[:10])



    for word in nltk.word_tokenize(text):

        if word not in stopwords:

            if word not in word_frequencies.keys():

                word_frequencies[word] = 1

            else:

                word_frequencies[word] += 1

#     print(len(word_frequencies))



    maximum_frequncy = max(word_frequencies.values())

#     print(maximum_frequncy)

#     print(max(word_frequencies,key=word_frequencies.get))



    for word in word_frequencies.keys():

        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)



    text_cur = text_original.replace('e.g.', 'eg')

    text_cur = text_cur.replace('et al.', 'et al')

    sentence_list = nltk.sent_tokenize(text_cur)

#     print(sentence_list)



    sentence_scores = {}

    for sent in sentence_list:

        for word in nltk.word_tokenize(sent.lower()):

            if word in word_frequencies.keys():

                if len(sent.split(' ')) < 30:

                    if sent not in sentence_scores.keys():

                        sentence_scores[sent] = word_frequencies[word]

                    else:

                        sentence_scores[sent] += word_frequencies[word]



    summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)

#     print(len(summary_sentences))



    summary = ' '.join(summary_sentences)

    summerize.append(summary)

#     print(summary)
# print(summerize[:2])

print(len(summerize))

df['result'] = summerize

print(df.columns)
evidence = []

for i in range(len(df)):

    word_frequencies = {}

    text_original = ''

    if not pd.isnull(df['title'][i]):

        text_original = text_original + df['title'][i] + ' '

    if not pd.isnull(df['abstract'][i]):

        text_original = text_original + df['abstract'][i] + ' '

    if not pd.isnull(df['contents'][i]):

        text_original = text_original + df['contents'][i]

    text = clean(text_original)



    text = text.split()

    num = 0

    evidence_cur = ' '

    for i in range(len(text)):

        if text[i] == 'patient' and text[i-1].isdigit() and text[i-2] != 'fig' and text[i-2] != 'figure' and text[i-2] != 'table':

            num = max(int(text[i-1]), num)

    if num != 0:

        evidence_cur += str(num)

        evidence_cur += 'patients'

#     print(evidence_cur)

    evidence.append(evidence_cur)

print(len(evidence))
df['measure of evidence'] = evidence

print(df.columns)
import csv

from pandas import Series,DataFrame

import pandas as pd

import re



data = {

#         'Query':df['query'],

#         'Rank':df['rank'],

        "Date":df['date'],

       "Study":df['study'],

       "Study Link":df['url'],

       "Journal":df['journal'],

       "Method":df['method'], 

        "Result":df['result'], 

        "Measure of Evidence":df['measure of evidence']}

data = DataFrame(data)

print(data.columns)

print(data.head(5))

# print(len(data))



# data.to_csv('results.csv', index = False)



for i in range(9):

    current = data[i*5:i*5+5]

    csv_str = df['query'][i*5]

    csv_str = re.sub(r'[^\w\s]','',csv_str) + '.csv'

#     print(csv_str)

    current.to_csv( csv_str, index = [0,1,2,3,4])
ls