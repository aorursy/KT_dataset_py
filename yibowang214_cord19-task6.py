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



# for file in range(len(data)):

#     s = []

#     if not isinstance(data['pdf_json_files'][file], float):

#         print(data['pdf_json_files'][file])

#         pdf_path = data['pdf_json_files'][file].split('; ')

#         filepath = path + "/" + pdf_path[0]

#     elif not isinstance(data['pmc_json_files'][file], float):

#         print(data['pmc_json_files'][file])

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



# #         contents = []



# #         file_path = data['cord_uid'][file]

# #         contents.append(file_path)

# #         contents.append('\n')



#         contents.append(filepath)

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

#         # f1 = open(path+"/extract_3/%d.txt" % (i + 1), 'w', encoding='utf-8')

#         # contents = "".join(contents)

#         # f1.write(contents)

#         print(contents)

#         print("!!!!!!!!!!!!!!!!!")

#         i += 1

# # Create csv files.

# import json

# import pandas as pd

# import csv



# BM25 = []

# paper_num_BM25 = []

# with open('lucene_result_BM25.txt', 'r', encoding='utf-8') as f1:

#     for line in f1:

#         # print(line)

#         line = line.split(' ')

#         BM25.append(line[5])

#         paper_num_BM25.append(line[4])



# RM3 = []

# paper_num_RM3 = []

# with open('lucene_result_RM3.txt', 'r', encoding='utf-8') as f2:

#     for line in f2:

#         # print(line)

#         line = line.split(' ')

#         RM3.append(line[6])

#         paper_num_RM3.append(line[4])



# query = []

# with open('/kaggle/input/covid19task6queries/queries.txt', 'r', encoding='utf-8') as f3:

#     for line in f3:

#         # line = line.split(' ')

#         line = line.strip('\n')

#         query.append(line[2:])

# # print(query)



# with open("/BM25.csv", "a") as csvfile:

#     writer_BM25 = csv.writer(csvfile)

#     writer_BM25.writerow(["query", "rank", "paper_id", "title", "abstract", "contents"])



#     for i in range(11):

#         k = 0



#         query_curr = query[i]

#         rank_curr = k

#         for file in BM25[20*i: 20*i+20]:

#             # print(file)

#             k += 1

#             insert = []

#             insert.append(query_curr)

#             insert.append(k)

#             with open(file, 'r', encoding='utf-8') as f:

#                 temp = json.loads(f.read())



# #                 paper_id_curr = temp['paper_id']

#                 paper_id_curr = paper_num_BM25[20*i+k-1]

#                 insert.append(paper_id_curr)



#                 metadata_dict = temp['metadata']



#                 if 'title' in metadata_dict.keys():

#                     title = metadata_dict['title']

#                 insert.append(title)



#                 abstract = []

#                 if 'abstract' in temp.keys():

#                     abstract_list = temp['abstract']

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



#             # print([insert])

#             writer_BM25.writerows([insert])





# with open("/RM3.csv", "a") as csvfile:

#     writer_RM3 = csv.writer(csvfile)

#     writer_RM3.writerow(["query", "rank", "paper_id", "title", "abstrct", "contents"])

#     for i in range(11):

#         k = 0

#         # writer.writerows([[0,1,3],[1,2,3],[2,3,4]])

#         query_curr = query[i]

#         rank_curr = k

#         for file in RM3[20*i: 20*i+20]:

#             # print(file)

#             k += 1

#             insert = []

#             insert.append(query_curr)

#             insert.append(k)

#             with open(file, 'r', encoding='utf-8') as f:

#                 temp = json.loads(f.read())



# #                 paper_id_curr = temp['paper_id']

#                 paper_id_curr = paper_num_RM3[20*i+k-1]

#                 insert.append(paper_id_curr)



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



#             # print([insert])

#             writer_RM3.writerows([insert])





# with open("/intersection.csv", "a") as csvfile:

#     writer_inter = csv.writer(csvfile)

#     writer_inter.writerow(["query", "rank", "paper_id", "title", "abstract", "contents"])

#     for i in range(11):

#         k = 0

#         # writer.writerows([[0,1,3],[1,2,3],[2,3,4]])

#         query_curr = query[i]

#         rank_curr = k

#         a = BM25[20*i: 20*i+20]

#         b = RM3[20*i: 20*i+20]

#         intersection = list(set(a).intersection(set(b)))

#         c = paper_num_BM25[20*i: 20*i+20]

#         d = paper_num_RM3[20*i: 20*i+20]

#         num_intersection = list(set(c).intersection(set(d)))

#         print(len(intersection))

#         for file in intersection[0:10]:

#             k += 1

#             # insert = [query_curr, k]

#             insert = []

#             insert.append(query_curr)

#             insert.append(k)

#             with open(file, 'r', encoding='utf-8') as f:

#                 temp = json.loads(f.read())



# #                 paper_id_curr = temp['paper_id']

#                 paper_id_curr = num_intersection[20*i+k-1]

#                 insert.append(paper_id_curr)



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



#             # print([insert])

#             writer_inter.writerows([insert])







# # for i in range(11):

# #     f3 = open(

# #         "/query/query%d.txt" % (

# #                     i+1), 'a', encoding='utf-8')

# #     a = BM25[20*i: 20*i+20]

# #     b = RM3[20*i: 20*i+20]

# #     intersection = list(set(a).intersection(set(b)))

# #     for file in intersection[0:3]:

# #         with open(file, 'r', encoding='utf-8') as f:

# #             temp = json.loads(f.read())

# #

# #             paper_id = temp['paper_id']

# #             # print(paper_id)

# #             # f3.write(paper_id+'\n')

# #

# #             metadata_dict = temp['metadata']

# #             if 'title' in metadata_dict.keys():

# #                 title = metadata_dict['title']

# #             # print(type(title))

# #             # print(title)

# #             # f3.write(title+'\n')

# #

# #             if 'abstract' in temp.keys():

# #                 abstract_list = temp['abstract']

# #                 abstract = []

# #                 for content in abstract_list:

# #                     if 'text' in content.keys():

# #                         abstract.append(content['text'])

# #                 # f3.write(str(abstract)+'\n')

# #                 # print(str(abstract))

# #             else:

# #                 # f3.write('None\n')

# #                 # print('None')
# # Collect useful information of the top-1000 documents to build a csv file.



# BM25 = []

# paper_num_BM25 = []

# with open('lucene_BM25_1000.txt', 'r', encoding='utf-8') as f1:

#     for line in f1:

#         # print(line)

#         line = line.split(' ')

#         BM25.append(line[5])

#         paper_num_BM25.append(line[4])

        

# with open("/BM25_1000.csv", "a") as csvfile:

#     writer_BM25 = csv.writer(csvfile)

#     writer_BM25.writerow(["query", "rank", "paper_id", "title", "abstract", "contents"])



#     for i in range(11):

#         k = 0



#         query_curr = query[i]

#         rank_curr = k

#         for file in BM25[1000*i: 1000*i+1000]:

#             # print(file)

#             k += 1

#             insert = []

#             insert.append(query_curr)

#             insert.append(k)

#             with open(file, 'r', encoding='utf-8') as f:

#                 temp = json.loads(f.read())



# #                 paper_id_curr = temp['paper_id']

                

#                 paper_id_curr = paper_num_BM25[1000*i+k-1]

#                 insert.append(paper_id_curr)



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



# !pip install -U sentence-transformers



# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')



# import pandas as pd

# import numpy as np

# np.set_printoptions(threshold=sys.maxsize)

# pd.set_option('display.width',None)



# df = pd.read_csv("/kaggle/input/preparationforbert/BM25_1000.csv")

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

# with open('/kaggle/input/covid19task6queries/queries.txt', 'r', encoding='utf-8') as f1:

#     for line in f1:

#         # line = line.split(' ')

#         line = line.strip('\n')

#         query.append(line[2:]) 

        

# from sklearn.metrics.pairwise import cosine_similarity

# import csv



# with open("result.csv", "a") as csvfile:

    

#     writer = csv.writer(csvfile)

#     writer.writerow(["query", "rank", "paper_id", "title", "abstract", "contents", "value"])

#     for i in range(11):

#         query_en = model.encode([query[i]])[0]

#         query_en = query_en.reshape(-1,1024)

#         print(query[i])



#         result_cur = []

#         for j in range(i*1000, i*1000+1000):

#             row = df.loc[j]

#             doc = row['encoding']

#             doc = doc.reshape(-1,1024)

#             value = cosine_similarity(doc, query_en)

#             query_cur = query[i]

#             paper_id = row['paper_id']

#             result_cur.append((paper_id, value))

#         print(result_cur[0])

#         result_cur = sorted(result_cur, key=lambda x:x[1], reverse=True)

#         print(result_cur[0:6])



#         t = 0

#         for k in range(20):

#             for j in range(i*1000, i*1000+1000):

#                 if df.loc[j]['paper_id'] == result_cur[k][0]:

#                     title_cur = df.loc[j]['title']

#                     abstract_cur = df.loc[j]['abstract']

#                     contents_cur = df.loc[j]['contents']

#             t += 1

#             result = []

#             result.append(query_cur)

#             result.append(str(t))

#             result.append(result_cur[k][0])

#             result.append(title_cur)

#             result.append(abstract_cur)

#             result.append(contents_cur)

#             result.append(str(result_cur[k][1][0][0]))

#             print(result)

#             writer.writerows([result])
import nltk

nltk.download('stopwords')

nltk.download('punkt')



import pandas as pd

df = pd.read_csv("/kaggle/input/bertresultstitleonly/result_queries_10.csv")
import re

import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import pandas as pd



stop_words = stopwords.words('english')



def clean(content):

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

    s = ' '

    content = s.join(content)



    # delete punctuations, but keep '-'

    del_estr = string.punctuation

    del_estr = list(del_estr)

    del_estr.remove('-')

    del_estr = ''.join(del_estr)

    replace = " " * len(del_estr)

    tran_tab = str.maketrans(del_estr, replace)

    content = content.translate(tran_tab)



    return content
import heapq

import nltk



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



    sentence_list = nltk.sent_tokenize(text_original.replace('e.g.', 'eg'))

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



    summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)



    summary = ' '.join(summary_sentences)

    summerize.append(summary)

#     print(summary)
print(df.head(5))

df.to_csv("summerize.csv",index=False)
import pandas as pd



data = pd.read_csv('/kaggle/input/bertresultstitleonly/result_queries_10.csv')

print(data.head(5))



contents = data['contents']

# print(contents[:5])
import nltk

nltk.download('punkt')



content_curr = nltk.sent_tokenize(contents[0].replace('e.g.', 'eg'))

# print(type(content_curr))

# content_curr = contents[0].replace('e.g.', 'eg').split('. ')

# print(content_curr)

print(len(content_curr))
!pip install --upgrade gap-stat

# !pip install gapkmean



import numpy as np

import pandas as pd

from sklearn.cluster import KMeans



def optimalK(data, nrefs=3, maxClusters=15):

    gaps = np.zeros((len(range(1, maxClusters)),))

    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})

    for gap_index, k in enumerate(range(1, maxClusters)):



        refDisps = np.zeros(nrefs)



        for i in range(nrefs):

            randomReference = np.random.random_sample(size=data.shape)

            

            km = KMeans(k)

            km.fit(randomReference)

            

            refDisp = km.inertia_

            refDisps[i] = refDisp



        km = KMeans(k)

        km.fit(data)

        

        origDisp = km.inertia_



        gap = np.log(np.mean(refDisps)) - np.log(origDisp)



        gaps[gap_index] = gap

        

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)



    return (gaps.argmax() + 1, resultsdf) 
from sklearn.metrics.pairwise import cosine_similarity

import csv

from sklearn.cluster import KMeans

import numpy as np

import pandas as pd



!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')





query = []

with open('/kaggle/input/covid19task6queries/queries.txt', 'r', encoding='utf-8') as f1:

    for line in f1:

        # line = line.split(' ')

        line = line.strip('\n')

        query.append(line[2:]) 





with open("result_sentence_knn_10.csv", "a") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(["query", "rank", "sentence", "value"])

    for i in range(11):

        query_en = model.encode([query[i]])[0]

        query_en = query_en.reshape(-1,1024)

        print(query[i])



        result_feature = []

        result_sentence = []

        result_value = []

        for j in range(10*i, 10*i+10):

            content_curr = nltk.sent_tokenize(contents[j].replace('e.g.', 'eg'))

            length = len(content_curr)

            for k in range(length):

                sentence = model.encode([content_curr[k]])[0]

                sentence = sentence.reshape(-1,1024)

                value = cosine_similarity(sentence, query_en)

                result_feature.append(sentence)

                result_sentence.append(content_curr[k])

                result_value.append(value)



        my_df = pd.DataFrame(data=result_sentence, columns=['sentence'])

        my_df['feature'] = result_feature

        my_df['value'] = result_value

        my_df = my_df.sort_values(by='value', ascending=False)

#         print(len(my_df))

        my_df = my_df.iloc[:50]

#         print(len(my_df))

#         print(my_df.head(5))



        # result_feature = np.asarray(my_df.feature.to_numpy()).reshape(-1,1024)

        # print(np.array(my_df.feature.to_list()).shape)

        bestKValue, gapdf = optimalK(np.array(my_df.feature.to_list()).reshape(-1,1024), nrefs=5, maxClusters=5)

#         print(bestKValue)

        



        km = KMeans(bestKValue)

        result = km.fit_predict(np.array(my_df.feature.to_list()).reshape(-1,1024))

        clusters = km.labels_.tolist()



        my_df['label'] = clusters



        mean_values = []

        for ii in range(bestKValue):

            a = np.mean(my_df[my_df.label==ii].value)

            mean_values.append(a)

        label = np.argmax(mean_values)



        sentences = my_df[my_df.label==label].sentence.tolist()

        values = my_df[my_df.label==label].value.tolist()



        query_sentence = query[i]

        for m in range(min(10, len(sentences))):

            input_contents = [query_sentence, str(m+1), sentences[m], values[m]]

            # print(input_contents)

            writer.writerows([input_contents])





        # query_sentence = query[i]

        # for m in range(10):

        #     input_contents = [query_sentence, str(m+1), result_sentence[m][0], str(result_sentence[m][1][0][0])]

        #     print(input_contents)

            # writer.writerows([input_contents])
import pandas as pd





df = pd.read_csv('result_sentence_knn_10.csv')

print(df.head(5))



sentences = df['sentence']

print(sentences[0])



print(df.columns)

# print(type(sentences))

# print(sentences)
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



relation = []



for sentence in sentences:

#     print(sentence)

    doc = client.annotate(sentence, properties={"annotators":"tokenize,ssplit,pos,depparse,natlog,openie",

                                                "outputFormat": "json",

                                                "triple.strict":"true"

                                                #  "openie.triple.strict":"true",

                                                # "openie.max_entailments_per_clause":"2"

                                                })

    result = [doc["sentences"][0]["openie"] for item in doc]



    extraction = []

    subjects = []

    for i in result:

        # print(i)

        for rel in i:

            relationSent=rel['subject'],rel['relation'],rel['object']

            if relationSent[0] not in subjects:

                subjects.append(relationSent[0])

                extraction.append(relationSent)

                # print(relationSent)

            # print(relationSent)

        # print("_____")

        # rel = i[-1]

        # relationSent=rel['subject'],rel['relation'],rel['object']

        # print(relationSent)

    # print(extraction)

    # print("____________")

    relation.append(extraction)
print(relation[:5])

relation = pd.Series(relation)

print(relation)

df["relation"] = relation

print(df.head(5))
df.to_csv("result_relation_knn_10.csv",index=False)