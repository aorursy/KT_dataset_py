!pip install spacy
!pip install scispacy
!pip install langdetect
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
# Python libraries

import pandas as pd

import scispacy

import spacy

from gensim.models import Word2Vec

from nltk import word_tokenize

import os

from string import punctuation

from nltk.corpus import stopwords

import numpy as np

import math

import json

from collections import OrderedDict

import re

import en_core_sci_lg

from tqdm.notebook import tqdm

from datetime import datetime

from gensim.models import KeyedVectors

import tarfile

from IPython.display import display, HTML

from langdetect import detect

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

import csv

import nltk

import string
# Helper function to read csv

def getKeywordLists(keywordFile, seperator):

    df = pd.read_csv(keywordFile,sep=seperator)

    return df
# source code to train Word2Vec using cbow model 

#nlp = en_core_sci_lg.load()

#new_lines = []

#total_word = []

#nlp.max_length = 3000000

def gettitlewords(data_df):

    print("getting title words...")

    for j in tqdm(range(len(data_df['title'])), total=len(data_df['title'])):

        item = str(data_df['title'][j])

        item = item.encode('ascii', 'ignore')

        item = item.decode('utf-8')

        if item != 'nan' and len(item) > 3:

            item = item.lower()

            words = word_tokenize(item)

            new_lines.append(words)

            for word in words:

                total_word.append(word)

                

    return total_word, new_lines



def getabstractwords(data_df):

    print("getting abstract words...")

    for l in tqdm(range(len(data_df['abstract'])), total=len(data_df['abstract'])):

        abstract = str(data_df['abstract'][l])

        if abstract != 'nan':

            doc = nlp(abstract)

            ab_sentences = list(doc.sents)

            for sent in ab_sentences:

                sent = str(sent)

                sent = sent.lower()

                sent = sent.encode('ascii','ignore')

                sent = sent.decode('utf-8')

                if len(sent) > 3:

                    words = word_tokenize(sent)

                    new_lines.append(words)

                    for word in words:

                        total_word.append(word)

                

    return total_word, new_lines



def gettextwords(data_df):

    print("getting text words...")

#    oov_word = []

    for i in tqdm(range(len(data_df['text'])), total=len(data_df['text'])):

        a = str(data_df['text'][i])

        if a != 'nan':

            doc = nlp(a)

            sentences = list(doc.sents)

            for sent in sentences:

                sent = str(sent)

                sent = sent.lower()

                sent = sent.encode('ascii','ignore')

                sent = sent.decode('utf-8')

                if len(sent) > 3:

                    words = word_tokenize(sent)

                    new_lines.append(words)

                    for word in words:

                        total_word.append(word)

    return total_word, new_lines

               

#data_df = getKeywordLists("/kaggle/output/covid19_dataset_new.csv", seperator='\t')

#total_word, new_lines = gettitlewords(data_df)

#total_word, new_lines = getabstractwords(data_df)

#total_word, new_lines = gettextwords(data_df)



#print('total new lines are : {}'.format(len(new_lines)))

#print('total words are : {}'.format(len(total_word)))

#model = Word2Vec(new_lines, min_count =1)

#print(model)

#model.save("word2vec_model_covid19.bin")
# function to extract .tar file

def extract(tar_file, path):

    opened_tar = tarfile.open(tar_file)

    if tarfile.is_tarfile(tar_file):

        opened_tar.extractall(path)

    else:

        print("the tar you entered is not a tar file")

        

extract("/kaggle/input/trained-word2vec-model/word2vec_model.tar.xz", "/kaggle/output/kaggle/working")
# function to check whether a given word is a number or not

def is_number(word):

    try:

        word = word.translate(str.maketrans('','',b))

        float(word)

    except ValueError:

        return False

    return True
# function to calculate L2 norm

def l2_norm(a):

    return math.sqrt(np.dot(a,a))



# function to Calculate cosine similarity

def cosine_similarity(a,b):

    return np.dot(a,b) / (l2_norm(a)*l2_norm(b))
# punctuations

b = '!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~'



# load nlp model

nlp = en_core_sci_lg.load()



# load word2vec model

model = KeyedVectors.load_word2vec_format("/kaggle/output/kaggle/working/word2vec_model.txt")

print('word2vec model loaded successfully')

word2vec_vocabulary = list(model.wv.vocab)



# load extended stopword list

new_file1 = open("/kaggle/input/extended-stopword-list/Extended_Stopwords.txt")

tfidf_stopwords = new_file1.readlines()

for i in range(len(tfidf_stopwords)):

    tfidf_stopwords[i] = tfidf_stopwords[i].replace("\n", "")

    

punctuation_list = list(punctuation)

punctuation_list.append('``')     

punctuation_list.append("''")     

punctuation_list.append("'s")

punctuation_list.append("n't")

new_stopwords = set(tfidf_stopwords + stopwords.words("english") + punctuation_list)

stop_words = set(stopwords.words("english"))

print('stopwords list successfully loaded')
# Helper function to filter stopwords from list of words

def remove_stopwords(words):

    for stopword in stop_words:

        if stopword in words:

            words = list(filter(lambda a: a != stopword, words))

    return words



# Helper function to create a list of list of clean words

def cleandocdata(description):

    for i in range(len(description)):

        #description[i] = description[i].encode('ascii','ignore')

        #description[i] = description[i].decode('utf-8')

        description[i] = word_tokenize(description[i])

        description[i] = [word.lower() for word in description[i]]

        description[i] = list(filter(lambda a: len(a) > 2, description[i]))

    # Remove all Stop words

    for j in range(len(description)):

        description[j] = remove_stopwords(description[j])    

    return description



# function to collect clean words from list of phrases

def gettingphrasewords(phrases):

    phrase_words = []

    for phrase in phrases:

        w1 = word_tokenize(phrase)

        for w in w1:

            w = w.lower()

            if w not in new_stopwords and w not in phrase_words:

                if w in word2vec_vocabulary:

                    phrase_words.append(w)

    return phrase_words



# function to calculate inverse document frequency(idf) of words

def calculate_idf(no_documents, no_documents_in_which_word_occured):

    if no_documents_in_which_word_occured != 0:

        idf = math.log1p(no_documents/(1 + no_documents_in_which_word_occured))

    else:

        idf = 1

    return idf



# function to create dictionary where keys are the phrase words and values are the idf of phrase words

def getting_phrase_word_dict_with_idf_value(description, phrase_words):

    idf_dict = dict()

    for iword in phrase_words:

        count = 0

        for text3 in description:

            if iword in text3:

                count+=1

        no_documents_in_which_word_occured = count

        if iword not in idf_dict:

            idf_dict[iword] = calculate_idf(len(description), no_documents_in_which_word_occured)    

    return idf_dict
# Helper function to get phrase embedding

def getphraseembedding(phrases):

    phrase_embedding = dict()

    for indicator in phrases:

        list1 = []

        ind_words = word_tokenize(indicator)

        for word in ind_words:

            word = word.lower()

            if word in indicator.lower() and word not in new_stopwords:

                if word in word2vec_vocabulary and not is_number(word):

                    list1.append(model[word])



        if indicator not in phrase_embedding:

            phrase_embedding[indicator] = np.mean(list1, axis =0)

        else:

            phrase_embedding[indicator] = 0 

    return phrase_embedding
ignoreList = [ 'SARS-CoV', 'MERS', 'H1N1', 'H5N1', 'H7N9', 'rhinovirus', 'RSV', 'respiratory syncytial virus', 'metapneumovirus', 'parainfluenza', 'SFTSV', 'OC43', 'SARS', 'MERS-CoV', 'SARS-COV', 'MERS-COV', 'SARS-nCoV', 'SARS-nCov', 'SARS-nCOV']

keepList = ['COVID-19', '2019-nCoV', 'SARS-CoV-2', 'Wuhan coronavirus', 'covid-19', 'covid19', 'covid -19', 'covid- 19', 'covid - 19', 'covid 19', 'SARS-CoV2', 'COVID-2019', 'COVID 2019', '2019n-CoV']



def calcNeighbour(sid, sentIdDict, max1):

    ns = []

    #dL = []

    lhs = 0

    rhs = max1

    row = sentIdDict[sid]

    paper = row.Cord_uid

    sent_id1 =  int(row.Sentence_id)

    if sid<5:

        lhs = 0

    else :

        lhs = sid-5

    if (sid + 5)>max1:

        rhs = max1

    else:

        rhs = sid+5

    for i in range(lhs, rhs+1):

        row = sentIdDict[i]

        paperN = row.Cord_uid

        sent_id2 = int(row.Sentence_id)

        if paper == paperN and abs(sent_id1-sent_id2) <= 5:

            ns.append(row.Sentence)

    containedDiseaseN = []

    for  s in ns:

        #print(sentence, disease)

        for idl in ignoreList:

            if idl in s and 'SARS-CoV-2'.lower() not in s.lower() and 'SARS-CoV2'.lower() not in s.lower() :

                containedDiseaseN.append(idl)

                #print(idl,'_____________',  s)

        #print(containedIrrevDisease)

        

        for kdl in keepList:

            if kdl.lower() in s.lower():

                containedDiseaseN.append(kdl)

                #print(kdl,'_____________', s)

    #print(len(ns), sid, rhs, lhs)

    containedDiseaseN = set(containedDiseaseN)

    #print(containedDiseaseN)

    if len(containedDiseaseN) == 0:

        x = []

        return x

    containedDiseaseN = list(containedDiseaseN)

    return containedDiseaseN





def extract_neighbouring_info(dfE):

    sentIdDict = dict()

    max1 = 0

    for row in dfE.itertuples(): 

        sid = int(row.Index)

        if sid > max1:

            max1 = sid

        sentIdDict[sid] = row



    #print(len(sentIdDict), max1)

    dictLen = len(sentIdDict)



    dfE['Rel_Disease'] = ''

    dfE['Irrel_Disease'] =  ''

    dfE['Neighboring_Sentence_Disease'] = ''

    for i in range(0,max1+1):

        row = sentIdDict[i]

        sentence = row.Sentence

        containedIrrevDisease = []

        for idl in ignoreList:

            if idl in sentence and 'SARS-CoV-2' not in sentence and 'SARS-CoV2'.lower() not in sentence.lower() :

                containedIrrevDisease.append(idl)

        #print(containedIrrevDisease)

        containedDisease = []

        for kdl in keepList:

            if kdl.lower() in sentence.lower():

                containedDisease.append(kdl)

        dfE.set_value(row.Index, 'Rel_Disease', containedDisease)

        dfE.set_value(row.Index, 'Irrel_Disease', containedIrrevDisease)

        containedDiseaseN = []

        containedDiseaseN = calcNeighbour(i, sentIdDict, max1)

        dfE.set_value(row.Index, 'Neighboring_Sentence_Disease', containedDiseaseN)

    return dfE
# helper function to generate list of unique words of sentences

def generate_list_non_repeating_words(non_dup_sent):

    list_of_words=[]

    for j in tqdm(range(len(non_dup_sent)), total=len(non_dup_sent)):

        words = word_tokenize(non_dup_sent[j])

        words = map(lambda x : x.lower(),words)

        list_of_words.extend(words)

    non_duplicate_list_of_words = list(set(list_of_words))       

    non_duplicate_list_of_words = filter(lambda x : x not in new_stopwords,non_duplicate_list_of_words)

    non_duplicate_list_of_words = filter(lambda x : x in word2vec_vocabulary,non_duplicate_list_of_words)

    non_duplicate_list_of_words = list(filter(lambda x : not is_number(x),non_duplicate_list_of_words))    

    return non_duplicate_list_of_words



# function to create word similarity dictionary

def generating_word_similarity_dictionary(non_dup_sent, phrase_words, phrase_embedding):

    similarity_dict = {}

    words_list = generate_list_non_repeating_words(non_dup_sent)

    for phrase in phrase_words:

        for s_word in words_list:

            if (phrase, s_word) not in similarity_dict.keys():

                similarity_dict[(phrase, s_word)] = round(cosine_similarity(phrase_embedding[phrase],model[s_word]),5)

    return similarity_dict



#print('Reading extracted sentences from filtered covid documents')

#tot_sent = list(sentence_df['Sentence'])

#non_dup_sent = list(set(tot_sent))

#print('Total sentences available after removing duplicates : ',len(non_dup_sent))

#phrases = ['covid-19 risk factors', 'hypertension covid-19', 'diabetes covid-19', 'heart disease covid-19', 'smoking covid-19', 'pulmonary disease covid-19', 'cancer covid-19', 'risk factors for neonates and pregnant women', 'respiratory disease covid-19', 'co-infections risk covid-19', 'incubation period covid-19', 'reproductive number covid-19', 'serial interval covid-19']

#phrase_single_words = gettingphrasewords(phrases)

#phrase_words = phrase_single_words + phrases

#print('total number of phrase_words avialable : ',len(phrase_words))

#phrase_embedding = getphraseembedding(phrase_words)

#print('phrase embedding generated successfully')

#similarity_dict = generating_word_similarity_dictionary(non_dup_sent, phrase_words, phrase_embedding)

#np.save('phrase_word_similarity_dictionary.npy', similarity_dict)

print('Reading word similarity score dictionary')

word_similarity_dictionary = np.load('/kaggle/input/kernel74dfc29773/phrase_word_similarity_dictionary.npy',allow_pickle='TRUE').item()

print('total length of dictionary is : ',len(word_similarity_dictionary))
# function to append list of words of a sentence

def adding_list_of_words(clean_sentence_df):

    non_dup_sent_words = []

    non_dup_sents = list(clean_sentence_df['Sentence'])

    for i in range(len(non_dup_sents)):

        #print(i)

        non_dup_sent_words.append(word_tokenize(non_dup_sents[i]))

    #print(len(non_dup_sent_words))

    clean_sentence_df['list_of_sent_words'] = non_dup_sent_words

    

    return clean_sentence_df
# function to get final similarity score between the phrase and a sentence

def phrasematching(indicator, idf_dict, clean_sentence_df1):

    all_score = []

    clean_keywords = []

    oov_word = []

    new_keywords = word_tokenize(indicator)

    if len(new_keywords) > 1:

        new_keywords.append(indicator)



    for m in range(len(new_keywords)):

        if new_keywords[m] not in new_stopwords:

            clean_keywords.append(new_keywords[m])

    #    match_phrase = []

    keyword_embedding = getphraseembedding(clean_keywords)

    non_dup_sents = list(clean_sentence_df1['Sentence'])

    for j in tqdm(range(len(non_dup_sents)), total=len(non_dup_sents)):

        sent = non_dup_sents[j]

        sent_words = clean_sentence_df1['list_of_sent_words'][j]

        sent_score = []

        for phrase in clean_keywords:

            list3 = []

            for s_word in sent_words:

                s_word = s_word.lower()

                if (phrase, s_word) in word_similarity_dictionary:

                    list3.append(word_similarity_dictionary[(phrase, s_word)])

                else:

                    oov_word.append(s_word)



            if list3 != []:

                sent_score.append(np.max(list3))

            else:

                sent_score.append(0)

        if len(sent_score) > 1:

            for k in range(len(sent_score)-1):

                if sent_score[k] >= 0.50:

                    sent_score[k] = idf_dict[clean_keywords[k]]*sent_score[k]



            sentence_score = np.mean(sent_score)

        else:

            sentence_score = np.mean(sent_score)

            

        all_score.append(sentence_score)



    clean_sentence_df1['sentence_score'] = all_score

    clean_sentence_df1 = clean_sentence_df1.sort_values(by=['sentence_score'], ascending = [False])

    clean_sentence_df1 = clean_sentence_df1[clean_sentence_df1.sentence_score > 0.7]

    return clean_sentence_df1
def word_tokenizer(text):

            tokens = word_tokenize(text)

            tokens = [t for t in tokens if t not in stopwords.words('english')]

            return tokens



#function to calculate idf matrix on abstract of documents

def tfidf(sentences):

            tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,

                                            stop_words=stopwords.words('english'),

                                            lowercase=True)

            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).todense()

            tfidf = tfidf_vectorizer.idf_

            dic = dict(zip(tfidf_vectorizer.get_feature_names(), tfidf))

            #print(dic)

            return dic



def get_tfidf_dict(filtered_covid_document_df):

    sentences=[]

    count=0

    #df1 = pd.read_csv("/home/hduser1/Desktop/COVID/Tushar/Kaggle_uploaded_files/tested_covid_documents.csv", sep='\t')

    for i in range(len(filtered_covid_document_df['abstract'])):

        sent = filtered_covid_document_df['abstract'][i]

        count+=1

        #print(count)

        sent = sent.replace("."," ") 

        sentences.append(sent)    

    dict_tfidf = tfidf(sentences)

    dict_tfidf_new={}

    for i in dict_tfidf:

        dict_tfidf_new[i]=math.log10(dict_tfidf[i])

    return dict_tfidf_new
myDict={}



myDict["hypertension covid-19"]=['hypertension', 'blood pressure', 'HTN', 'HBP']

myDict["diabetes covid-19"]=['insulin','insulin-dependent','insulin resistance','glucose control','blood glucose level','metaformin', 'hemoglobin A1c','hyperglycemia','hypoglycemia', 'hyperglycemic', 'hypoglycemic', 'pre-diabetes', 'mellitus']

myDict["heart disease covid-19"]=['cardiac', 'cardiovascular', 'ventricular', 'cardiopulmonary', 'valvular', 'systolic', 'coronary', 'cardiorespiratory']

myDict["smoking covid-19"]=['smoker','smoke','smokers','cigarette']

myDict["pulmonary disease covid-19"]=['lung','vascular','airway','coronary', 'alveolar', 'bronchial']

myDict["cancer covid-19"]=['cancers','carcinoma','hcc','cancer23','tumour','gbm','adenocarcinoma','tumor','nsclc']

myDict['covid-19 risk factors']=['covid-19','covid19','covid -19', 'covid- 19','covid - 19','covid 19','covid-2019','covid 2019','sars-cov2','sars-cov-2','2019-ncov','2019n-cov','wuhan coronavirus','risks','hazard','determinants','factor','cofactors','co-factors']

myDict['respiratory disease covid-19']=['lower-respiratory', 'upper-respiratory', 'respira-tory', 'upperrespiratory','aerodigestive','respiratory']

myDict['co-infections risk covid-19']=['coinfections','co-detections', 'coinfection','co-infection','co-detection','codetection', 'codetections','co-pathogens', 'copathogens','uris']

myDict['risk factors for neonates and pregnant women']=['newborns','infants','foals', 'babies', 'children', 'nonpregnant','non-pregnant','postmenopausal', 'pregnancy','infertile', 'mothers']



#print(len(myDict))
# function to rescore the sentences based on other factors

def rescoring(df, dict_tfidf_new, indicator, synonym_list_for_indicator):

    final_score=[]

    bad_char = '''['"()]''' 

    exclude = set(string.punctuation)

    good_list=['covid-19','covid19','covid -19', 'covid- 19','covid - 19','covid 19','covid-2019','covid 2019','sars-cov2','sars-cov-2','2019-ncov','2019n-cov','wuhan coronavirus']

    bad_list=['covid','mers-cov','sars-cov','corona','corona virus','coronavirus','sars','mers']

    suppress_list=['figure','table','fig','objective','aim','he','she','?']

    #synonym_list_for_indicator=['insulin','insulin-dependent','insulin resistance','glucose control','blood glucose level','metaformin', 'hemoglobin A1c','hyperglycemia','hypoglycemia', 'hyperglycemic', 'hypoglycemic', 'pre-diabetes', 'obesity', 'mellitus', 'hypercholesterolemia', 'hypertension','hyperlipidemia','t2dm','dyslipidemia']

    for ind in tqdm(range(len(df)), total=len(df)):

        score=0      

        score1=0.5

        score2=0.3

        flag=0

        sent=df['Sentence'][ind]

        #print(sent)

        sent=sent.lower()

        sent_without_punct = ''.join(ch for ch in sent if ch not in exclude)     #### removing punctuations

        sent_without_punct_tokenized=word_tokenize(sent_without_punct)      #### tokenizing words

        sent_without_punct_tokenized_without_sw=[word for word in sent_without_punct_tokenized if not word in stopwords.words()] ## removing sws

        len_sent=len(sent_without_punct_tokenized_without_sw)  ### computing length of processed sent

        indicator = indicator.lower()

        search = indicator.split("_")

        search2 = indicator.replace("_"," ") 



        dis_rel=df['Rel_Disease'][ind]

        dis_rel = ''.join(ch for ch in dis_rel if ch not in bad_char)    ### joining all entities and removing punctuations

        listt=dis_rel.split(",")   ### making a list of entities (kp,np,disease,gene,cell,chemical,etc.)

        listt_dis_rel=[]     ### pre-processing list of entities

        for i in listt:

            i=i.lower()

            i=i.strip()

            if i!="":

                listt_dis_rel.append(i)

        

        dis_irr=df['Irrel_Disease'][ind]

        dis_irr = ''.join(ch for ch in dis_irr if ch not in bad_char)    ### joining all entities and removing punctuations

        listt=dis_irr.split(",")   ### making a list of entities (kp,np,disease,gene,cell,chemical,etc.)



        listt_dis_irr=[]     ### pre-processing list of entities

        for i in listt:

            i=i.lower()

            i=i.strip()

            if i!="":

                listt_dis_irr.append(i)



        dis_neigh=df['Neighboring_Sentence_Disease'][ind]

        dis_neigh = ''.join(ch for ch in dis_neigh if ch not in bad_char)    ### joining all entities and removing punctuations

        listt=dis_neigh.split(",")   ### making a list of entities (kp,np,disease,gene,cell,chemical,etc.)



        listt_dis_neigh=[]     ### pre-processing list of entities

        for i in listt:

            i=i.lower()

            i=i.strip()

            if i!="":

                listt_dis_neigh.append(i)



        if len(listt_dis_rel)>0:    #### covid related term in sent

            flag=1



        if flag==0:                 #### discard sent (irrelevant disease in sent)

            if len(listt_dis_irr)>0:

                flag=2



        if flag==0:                #### here flag=0 depicts that none of upper given conditions hold

            for i in good_list:       #### if covid related term in neighbouring sent

                if i in listt_dis_neigh:

                    flag=1

                    break   



        if flag!=1:

            final_score.append(0)



        else:   

            for i in search:

                if i in dict_tfidf_new:

                    if i in sent:

                        score=score+dict_tfidf_new[i]



            for i in synonym_list_for_indicator:

                i=i.lower()

                i=i.strip()

                if i in dict_tfidf_new:

                    if i in sent:

                        score=score+dict_tfidf_new[i]



            if search2 in sent:

                score=score+score1

            else:

                if search[0] in sent:

                    score=score+score2



            for i in suppress_list:    

                if i.lower() in sent:

                    score=score-score1



            score=score+df['sentence_score'][ind]  

            Text=df['sent_jnlpba_ent'][ind]+","+df['sent_craft_ent'][ind]+","+df['sent_bc5cdr_ent'][ind]+","+df['sent_bionlp13cg_ent'][ind]+","+df['sent_sci_ent'][ind]

            Text = ''.join(ch for ch in Text if ch not in bad_char)    ### joining all entities and removing punctuations

            listt=Text.split(",")   ### making a list of entities (kp,np,disease,gene,cell,chemical,etc.)

            listtt=[]     ### pre-processing list of entities

            for i in listt:

                i=i.lower()

                i=i.strip()

                if i!="":

                    listtt.append(i)



            listtt_sorted=sorted(listtt, key=len, reverse=True)    #### sorting list by length of elements

            superset_list=[]    #### keeping only superset strings

            for i in listtt_sorted:

                flag=0

                for j in superset_list:

                    if set(i)<=set(j):

                        flag=1

                        break

                if flag==0:

                    superset_list.append(i)

            score=score+float(len(superset_list)/len_sent)

            final_score.append(score)

            #print(ind)

    df['New_Scores']=final_score

    return df
def finding_relevant_document(clean_sentence_df1, filtered_covid_document_df1):

    dict_doc_scores={}

    for i in range(0,len(clean_sentence_df1)):

        doc_id=clean_sentence_df1['Cord_uid'].loc[i]

        if doc_id in dict_doc_scores:

            if clean_sentence_df1['New_Scores'].loc[i]>0:

                dict_doc_scores[doc_id]=dict_doc_scores[doc_id]+clean_sentence_df1['New_Scores'].loc[i]

        else:

            if clean_sentence_df1['New_Scores'].loc[i]>0:

                dict_doc_scores[doc_id]=clean_sentence_df1['New_Scores'].loc[i]





    #filtered_covid_docs_df = pd.read_csv("Filtered_Covid_Documents.csv", sep='\t')

    doc_score = []

    for i in range(0,len(filtered_covid_document_df1)):

        doc_id = filtered_covid_document_df1['Cord_uid'].loc[i]

        if doc_id in dict_doc_scores:

            doc_score.append(dict_doc_scores[doc_id])

        else:

            doc_score.append(0)



    filtered_covid_document_df1['Doc_score'] = doc_score

    return filtered_covid_document_df1

#filtered_covid_docs.to_csv("Filtered_Covid_Documents_with_Scores.csv", sep='\t')
tags = dict()

tags['smoke'] =  'smoking_status.csv'

tags['pulmonary'] = 'pulmonary.csv'

tags['respiratory'] = 'respiratory_disease.csv'

tags['diabetes'] = 'diabetes.csv'

tags['asthma'] = 'asthma.csv'

tags['comorbidity'] = 'comorbidities.csv'

tags['pregnant'] = 'neonatal_pregnancy.csv'

tags['hypertension'] = 'hypertension.csv'

tags['cerebral'] = 'cerebral.csv'

tags['cancer'] = 'cancer.csv'

tags['obesity'] = 'obesity.csv'

tags['heart'] = 'heart_disease.csv'

tags['alcohol'] = 'drinking.csv'

tags['tuberculosis'] = 'tuberculosis.csv'

tags['kidney'] = 'chronic_kidney_disease.csv'

tags['risk_factor'] = 'covid_risk_factor.csv'

tags['coinfection'] = 'coinfections.csv'
#function to load extracted_DocData.csv

def loadFromCsv():

    sevDict = dict()

    fatalDict = dict()

    sampleDict = dict()

    sampleMethodDict = dict()

    designDict = dict()

    nameDict = dict()

    dfP1 = pd.read_csv(dataFile, sep='\t' )

    dfP1 = dfP1.astype(str)

    for row in dfP1.itertuples(): 

        paper = row.Cord_uid

        name = row.Titles

        nameDict[paper] =  name

        string = row.Severe

        string = string.strip('[')

        string = string.strip(']')

        string = string.split('\'')

        sevlis = []

        for item in string:

            if item!=',':

                sevlis.append(item)

        sevDict[paper] = sevlis

        

        string = row.Fatal

        string = string.strip('[')

        string = string.strip(']')

        string = string.split('\'')

        fatallis = []

        for item in string:

            if item!=',':

                fatallis.append(item)

        fatalDict[paper] = fatallis



        if row.Design ==  'nan':

            designDict[paper] = ''

        else:

            designDict[paper] = row.Design

            

        if row.Sample ==  'nan':

            sampleDict[paper] = ''

        else:

            sampleDict[paper] = row.Sample

            

        if dfP1.at[row.Index, 'Sampling Method'] ==  'nan':

            sampleMethodDict[paper] = ''

        else:

            sampleMethodDict[paper] = dfP1.at[row.Index, 'Sampling Method']

            

        #print(paper, sampleMethodDict[paper])

            

    return sevDict, fatalDict, nameDict, designDict, sampleDict, sampleMethodDict
#get required values from dict

def getData(paper, indicator):

    severe = ''

    fatal = ''

    design = ''

    sample = ''

    sampleMethod = ''

    design = designDict[paper]

    sample = sampleDict[paper]

    sampleMethod  = sampleMethodDict[paper]

    

    val = sevDict[paper]

    #print('in getData', val)

    for v in val:

        #print(v)

        if indicator in v:

            if severe == '':

                severe = v.split(' : ')[1]

            else:

                res = any(ele in v for ele in ['OR', 'CI', 'HR'])

                if  not res:

                    if not 'p' in v.lower():

                        severe = v.split(' : ')[1]

    val = fatalDict[paper]

    #print('in getData', val)

    for v in val:

        #print(v)

        if indicator in v:

            if fatal == '':

                fatal = v.split(' : ')[1]

                

            else:

                res = any(ele in v for ele in ['OR', 'CI', 'HR'])

                #print(res, v)

                if  not res:

                    if not 'p' in v.lower():

                        fatal = v.split(' : ')[1]

    

    return severe, fatal, design, sample, sampleMethod



def formatData(indicator, sentDf):

    #sentDf = pd.read_csv(sentFie, sep='\t' )

    sentDf = sentDf.sort_values('Doc_score',ascending=False)

    takenDict = dict()

    

    

    #indicator = 'hypertension'

    dfObj = pd.DataFrame(columns=fields)

    for row in sentDf.itertuples(): 

        if float(row.Doc_score) <= 0:

            continue

        #pid = row.Cord_uid

        pid = row.Cord_uid

        #pid = pid.split('.(')[0]

        #pid = pid + '.'

        if pid not  in takenDict.keys() and pid in nameDict.keys():

            #print('inside')

            severe, fatal, design, sample, sampleMethod = getData(pid, indicator)

            takenDict[pid] = 1

            #snippet = row.snippet

            journal = row.Journal

            date = row.Date

            url = row.Study_link

            study = row.title

            sevSig =''

            fatalSig = ''

            searchVal = ['P=', 'P<', 'P>', 'P =', 'P <', 'P >', 'p=', 'p<', 'p>', 'p =', 'p <', 'p >', 'p-value']

            study = study + '.('+row.source+')'

            sevExtracted = ''

            if severe!= '' :

                sevExtracted = 'Extracted'

                

                sevSig = 'Significant'

                for sv in searchVal:

                    match = re.search(sv+r'[0-9. ]+', severe)

                    if match:

                        #print("match ", match.group())

                        severeM = match.group()

                        match2 = re.search(r'[0-9.]+', severeM)

                        if match2:

                            num = float(match2.group())

                            if num>=0.05 and ('>' in match.group()):

                                #print("Not Significant")

                                sevSig =  'Not Significant'

                                

            fatalExtracted = ''

            if fatal != '':

                fatalExtracted = 'Extracted'

                fatalSig = 'Significant'

                for sv in searchVal:

                    match = re.search(sv+r'[0-9. ]+', severe)

                    if match:

                        #print("match ", match.group())

                        severeM = match.group()

                        match2 = re.search(r'[0-9.]+', severeM)

                        if match2:

                            num = float(match2.group())

                            if num>=0.05 and ('>' in match.group()):

                                #print("Not Significant")

                                fatalSig =  'Not Significant'

                

            

                

            #dfObj = dfObj.append({ 'Date':date , 'Study': study , 'Study Link': url, 'Journal': journal , 'Severe': severe, 'Severe Significant':sevSig, 'Severe Adjusted':'' , 'Severe Calculated':sevExtracted , 'Fatality': fatal , 'Fatality Significant':fatalSig , 'Fatality Adjusted':'' , 'Fatality Calculated':fatalExtracted , 'Multivariate adjustment':'', 'Design': design, 'Sample': sample, 'Study Population': sampleMethod, 'Snippet': snippet},ignore_index=True )

            dfObj = dfObj.append({ 'Date':date , 'Study': study , 'Study Link': url, 'Journal': journal , 'Severe': severe, 'Severe Significant':sevSig, 'Severe Adjusted':'' , 'Severe Calculated':sevExtracted , 'Fatality': fatal , 'Fatality Significant':fatalSig , 'Fatality Adjusted':'' , 'Fatality Calculated':fatalExtracted , 'Multivariate adjustment':'', 'Design': design, 'Sample': sample, 'Study Population': sampleMethod}, ignore_index=True )

                    

    display(HTML(dfObj[:20].to_html()))

    dfObj.to_csv(tags[indicator])
document_path = "/kaggle/input/covid-19-dataset-filtering-and-sentence-extraction/Filtered_covid_documents_with_metadata.csv"

print('Data from filtered document is collected')

filtered_covid_document_df = getKeywordLists(document_path, seperator='\t')

filtered_covid_document_df = filtered_covid_document_df.drop(['Unnamed: 0'], axis=1)

dict_tfidf_new = get_tfidf_dict(filtered_covid_document_df)

print('Total number of documents having COVID-19 related terms and published in 2020 are : ',len(filtered_covid_document_df))

doc_abstractandtext = list(filtered_covid_document_df['Abstract_and_text'])

doc_abstractandtext = cleandocdata(doc_abstractandtext)

sentence_path = "/kaggle/input/extracting-entities-from-covid-documents/Extracted_entities_from_extracted_sentences_from_filtered_covid_documents.csv"

sentence_df = getKeywordLists(sentence_path, seperator='\t')

sentence_df = sentence_df.drop(['Unnamed: 0.1'], axis=1)

sentence_df = extract_neighbouring_info(sentence_df)

clean_sentence_df = sentence_df.drop_duplicates(subset=['Sentence'])

clean_sentence_df = clean_sentence_df.reset_index()

clean_sentence_df = clean_sentence_df.drop(['index', 'Unnamed: 0'], axis=1)

print('appending list of words of sentence in dataframe ')

clean_sentence_df = adding_list_of_words(clean_sentence_df)

print('loading design and sample data ')

dataFile = '/kaggle/input/cord19-obtaining-design-and-sampling-information/extracted_DocData.csv'

sevDict, fatalDict, nameDict, designDict, sampleDict, sampleMethodDict = loadFromCsv()

print('design and sample data is loaded')

fields = ['Date', 'Study', 'Study Link', 'Journal', 'Severe', 'Severe Significant', 'Severe Adjusted', 'Severe Calculated', 'Fatality', 'Fatality Significant', 'Fatality Adjusted', 'Fatality Calculated', 'Multivariate adjustment', 'Design', 'Sample', 'Study Population']
phrases = ['covid-19 risk factors']

indicator1 = 'risk_factor'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, file_name, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['cancer covid-19']

indicator1 = 'cancer'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, file_name, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['hypertension covid-19']

indicator1 = 'hypertension'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, indicator1, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['heart disease covid-19']

indicator1 = 'heart'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, file_name, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['smoking covid-19']

indicator1 = 'smoke'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, file_name, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['diabetes covid-19']

indicator1 = 'diabetes'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, indicator1, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['pulmonary disease covid-19']

indicator1 = 'pulmonary'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, indicator1, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['risk factors for neonates and pregnant women']

indicator1 = 'pregnant'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, file_name, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['respiratory disease covid-19']

indicator1 = 'respiratory'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, indicator1, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)
phrases = ['co-infections risk covid-19']

indicator1 = 'coinfection'

phrase_words = gettingphrasewords(phrases)

idf_dict = getting_phrase_word_dict_with_idf_value(doc_abstractandtext, phrase_words)

clean_sentence_df1 = phrasematching(phrases[0], idf_dict, clean_sentence_df)

clean_sentence_df1= clean_sentence_df1.reset_index()

clean_sentence_df1= clean_sentence_df1.drop(['index'], axis=1)

file_name = '_'.join(phrases[0].split())

clean_sentence_df1 = rescoring(clean_sentence_df1, dict_tfidf_new, indicator1, myDict[phrases[0]])

filtered_covid_document_df1 = finding_relevant_document(clean_sentence_df1, filtered_covid_document_df)

formatData(indicator1, filtered_covid_document_df1)