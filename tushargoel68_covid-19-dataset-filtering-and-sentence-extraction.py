!pip install spacy
!pip install scispacy
!pip install langdetect
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

from tqdm.notebook import tqdm

from datetime import datetime

from langdetect import detect

import scispacy

import spacy

import re

import en_core_sci_lg

from IPython.display import display, HTML

import os
all_json = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for f in filter(lambda x: x.split('.')[-1]=='json', filenames):

    #for filename in filenames:

        all_json.append(os.path.join(dirname,f))

        #print(os.path.join(dirname, f))

print('Total json files available : ',len(all_json))



print("reading metadata...")

metadf = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

print("total entries in metadata : ",len(metadf))
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            if 'abstract' in content:

                for entry in content['abstract']:

                    self.abstract.append(entry['text'])

            else:

                self.abstract.append('no abstract')

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):

        return self.paper_id +':' + ' '+ self.abstract[:200]+"..."+ self.body_text[:200]+'...'
paper_id = []

abstract = []

text = []

title = []

study_link = []

source = []

date = []

journal = []

authors = []

url = []

cord_id = []

pmc_id =[]

pubmed_id = []

microsoft_id = []

full_text_file = []



for i in tqdm(range(len(all_json)), total=len(all_json)):

    #print(all_json[i])

    content = FileReader(all_json[i])

    meta_data = metadf.loc[metadf['sha'] == content.paper_id]

    if len(meta_data) != 0:

        title.append(meta_data['title'].values[0])

        paper_id.append(content.paper_id)

        if content.abstract != 'no abstract':

            abstract.append(content.abstract)

        else:

            abstract.append(meta_data['abstract'].values[0])

        text.append(content.body_text)

        link = meta_data['doi'].values[0]

        if link != 'nan':

            study_link.append(link)

        else:

            study_link.append('not available')

        publish_time = meta_data['publish_time'].values[0]

        if publish_time != 'nan':

            date.append(publish_time)

        else:

            date.append('not available')

        jrnl = meta_data['journal'].values[0]

        if jrnl != 'nan':

            journal.append(jrnl)

        else:

            journal.append('not available')

        src = meta_data['source_x'].values[0]

        if src != 'nan':

            source.append(src)

        else:

            source.append('not available')

        pmc = str(meta_data['pmcid'].values[0])

        if pmc != 'nan':

            pmc_id.append(pmc)

        else:

            pmc_id.append('not available')

        micro = str(meta_data['Microsoft Academic Paper ID'].values[0])

        if micro != 'nan':

            microsoft_id.append(micro)

        else:

            microsoft_id.append('not available')

        pub_id = str(meta_data['pubmed_id'].values[0])

        if pub_id != 'nan':

            pubmed_id.append(pub_id)

        else:

            pubmed_id.append('not available')

        authors.append(str(meta_data['authors'].values[0]))

        url.append(str(meta_data['url'].values[0]))

        cord_id.append(str(meta_data['cord_uid'].values[0]))

        full_text_file.append(str(meta_data['full_text_file'].values[0]))

    else: 

        meta_data = metadf.loc[metadf['pmcid'] == content.paper_id]

        if len(meta_data) != 0:

            title.append(meta_data['title'].values[0])

            paper_id.append(content.paper_id)

            if content.abstract != 'no abstract':

                abstract.append(content.abstract)

            else:

                abstract.append(meta_data['abstract'].values[0])

            text.append(content.body_text)

            link = meta_data['doi'].values[0]

            if link != 'nan':

                study_link.append(link)

            else:

                study_link.append('not available')

            publish_time = meta_data['publish_time'].values[0]

            if publish_time != 'nan':

                date.append(publish_time)

            else:

                date.append('not available')

            jrnl = meta_data['journal'].values[0]

            if jrnl != 'nan':

                journal.append(jrnl)

            else:

                journal.append('not available')

            src = meta_data['source_x'].values[0]

            if src != 'nan':

                source.append(src)

            else:

                source.append('not available')

            pmc = str(meta_data['pmcid'].values[0])

            if pmc != 'nan':

                pmc_id.append(pmc)

            else:

                pmc_id.append('not available')

            micro = str(meta_data['Microsoft Academic Paper ID'].values[0])

            if micro != 'nan':

                microsoft_id.append(micro)

            else:

                microsoft_id.append('not available')

            pub_id = str(meta_data['pubmed_id'].values[0])

            if pub_id != 'nan':

                pubmed_id.append(pub_id)

            else:

                pubmed_id.append('not available')

            authors.append(str(meta_data['authors'].values[0]))

            url.append(str(meta_data['url'].values[0]))

            cord_id.append(str(meta_data['cord_uid'].values[0]))

            full_text_file.append(str(meta_data['full_text_file'].values[0]))



data_df = pd.DataFrame({'Cord_uid':cord_id, 'Paper_id':paper_id, 'Study_link':study_link, 'Date':date, 'Journal':journal, 'source':source, 'title':title, 'abstract':abstract, 'text':text, 'authors':authors, 'Study_url':url, 'Pmcid':pmc_id, 'Pubmed_id':pubmed_id, 'Microsoft_paper_id':microsoft_id, 'Full_text_file':full_text_file})
# Helper function to filter document containing covid-19 related terms 

def filter_document(original_data_df, corona_words):

    flag_ent = []

    for l in range(len(original_data_df)):

        full_text = original_data_df['text'].iloc[l]

        c_list = []

        for cword in corona_words:

            if cword in str(full_text).lower():

                c_list.append(1)

            else:

                c_list.append(0)    

        boolean = any([y == 1 for y in c_list])

        if boolean:

            flag = 1

        else:

            flag = 0

        full_abstract = original_data_df['abstract'].iloc[l]

        c_list1 = []

        for cword1 in corona_words:

            if cword1 in str(full_abstract).lower():

                c_list1.append(1)

            else:

                c_list1.append(0)    

        boolean = any([z == 1 for z in c_list1])

        if boolean:

            flag1 = 1

        else:

            flag1 = 0

        f_list = []

        f_list.append(flag)

        f_list.append(flag1)

        boolean = any([x == 1 for x in f_list])

        if boolean:

            flag_ent.append(1)

        else:

            flag_ent.append(0)        

    original_data_df['covid_flag'] = flag_ent

    r = (original_data_df['covid_flag']==1)

    filtered_df = original_data_df.loc[r,:].reset_index()

    return filtered_df, original_data_df
def get_year_of_publish(filtered_df):

    paper_year = []

    for i in range(len(filtered_df['Date'])):

        try:

            datetime1 = datetime.strptime(filtered_df['Date'][i], '%Y-%m-%d')

            paper_year.append(datetime1.date().year)

        except:

            paper_year.append(int(filtered_df['Date'][i]))

    return paper_year
# Helper function to collect text and abstract of dataset

def getting_text_and_abstract(filtered_df, paper_year):

    doc_id = []

    doc_titles = []

    doc_abstract = []

    doc_text = []

    doc_source = []

    doc_study_link = []

    doc_date = []

    doc_journal = []

    doc_abstractandtext = []

    doc_authors = []

    doc_pmcid = []

    doc_pubmedid = []

    doc_micro_id = []

    doc_full_text = []

    doc_cord_uid = []

    doc_url = []

#    count = 0

    for j in range(len(paper_year)):

        if paper_year[j] == 2020 :

            item = str(filtered_df['abstract'].iloc[j])

            item = item.replace('Abstract\n\n', '')

            item = item.replace('\n\n','')

            item1 = str(filtered_df['text'].iloc[j])

            item1 = item1.replace('Introduction\n\n', '')

            item1 = item1.replace('INTRODUCTION\n\n', '')

            item1 = item1.replace('\n\n',' ')

            item1 = item1.replace('\n', ' ')

            if item != '' and item1 != '':

                if detect(item1) == 'en':

                    combine_item = item + ' ' + item1

                    doc_abstract.append(item)

                    doc_id.append(filtered_df['Paper_id'].iloc[j])

                    doc_text.append(item1)

                    doc_titles.append(filtered_df['title'].iloc[j])

                    doc_source.append(filtered_df['source'].iloc[j])

                    doc_study_link.append(filtered_df['Study_link'].iloc[j])

                    doc_date.append(filtered_df['Date'].iloc[j])

                    doc_journal.append(filtered_df['Journal'].iloc[j])

                    doc_abstractandtext.append(combine_item)

                    doc_authors.append(filtered_df['authors'].iloc[j])

                    doc_pmcid.append(filtered_df['Pmcid'].iloc[j])

                    doc_pubmedid.append(filtered_df['Pubmed_id'].iloc[j])

                    doc_micro_id.append(filtered_df['Microsoft_paper_id'].iloc[j])

                    doc_full_text.append(filtered_df['Full_text_file'].iloc[j])

                    doc_cord_uid.append(filtered_df['Cord_uid'].iloc[j])

                    doc_url.append(filtered_df['Study_url'].iloc[j])

#            count = count + 1

    return doc_id, doc_abstract, doc_text, doc_titles,doc_source, doc_date, doc_study_link, doc_journal, doc_abstractandtext, doc_authors, doc_pmcid, doc_pubmedid, doc_micro_id, doc_full_text, doc_cord_uid, doc_url
# Helper function to extract sentences from full text (abstract + text)

def get_sentences_from_full_text(doc_cord_uid, doc_abstractandtext, doc_titles):

    nlp = en_core_sci_lg.load()

    clean_full_text_sentences = []

    clean_doc_uid = []

    clean_titles = []

    clean_sentence_id = []

    for k in range(len(doc_cord_uid)):

        full_text = doc_abstractandtext[k]

        doc = nlp(full_text)

        sentences = list(doc.sents)

        for n in range(len(sentences)):

            sent = str(sentences[n])

            sent = sent.encode('ascii','ignore')

            sent = sent.decode('utf-8')

            sent = re.sub(r"\[[0-9]*?\]", "",sent)

            sent = sent.strip()

            if not sent.startswith('https') and len(sent) > 15:

                clean_full_text_sentences.append(sent)

                clean_doc_uid.append(doc_cord_uid[k])

                clean_titles.append(doc_titles[k])

                clean_sentence_id.append(str(n))

    return clean_full_text_sentences, clean_doc_uid, clean_titles, clean_sentence_id
# COVID-19 synonyms

corona_words = ['covid','covid19','covid-19','2019-ncov','sars-cov-2', '2019 ncov','ncov 2019','2019ncov','2019 n cov','coronavirus 2019','wuhan coronavirus','novel coronavirus']

clean_data_df = data_df.drop_duplicates(subset=['title'])

new_data_df = clean_data_df.reset_index()

new_data_df.to_csv("full_dataset_along_with_metadata.csv", sep='\t')

original_data_df = clean_data_df.dropna()

filtered_df, original_data_df = filter_document(original_data_df, corona_words)

filtered_df = filtered_df.drop(['index'], axis=1)

paper_year = get_year_of_publish(filtered_df)

doc_id, doc_abstract, doc_text, doc_titles,doc_source, doc_date, doc_study_link, doc_journal, doc_abstractandtext, doc_authors, doc_pmcid, doc_pubmedid, doc_micro_id, doc_full_text, doc_cord_uid, doc_url = getting_text_and_abstract(filtered_df, paper_year)

# saving tested documents

tested_df = pd.DataFrame({'Cord_uid':doc_cord_uid,'Paper_id':doc_id, 'abstract':doc_abstract, 'text':doc_text, 'title':doc_titles, 'source':doc_source, 'Date':doc_date, 'Study_link':doc_study_link, 'Journal':doc_journal, 'Abstract_and_text':doc_abstractandtext, 'authors':doc_authors, 'Pmcid':doc_pmcid, 'Pubmed_id':doc_pubmedid, 'Microsoft_paper_id':doc_micro_id, 'Full_text_file':doc_full_text, 'Study_url':doc_url})

#Save this file for future reference

print('total documents to be filtered : ',len(tested_df))

print('displaying samples of filtered documents...')

display(HTML(tested_df[:3].to_html()))

tested_df.to_csv("Filtered_covid_documents_with_metadata.csv", sep='\t')

print('Collecting sentences')

clean_full_text_sentences, clean_doc_uid, clean_titles, clean_sentence_id = get_sentences_from_full_text(doc_cord_uid, doc_abstractandtext, doc_titles)

sentence_df = pd.DataFrame({'Cord_uid':clean_doc_uid,'Sentence':clean_full_text_sentences, 'Titles':clean_titles, 'Sentence_id':clean_sentence_id})

print('total number of sentences : ',len(clean_full_text_sentences))

sentence_df.to_csv("Extracted_sentences_from_filtered_covid_documents.csv", sep='\t')