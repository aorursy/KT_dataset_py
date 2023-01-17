# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
### importing required modules



import matplotlib.pyplot as plt

import re

import spacy

from spacy import displacy

import nltk

from fuzzywuzzy import fuzz, process

import warnings

import json

import gensim

import seaborn as sns

import wordcloud

from PIL import Image

import requests

import textblob

from gensim.parsing.preprocessing import remove_stopwords

warnings.filterwarnings(action="ignore")
job_bulletins=os.listdir("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/")
path_job_bulletins = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"

path_additional_data = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/"

### regex list:

### 1. multiple spaces

mul_sp_pat = re.compile(r"\s+")

### 2. removing \n

nxt_ln_pat = re.compile(r"\n")

bulletins_list1 = list()

for job in job_bulletins:

    with open("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"+(job), 'r', encoding='ISO-8859-1') as file: 

        bulletine = file.readlines()

    mod_bulletine = list()

    k = str()

    for line in bulletine:

        k = (mul_sp_pat.sub(repl=" ", string=line)).strip()

        k = nxt_ln_pat.sub(repl="", string=k)

        if k!="":

            mod_bulletine.append(k)

    bulletins_list1.append(mod_bulletine)
## picking the headings in the each job bulletin based on the condition that heading is in uppercase.

heading_list=list()

for each_bulletin in bulletins_list1:

    ### passing the very first line in each job bulletin since it is the position name which is in uppercase. so not to confuse it with heading names

    iteration = iter(each_bulletin)

    next(iteration)

    for line in iteration:

        if line.isupper():

            heading_list.append(line)
np.unique(heading_list)
### selecting the reasonable headings to form the list

key_words_list = ['ANNUAL SALARY','APPLICATION DEADLINE','APPLICATION PROCESS','CERTIFICATION','CONDITIONS OF EMPLOYMENT', 'DUTIES','NOTE','NOTES',

                  'NOTICE', 'WHERE TO APPLY', 'SCORE BANDING','SELECTIVE CERTIFICATION', 'SELECTION PROCEDURE','SELELCTION PROCESS', 'SKILLS, KNOWLEDGES, ABILITIES, AND PERSONAL QUALIFICATIONS', 

                 'SPECIAL INFORMATION', 'POSITIONS AVAILABLE', 'PROCESS NOTE','QUALIFICATIONS REVIEW','REQUIRED MATERIALS','REQUIREMENTS', 'REQUIREMENTS/ MINIMUM QUALIFICATIONS','ADDITIONAL JOB INFORMATION',

                  'ALLOWABLE CALCULATORS', 'EXPERT REVIEW COMMITTEE',  ]

k = bulletins_list1[15]

k
%%time



final_data_json = list()

for job_title,jobs in zip(job_bulletins,bulletins_list1):

    job_document = dict()

    job_document.update({"job_position":re.sub(pattern="\s+\d+.*",repl="", string=job_title.replace(".txt",""))})

    flag = False

    for lines in jobs:

        ### condition for adding the job class code

        if True in [lines.lower().startswith(word) for word in ['classcode', 'class code', 'class code:', 'classcode:']]:

            job_document.update({"Class Code" : re.findall(pattern=r"(\d+)", string=lines)[0]})

        fuzzy_process_tuple = process.extractOne(lines, key_words_list)

        if fuzzy_process_tuple[1]>=90:

            if flag:

                job_document.update({dict_key: data_list})

            flag = True

            data_list = list()

            dict_key = fuzzy_process_tuple[0]

        else:

            if flag:

                data_list.append(lines)

    left_out_columns = set(key_words_list)-set(job_document.keys())

    for cols in left_out_columns:

        job_document.update({cols:np.nan})

    final_data_json.append(job_document)

### creating the dataframe based on the created json object

jobs_dataframe=pd.DataFrame(final_data_json)

jobs_dataframe.head()
### Need to do some twicks here i hope the REQUIRMENTS and REQUIREMENTS/ MINIMUM QUALIFICATIONS are same as they tell what kind of qualification are required for the job position. So combine both as single column

jobs_dataframe['REQUIREMENTS/ MINIMUM QUALIFICATIONS'][jobs_dataframe['REQUIREMENTS/ MINIMUM QUALIFICATIONS'].isna()]=jobs_dataframe['REQUIREMENTS'][jobs_dataframe['REQUIREMENTS/ MINIMUM QUALIFICATIONS'].isna()]
### we can see all the REQUIREMENTS/ MINIMUM QUALIFICATIONS has been filled up so we will remove the REQUIREMENTS column.

jobs_dataframe[jobs_dataframe['REQUIREMENTS/ MINIMUM QUALIFICATIONS'].isna()]
jobs_dataframe.drop(columns=['REQUIREMENTS'], inplace=True)
#### regex patterns for finding salary figures and salary range figures with (to or -)

sal_fig_rng_pat = re.compile(r"(?:(\$\s*\d+(?:\,\d+)?\s+(?:to|-)\s+\$\d+(?:\,\d+)?)|(\$\s*\d+(?:\,\d+)?))") ### compiling it once so it can fasten the pattern finding





### according the given criteria in kaggle_data_dictionary.csv in general salary range only the first range is to be picked and the other ranges should be ignored.

only_first_sal_rng_pat = re.compile(r"(?:(^\$\s*\d+(?:\,\d+)?\s+(?:to|-)\s+\$\d+(?:\,\d+)?)|(^\$\s*\d+(?:\,\d+)?))")





annual_sal = list()

for sal_list,job_title in zip(jobs_dataframe['ANNUAL SALARY'],jobs_dataframe['job_position']):

    sample_dict = dict()

    if str(sal_list)!="nan":

        if len(sal_list)>1:

            flag_temp = False

            for line_no in range(len(sal_list)):

                if fuzz.partial_token_set_ratio(sal_list[line_no],'department')>=80 and line_no>0:

                    temp = sal_fig_rng_pat.findall(string=sal_list[line_no])

                    if temp:

                        sample_dict.update({"Department_salary":"".join(temp[0])})

                    else:

                        sample_dict.update({"Department_salary":"Not specified"})

                else:

                    temp = sal_fig_rng_pat.findall(string=sal_list[line_no])

                    if temp and not flag_temp:

                        flag_temp = True

                        sample_dict.update({"General_salary":"".join(temp[0])})

                    elif temp and flag_temp:

                        sample_dict.update({"Other_salary_range":"".join(temp[0])})

                    else:

                         sample_dict.update({"General_salary":"Not specified"})

        else:

            ## divide the line into two parts like before the word department and after the word department

            ## before department word is general salary range and after department word is department salary range

            dept_flg = False

            aftr_dept_line = str()

            gen_sal_line = str()

            for word in sal_list[0].split():

                if fuzz.partial_token_set_ratio(word,"department")>=75:

                    dept_flg = True

                if dept_flg:

                    aftr_dept_line = aftr_dept_line+" "+word

                else:

                    gen_sal_line = gen_sal_line+" "+word

            temp = re.findall(pattern=sal_fig_rng_pat,string=aftr_dept_line.strip())

            if temp:

                sample_dict.update({"Department_salary":"".join(temp[0])})

            else:

                sample_dict.update({"Department_salary":"Not specified"})

            temp = re.findall(pattern=only_first_sal_rng_pat, string=gen_sal_line.strip())

            if temp:

                sample_dict.update({"General_salary":"".join(temp[0])})

            else:

                sample_dict.update({"General_salary":"Not specified"})             

    else:

        sample_dict = dict()

        sample_dict.update({"General_salary":"Not specified"})

        sample_dict.update({"Department_salary":"Not specified"})

        

    annual_sal.append(sample_dict)
## Annual Salary

## Salary range has the General and Department wise salaries so im splitting here into to columns.

## create a 2 columned dataframe with column names as General_salary_range and department_salary_range



annual_salary_data = pd.DataFrame(columns=['job_position','Department_salary','General_salary','Other_salary_range'])

annual_salary_data['job_position']=jobs_dataframe['job_position']

annual_salary_data[['Department_salary','General_salary','Other_salary_range']]=pd.read_json(json.dumps(annual_sal))



### replace the NaNs with "Not Specified"

annual_salary_data.fillna(value="Not specified", inplace=True)

print(annual_salary_data.isna().sum())

annual_salary_data.head()
requirements_df = jobs_dataframe[['job_position','REQUIREMENTS/ MINIMUM QUALIFICATIONS']]
for i in requirements_df.loc[:,'REQUIREMENTS/ MINIMUM QUALIFICATIONS']:

    print(i)
### creating the requirments, requirment subset, lemma, pos, tag, is_stop_words

nlp=spacy.load(name="en_core_web_sm")

spacy_df = pd.DataFrame(columns=['job_title','requirement_no','requirement_subset','tokens_text','tokens_lemma','tokens_pos','tokens_tag','tokens_is_stop'])

putn_pat = re.compile(r"[.,\/#!$%\^&\*;?<>:{}=\_`~\[\]\"\']")

bullet_pat = re.compile(r"^\d.|^\w\.")

for job_title,each_req in zip(requirements_df['job_position'],requirements_df['REQUIREMENTS/ MINIMUM QUALIFICATIONS']):

    requirment_subset="nan"

    requirement_no = "nan"

    for line in each_req:

        dummy_list = list()

        ### finding the bullet numbers and alphabets for each line

        req_number = bullet_pat.findall(string=line)

        if req_number:

            line = bullet_pat.sub(repl="",string=line)

            if req_number[0][0].replace(".","").isalpha():

                requirment_subset = req_number[0][0].replace(".","")

            else:

                requirement_no = req_number[0][0].replace(".","")

                requirment_subset = "nan"

        ### Puntuation removal

        line = putn_pat.sub(repl=" ",string=line)

        doc = nlp(line)

        for tokens in doc:

            dummy_list.append([job_title,requirement_no,requirment_subset,tokens.text,tokens.lemma_,tokens.pos_,tokens.tag_,tokens.is_stop])

        spacy_df = pd.concat([spacy_df,pd.DataFrame(dummy_list,columns=['job_title','requirement_no','requirement_subset','tokens_text','tokens_lemma','tokens_pos','tokens_tag','tokens_is_stop'])], axis=0)

        
token_understanding  = pd.DataFrame(spacy_df[['tokens_pos','tokens_tag']].groupby(['tokens_pos','tokens_tag'])['tokens_tag'].count())

token_understanding.rename({'tokens_tag':'count_of_tags'}, axis='columns', inplace=True)

token_understanding.reset_index(inplace=True)

### expanding the tags and pos so that we can get the better understanding of it

token_understanding.tokens_pos = token_understanding.tokens_pos.map(lambda word: word+" - "+spacy.explain(word))

token_understanding.tokens_tag = token_understanding.tokens_tag.map(lambda word: "" if not spacy.explain(word) else word+" - "+spacy.explain(word))
spacy_df.head()
### removing the Punctuations spaces determiner adposition coordinating conjunction

new_spacy_df = spacy_df[~spacy_df.tokens_pos.isin(['PUNCT','SPACE','DET','ADP','CCONJ'])]

new_spacy_df.head()
for i in jobs_dataframe.DUTIES:

    print(i)
## flattening the list and removing the nan in the Duties column for building the Word cloud.

flattened_duties_list =sum(list(jobs_dataframe[~jobs_dataframe.DUTIES.isna()]['DUTIES']),[])

text = " ".join(flattened_duties_list)

word_cloud = wordcloud.WordCloud(background_color="white").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(word_cloud, interpolation='bilinear')

plt.axis('off')

plt.show()
## remove the punctuations

putn_pat = re.compile(r"[.,\/#!$%\^&\*;?<>:{}=\_`~\[\]\"\-\']")

cleaned_duties = list(map(lambda line: putn_pat.sub(repl=" ",string=line),flattened_duties_list))



### using simple module textblob for pos tagging before i used spacy but of no use and i got some misleading results

tb_cleaned_duties = list(map(lambda line: textblob.TextBlob(line), cleaned_duties))

### list the two types of Pronouns and the four types of nouns 

pronouns = list()

nouns = list()

for indx,duties in enumerate(tb_cleaned_duties):

    for i in duties.tags:

        if i[1]=='PRP' or i[1]=='PRP$' : ### Personel Pronoun and Possesive Pronoun

            x = list(i)

            x.append(indx)

            pronouns.append(x)

        if i[1] in ['NN','NNS','NNP','NNPS']: ### Noun singular, Noun plural, Popular noun singular, Popular noun plural

            x = list(i)

            x.append(indx)

            nouns.append(x)
pronouns = pd.DataFrame(pronouns)

plt.figure(figsize=(10,5))

pronouns[0].map(lambda line: line.lower()).value_counts().plot(kind='bar')
pronouns[pronouns[0].str.lower()=='i']
cleaned_duties[264]
with open (path_job_bulletins+"Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt", 'r', encoding='ISO-8859-1') as file:

    posting = file.readlines()

for i in posting:

    if i not in ['\n']:

        print(i)
## converting the list of list to dataframe for easy analysis

nouns = pd.DataFrame(nouns)

filter_nouns = list()

for i in nouns[0].str.lower():

    if any([i.endswith(j) for j in ['or','er','ar']]):

        filter_nouns.append(i)



plt.figure(figsize=(25,8))

pd.Series(filter_nouns).value_counts().plot(kind='bar')
### all the words are converted to its root level so we can find the different variations of the same words

masculine_words = list(('active',

'adventurous',

'aggress',

'ambitio',

'analy',

'assert',

'athlet',

'autonom',

'battle',

'boast',

'challeng',

'champion',

'compet',

'confident',

'courag',

'decid',

'decisio,n'

'decisive',

'defend',

'determin,'

'domina',

'dominant',

'driven',

'fearless',

'fight',

'force',

'greedy',

'head-strong',

'headstrong',

'hierarch',

'hostil',

'impulsive',

'independen',

'individual',

'intellect',

'lead',

'logic',

'objective',

'opinion',

'outspoken',

'persist',

'principle',

'reckless',

'self-confiden',

'self-relian',

'self-sufficien',

'selfconfiden',

'selfrelian',

'selfsufficien',

'stubborn',

'superior',

'unreasonab'))



feminine_words = list(('agree',

'affectionate',

'child',

'cheer',

'collab',

'commit',

'communal',

'compassion',

'connect',

'considerate',

'cooperat',

'co-operat',

'depend',

'emotiona',

'empath',

'feel',

'flatterable',

'gentle',

'honest',

'interpersonal',

'interdependen',

'interpersona',

'inter-personal',

'inter-dependen',

'inter-persona',

'kind',

'kinship',

'loyal',

'modesty',

'nag',

'nurtur',

'pleasant',

'polite',

'quiet',

'respon',

'sensitiv',

'submissive',

'support',

'sympath',

'tender',

'together',

'trust',

'understand',

'warm',

'whin',

'enthusias',

'inclusive',

'yield',

'share',

'sharin',

))
### skipping the jobs with no duties

spacy.l

duties = jobs_dataframe[~jobs_dataframe.DUTIES.isna()][['job_position','DUTIES']]

for i in range(duties.shape[0]):

    doc = nlp(' '.join(duties.iloc[i,1]))

    for d in doc:

        print(d.text)
## we can see no gender biased are being used in the Duties like he,she etc.,

nouns_words = pd.Series([i[0].lower() for i in nouns])


nouns_count = nouns_words.value_counts().reset_index()

nouns_count.rename(columns={'index':'words', 0:'count'}, inplace=True)

### lets see the noun words used only once in duties in all the job requests

plt.figure(figsize=(30,10))

nouns_count[nouns_count['count']>5].plot(kind='bar', x='words',y='count')