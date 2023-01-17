#These open source libraries are using python 3.6

from __future__ import unicode_literals, print_function

import numpy as np 

import pandas as pd 

import re

import nltk

import string

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from nltk import word_tokenize

import random

import dateutil.parser as dparser

from datetime import datetime

!pip install datefinder

import datefinder

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize

from wordcloud import WordCloud, STOPWORDS

from flashtext import KeywordProcessor

import os

import plac

import spacy
def load_jobopening_dataset():



    data_path = "../input/data-science-for-good-city-of-los-angeles/CityofLA/CityofLA/Job Bulletins/"



    texts = []

    positions = []

    file_names=[]

    for fname in sorted(os.listdir(data_path)):

        if fname.endswith('.txt'):

            file_names.append(fname)

            with open(os.path.join(data_path, fname),"rb") as f:

                texts.append(str(f.read()))

                positions.append((re.split(' (?=class)', fname))[0])

    

    #print the length of the List of text, length of file_names and positions and make sure they are all equal

    return (texts,positions,file_names)



#job_data, positions, file_names = load_jobopening_dataset() #This code can't read texts from zip file so I would manually import the raw text files I exported from my personal computer

job_data = pd.read_csv("../input/raw-file/job_data.csv").values.tolist()

positions =  pd.read_csv("../input/raw-file/positions.csv").values.tolist()

file_names =  pd.read_csv("../input/raw-file/file_names.csv").values.tolist()
str(job_data[0]).replace("\\r\\n"," ").replace("\\\'s","")[:250]
titles = pd.read_csv("../input/additional-data/job_titles.csv", header=None)

titles.head()
data_dict=pd.read_csv("../input/additional-data/kaggle_data_dictionary.csv")

data_dict[:5]
def Position_parser(s):

    title_match=False

    pos = re.findall(r'(.*?)Class Code',s)

    pos1 = re.findall(r'(.*?)Class  Code',s)

    if (len(pos1) > 0):

        pos = pos1

    if (len(pos) > 0):

        job_title= pos[0].replace("b'","").replace("b\"","").replace("'","").replace("\\","").strip()

        for title in titles[0]:

            if (title.replace("'","")==job_title):

                title_match=True

                break

    if(title_match==True):

        return job_title

    else:

        return "Invalid job title" 
def JobCode_parser(s):

    job_code = 0

    code = re.findall(r'Class Code:(.*?)Open',s)

    if (len(code)>0):

        job_code= int(code[0].strip())

    return job_code
def OpenDate_parser(s):

    openDateRet=""

    openDate = re.findall(r'Open Date:(.*?)ANNUAL',s)

    openStr=""

    if (len(openDate)>0):

        #print(openDate)

        openDate = openDate[0].strip()

        openStr=re.findall(r'(?:Exam).*',openDate)

        #print(openStr)

    

    matches = list(datefinder.find_dates(openDate))



    if len(matches) > 0:

        for i in range(len(matches)):

            date = matches[i]

            openDateRet=str(date.date())

    return openDateRet,openStr
def SalaryRange_parser(s):

    salaryRange = re.findall(r'ANNUAL SALARY(.*?)NOTE',s)

    salaryRange_1 = re.findall(r'ANNUAL SALARY(.*?)DUTIES',s)

    salaryRange_2 = re.findall(r'ANNUAL SALARY(.*?)\(flat',s)

    len1=0

    len2=0

    len3=0

    if (len(salaryRange) > 0):

        len1 = len(salaryRange[0])

    if (len(salaryRange_1) > 0):

        len2 = len(salaryRange_1[0])

    if (len(salaryRange_2) > 0):

        len3 = len(salaryRange_2[0])

    if ((len1 > 0) & (len2 > 0)):

        if (len1 < len2):

            salaryRange = salaryRange

        else:

            salaryRange = salaryRange_1

        

    if (len(salaryRange)>0):

        salaryRange = salaryRange[0].strip()

    return salaryRange
def Qualification_parser(s):

    qual = re.findall(r'REQUIREMENTS/MINIMUM QUALIFICATIONS(.*?)WHERE TO APPLY',s)

    if (len(qual)==0):

        qual = re.findall(r'REQUIREMENT/MINIMUM QUALIFICATION(.*?)WHERE TO APPLY',s)

    if (len(qual)==0):

        qual = re.findall(r'REQUIREMENTS(.*?)WHERE TO APPLY',s)

    if (len(qual)==0):

        qual = re.findall(r'REQUIREMENT(.*?)WHERE TO APPLY',s)

    if (len(qual)>0):

        qual = qual[0].replace("\\'s","'s").strip()

    else:

        qual=""

    return qual
def Education_parser(s):

    educationMajor=""

    sentences = sent_tokenize(s)

    selected_sentences=[sent for sent in sentences if "major" in word_tokenize(sent)]

    for i in range(len(selected_sentences)):

        major = re.findall(r'major in(.*?),',selected_sentences[i])

        if (len(major)>0):

            educationMajor=major[0].strip()

    return educationMajor
def EduSemDur_parser(s):

    educationDur=""

    sentences = sent_tokenize(s)

    selected_sentences=[sent for sent in sentences if "semester" in word_tokenize(sent)]

    for i in range(len(selected_sentences)):

        dur = re.findall(r'(.*?)semester',selected_sentences[i])

        #print(dur)

        if (len(dur)>0):

            educationDur=dur[0]+'sememster'

    return educationDur
def Duties_parser(s):

    duties = re.findall(r'DUTIES(.*?)REQUIREMENT',s)

    jobDuties=""

    if (len(duties)>0):

        jobDuties= duties[0].strip()

    return jobDuties
def eduYears_parser(s):

    keyword_processor = KeywordProcessor()

    education_yrs=0.0

    keyword_processor.add_keyword('four-year')

    keyword_processor.add_keyword('four years')

    sentences = sent_tokenize(s)

    selected_sentences=[sent for sent in sentences if "degree" in word_tokenize(sent)]

    selected_sentences1=[sent for sent in sentences if "Graduation" in word_tokenize(sent)]



    for i in range(len(selected_sentences)):

        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])

        if (len(keywords_found) > 0):

            education_yrs=4.0

    for i in range(len(selected_sentences1)):

        keywords_found = keyword_processor.extract_keywords(selected_sentences1[i])

        if (len(keywords_found) > 0):

            education_yrs=4.0

   

    return education_yrs
def expYears_parser(s):

    keyword_processor = KeywordProcessor()

    exp_yrs=0.0

    keyword_processor.add_keyword('four-year')

    keyword_processor.add_keyword('four years')

    keyword_processor.add_keyword('three years')

    keyword_processor.add_keyword('one year')

    keyword_processor.add_keyword('two years')

    keyword_processor.add_keyword('six years')

    sentences = sent_tokenize(s)

    selected_sentences=[sent for sent in sentences if "experience" in word_tokenize(sent)]



    for i in range(len(selected_sentences)):

        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])

        for i in range(len(keywords_found)):

            if keywords_found[i]=='two years':

                exp_yrs=2.0

            elif keywords_found[i]=='one year':

                exp_yrs=1.0

            elif keywords_found[i]=='three years':

                exp_yrs=3.0

            elif keywords_found[i]=='six years':

                exp_yrs=6.0

            elif keywords_found[i]=='four years':

                exp_yrs=4.0

            elif keywords_found[i]=='four-year':

                exp_yrs=4.0

                

    return exp_yrs
def fullTimePartTime_parser(s):

    keyword_processor = KeywordProcessor()

    fullTimePartTime=""

    keyword_processor.add_keyword('full-time')

    keyword_processor.add_keyword('part-time')

    sentences = sent_tokenize(s)

    selected_sentences=[sent for sent in sentences if "experience" in word_tokenize(sent)]



    for i in range(len(selected_sentences)):

        keywords_found = keyword_processor.extract_keywords(selected_sentences[i])

        for i in range(len(keywords_found)):

            if keywords_found[i]=='full-time':

                fullTimePartTime="FULL TIME"

            elif keywords_found[i]=='part-time':

                fullTimePartTime="PART TIME"

           

                

    return fullTimePartTime
def DL_parser(s):

    dl = False

    dl_valid = False

    dl_State = ""

    arr = ['driver', 'license']

    keyword_processor = KeywordProcessor()

    keyword_processor.add_keyword('california')

    if any(re.findall('|'.join(arr), qual)):

        dl = True

    if (dl==True):

        sentences = sent_tokenize(s)

        selected_sentence=[sent for sent in sentences if "driver" in word_tokenize(sent)]

        if (len(selected_sentence)>0):

            words = selected_sentence[0].split()

            selected_word = [word for word in words if "valid" in words]

            if len(selected_word)>0:

                dl_valid=True

        for i in range(len(selected_sentence)):   

            keywords_found = keyword_processor.extract_keywords(selected_sentence[i])

            for i in range(len(keywords_found)):

                if keywords_found[i]=='california':

                    dl_State="CA"

                

    if (dl_valid)==True:

        dl_valid="R"

    else:

        dl_valid="P"

    return dl_valid,dl_State
def Relations_parser(TEXTS, nlp, ENTITY_TYPE):

    entities=[]

    for text in TEXTS:

        doc = nlp(text)

        relations = extract_entity_relations(doc,ENTITY_TYPE)

        for r1, r2 in relations:

            relation=r1.text+"-"+r2.text

            entities.append(relation)

    imp_entities='::::'.join(entities)   

    return imp_entities
def College_parser(s):

    college=""

    keyword_processor = KeywordProcessor()

    keyword_processor.add_keyword('college or university')

    keyword_processor.add_keyword('college')

    keyword_processor.add_keyword('university')

    keyword_processor.add_keyword('high school')

    sentences = sent_tokenize(s)

    for j in range(len(sentences)):

        sentence = sentences[j]

        keywords_found = keyword_processor.extract_keywords(sentence)

        if (len(keywords_found) > 0):

            for i in range(len(keywords_found) ):

                if (keywords_found[i]=='college or university'):

                    college='college or university'

                    break

                elif (keywords_found[i]=='college'):

                    college='college'

                    break

                elif (keywords_found[i]=='university'):

                    college='university'

                    break

                elif (keywords_found[i]=='high school'):

                    college='high school'

                    break

    



    return college
def filter_spans(spans):

    get_sort_key = lambda span: (span.end - span.start, span.start)

    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)

    result = []

    seen_tokens = set()

    for span in sorted_spans:

        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:

            result.append(span)

            seen_tokens.update(range(span.start, span.end))

    return result





def extract_entity_relations(doc,entity):

    # Merge entities and noun chunks into one token

    seen_tokens = set()

    spans = list(doc.ents) + list(doc.noun_chunks)

    spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:

        for span in spans:

            retokenizer.merge(span)



    relations = []

    for money in filter(lambda w: w.ent_type_ == entity, doc):

        if money.dep_ in ("attr", "dobj"):

            subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]

            if subject:

                subject = subject[0]

                relations.append((subject, money))

        elif money.dep_ == "pobj" and money.head.dep_ == "prep":

            relations.append((money.head.head, money))

    return relations
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")



job_data_export=pd.DataFrame(columns=["FILE_NAME","JOB_CLASS_TITLE","JOB_CLASS_NO","REQUIREMENT_SET_ID",

                                      "REQUIREMENT_SUBSET_ID","JOB_DUTIES",

                                      "EDUCATION_YEARS","SCHOOL_TYPE","EDUCATION_MAJOR","EXPERIENCE_LENGTH","IMP_ENTITIES_QUAL",

                                     "FULL_TIME_PART_TIME","EXP_JOB_CLASS_TITLE","EXP_JOB_CLASS_ALT_RESP"

                                     ,"EXP_JOB_CLASS_FUNCTION","COURSE_COUNT","COURSE_LENGTH","COURSE_SUBJECT"

                                     ,"MISC_COURSE_DETAILS","DRIVERS_LICENSE_REQ","DRIV_LIC_TYPE",

                                     "ADDTL_LIC","EXAM_TYPE","ENTRY_SALARY_GEN","ENTRY_SALARY_DWP","OPEN_DATE","LEGAL_TERMS"])



for i in range(0, len(job_data)-1):



    s = str(job_data[i]).replace("\\r\\n"," ").replace("\\t","")

    position = Position_parser(s)

    qual = Qualification_parser(s)

    DL_valid,DL_state = DL_parser(qual)

    education_yrs = eduYears_parser(qual)

    education_major = Education_parser(qual)

    try:

        job_code = JobCode_parser(s)

        openDate, openStr = OpenDate_parser(s)

    except:

        job_code = "NaN"

        openDate = "NaN"

        openStr = "NaN"

    salaryRange = SalaryRange_parser(s)

    expYrs = expYears_parser(s)

    duties = Duties_parser(s)

    course_length = EduSemDur_parser(qual)

    fullTimePartTime = fullTimePartTime_parser(qual)

    imp_qual_entities=Relations_parser([qual],nlp,"ORG")

    imp_qual_cardinals=Relations_parser([qual],nlp,"CARDINAL")

    imp_legal_terms=Relations_parser([s],nlp,"LAW")

    college = College_parser(qual)

    job_data_export.loc[i,"JOB_CLASS_TITLE"]=position

    job_data_export.loc[i,"FILE_NAME"]=file_names[i]

    job_data_export.loc[i,"DRIVERS_LICENSE_REQ"]=DL_valid

    job_data_export.loc[i,"EDUCATION_YEARS"]=education_yrs

    job_data_export.loc[i,"JOB_CLASS_NO"]=job_code

    job_data_export.loc[i,"OPEN_DATE"]=openDate

    job_data_export.loc[i,"ENTRY_SALARY_GEN"]=salaryRange

    job_data_export.loc[i,"JOB_DUTIES"]=duties

    job_data_export.loc[i,"EXPERIENCE_LENGTH"]=expYrs

    job_data_export.loc[i,"DRIV_LIC_TYPE"]=DL_state

    job_data_export.loc[i,"EDUCATION_MAJOR"]=education_major

    job_data_export.loc[i,"IMP_ENTITIES_QUAL"]=imp_qual_entities

    job_data_export.loc[i,"COURSE_LENGTH"]=course_length

    job_data_export.loc[i,"FULL_TIME_PART_TIME"]=fullTimePartTime

    job_data_export.loc[i,"SCHOOL_TYPE"]=college

    job_data_export.loc[i,"MISC_COURSE_DETAILS"]=imp_qual_cardinals

    job_data_export.loc[i,"LEGAL_TERMS"]=imp_legal_terms

    job_data_export.loc[i,"EXAM_TYPE"]=openStr
job_data_export.head()
job_data_export.to_csv("LA_job_class_export.csv",index=False)
job_data_export = pd.read_csv("../input/csv-file/LA_job_class_export.csv")

job_data_export.to_csv("LA_job_class_export.csv",index=False)
nltk.download('stopwords')
word_count = []

for s in job_data:

    word_count.append(len(str(s).split()))
fig, ax = plt.subplots()

data = np.random.rand(1000)



N, bins, patches = ax.hist(word_count, edgecolor='white', bins=50, linewidth=0)



q1 = 0.25*len(patches)



q3 = 0.75*len(patches)

for i in range(0,int(q1)):

    patches[i].set_facecolor('b')

for i in range(int(q1), int(q3)):

    patches[i].set_facecolor('g')

for i in range(int(q3), len(patches)):

    patches[i].set_facecolor('r')

plt.xlabel('Number of words')

plt.ylabel('Number of samples')

plt.title('Sample length distribution')

plt.show()
stats = pd.Series(word_count)

q = pd.DataFrame(stats.describe()[3:]).transpose()

q
!pip install textstat

import textstat

score_list = []

for text in job_data:

    score_list.append(textstat.flesch_reading_ease(str(text)))
readability=pd.DataFrame(job_data_export["FILE_NAME"])

readability.insert(1, "SCORE", score_list[:len(score_list)-1], True) 

readability.head(10)
rstats = readability["SCORE"].describe()[3:]

pd.DataFrame(rstats).transpose()
exclude = set(string.punctuation) 

wpt = nltk.WordPunctTokenizer()

stop_words = nltk.corpus.stopwords.words('english')

#

newStopWords = ['city','los','angele','angeles','may']

stop_words.extend(newStopWords)

table = str.maketrans('', '', string.punctuation)



lemma = WordNetLemmatizer()

porter = PorterStemmer()
def normalize_document(doc):

    #replace newline and tab chars

    doc = doc.replace("\\r\\n"," ").replace("\\\'s","").replace("\t"," ") #.split("b'")[1]

    # tokenize document

    tokens = doc.split()

    # remove punctuation from each word

    tokens = [w.translate(table) for w in tokens]

    # convert to lower case

    lower_tokens = [w.lower() for w in tokens]

    #remove spaces

    stripped = [w.strip() for w in lower_tokens]

    # remove remaining tokens that are not alphabetic

    words = [word for word in stripped if word.isalpha()]

    # filter stopwords out of document

    filtered_tokens = [token for token in words if token not in stop_words]

    #normalized = " ".join(lemma.lemmatize(word) for word in filtered_tokens)

    #join the tokens back to get the original doc

    doc = ' '.join(filtered_tokens)

    return doc



normalize_corpus = np.vectorize(normalize_document)

#apply the text normalization to list of job positions

norm_positions=[]

for text_sample in positions:

    norm_positions.append(normalize_document(str(text_sample)))

#apply the text normalization to list of job ads

norm_corpus=[]

for text_sample in job_data:

    norm_corpus.append(normalize_document(str(text_sample)))
full_norm_corpus=' '.join(norm_corpus)

stopwords = set(STOPWORDS)

stopwords.update(["class", "code"])



wordcloud = WordCloud(background_color='white', stopwords=stopwords,max_words=100).generate(full_norm_corpus)

plt.figure(figsize=(15,10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
def ngrams(sample_texts, ngram_range, num_ngrams=30):

    """Plots the frequency distribution of n-grams.



    # Arguments

        samples_texts: list, sample texts.

        ngram_range: tuple (min, mplt), The range of n-gram values to consider.

            Min and mplt are the lower and upper bound values for the range.

        num_ngrams: int, number of n-grams to plot.

            Top `num_ngrams` frequent n-grams will be plotted.

    """

    # Create args required for vectorizing.

    kwargs = {

            'ngram_range': ngram_range,

            'dtype': 'int32',

            'strip_accents': 'unicode',

            'decode_error': 'replace',

            'analyzer': 'word',  # Split text into word tokens.

    }

    vectorizer = CountVectorizer(**kwargs)

    vectorized_texts = vectorizer.fit_transform(sample_texts)



    # This is the list of all n-grams in the index order from the vocabulary.

    all_ngrams = list(vectorizer.get_feature_names())

    num_ngrams = min(num_ngrams, len(all_ngrams))

    ngrams = all_ngrams[:num_ngrams]



    # Add up the counts per n-gram ie. column-wise

    all_counts = vectorized_texts.sum(axis=0).tolist()[0]



    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.

    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(

        zip(all_counts, all_ngrams), reverse=True)])

    ngrams = list(all_ngrams)[:num_ngrams]

    counts = list(all_counts)[:num_ngrams]

    return ngrams, counts





ngrams4, counts4 = ngrams(norm_corpus,ngram_range=(4, 4))
# Fixing random state for reproducibility

idx = np.arange(30)

np.random.seed(19680801)

plt.rcdefaults()

fig, ax = plt.subplots()

#horizontal

ax.barh(idx, counts4, align='center', color='g')

ax.set_yticks(idx)

ax.set_yticklabels(ngrams4, rotation=0, fontsize=8)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Frequencies',fontsize="12")

ax.set_title('Frequency distribution of n-grams',fontsize="12")



plt.show()
from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

import nltk

#nltk.download('movie_reviews')

#nltk.download('punkt')



text          = "I only hire male applicants" 



sent          = TextBlob(text)

# The polarity score is a float within the range [-1.0, 1.0]

# where negative value indicates negative text and positive

# value indicates that the given text is positive.

polarity      = sent.sentiment.polarity

# The subjectivity is a float within the range [0.0, 1.0] where

# 0.0 is very objective and 1.0 is very subjective.

subjectivity  = sent.sentiment.subjectivity



sent          = TextBlob(text, analyzer = NaiveBayesAnalyzer())

classification= sent.sentiment.classification

positive      = sent.sentiment.p_pos

negative      = sent.sentiment.p_neg





dict1 = {'Polarity': polarity,'Subjectivity': subjectivity, 'Classification': classification, 'Posititve': positive, 'Negative': negative}

df1 = pd.Series(dict1)

df1

def pos_neg_classify(text):

    sent          = TextBlob(text)

    # The polarity score is a float within the range [-1.0, 1.0]

    # where negative value indicates negative text and positive

    # value indicates that the given text is positive.

    polarity      = sent.sentiment.polarity

        

    # The subjectivity is a float within the range [0.0, 1.0] where

    # 0.0 is very objective and 1.0 is very subjective.

    subjectivity  = sent.sentiment.subjectivity

    

    sent          = TextBlob(text, analyzer = NaiveBayesAnalyzer())

    classification= sent.sentiment.classification

    pos_score = round(sent.sentiment.p_pos,2)

    neg_score = round(sent.sentiment.p_neg,2)

    if pos_score > neg_score:

        clas = 'POSITIVE'

    elif pos_score < neg_score:

        clas = 'NEGATIVE'

    else:

        clas = 'NEUTRAL'

    return clas



def polarity_classify(text):

    sent          = TextBlob(text)

    # The polarity score is a float within the range [-1.0, 1.0]

    # where negative value indicates negative text and positive

    # value indicates that the given text is positive.

    polarity      = round(sent.sentiment.polarity, 1)

    

    if polarity > 0:

        pol = 'EMOTIONAL POSITIVE'

    elif polarity < 0:

        pol = 'EMOTIONAL NEGATIVE'

    else:

        pol = 'EMOTIONAL NEUTRAL'

        

    return pol
pd.set_option('display.max_colwidth', -1)

df_4 = pd.DataFrame(ngrams4, columns=['N-grams Sentence'])

class_list4 = []

pol_list4 = []

for text in ngrams4:

    classification = pos_neg_classify(text)

    polarity = polarity_classify(text)

    class_list4.append(classification)

    pol_list4.append(polarity)

df_4["Content"] = class_list4

df_4["Polarity"] = pol_list4

df_4.to_csv("ngrams4sentiment.csv",index=False)

df_4
df_4_neg = df_4[df_4["Content"]=="NEGATIVE"]

df_4_neg.to_csv("negative.csv",index=False)

df_4_neg
neg_4grams = df_4_neg["N-grams Sentence"]

neg_list = []

class_list10 = []

pol_list10 = []

ngrams10, counts10 = ngrams(norm_corpus,ngram_range=(10, 10))   

for text in ngrams10:

    for neg_word in neg_4grams:

        neg_index = text.find(neg_word)    

        if neg_index != -1:

            if text not in neg_list:

                neg_list.append(text)

                classification = pos_neg_classify(text)

                polarity = polarity_classify(text)

                class_list10.append(classification)

                pol_list10.append(polarity)

                

df_neg = pd.DataFrame(neg_list, columns=['N-grams Sentence'])

df_neg["Content"] = class_list10

df_neg["Polarity"] = pol_list10

df_neg.to_csv("Ngrams30sentiment.csv",index=False)

df_neg
old_jobposting = open('../input/sample-text/ADMINISTRATIVE ANALYST 1590 060118.txt', 'r')

content1 = old_jobposting.read()
!pip install textstat
import textstat

print("Statistics of Revised Job Posting:")

numword = textstat.lexicon_count(content1, removepunct=True)

score = textstat.flesch_reading_ease(content1)

class1 = pos_neg_classify(content1)

pol1         = polarity_classify(content1)

contentstat1 = pd.Series({"Word counts": numword, "Readbility Score": score, "Content": class1, "Polarity": pol1})

contentstat1
revised_jobposting = open('../input/revised-text-posting/Revised posting.txt', 'rb')

content2 = str(revised_jobposting.read()).replace("\\r\\n"," ").replace("\\\'s","")

print(content2)
print("Statistics of Revised Job Posting:")

numword = textstat.lexicon_count(content2, removepunct=True)

score = textstat.flesch_reading_ease(content2)

class2 = pos_neg_classify(content2)

pol2         = polarity_classify(content2)

contentstat2 = pd.Series({"Word counts": numword, "Readbility Score": score, "Content": class2, "Polarity": pol2})

contentstat2