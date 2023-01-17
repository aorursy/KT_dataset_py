!pip install word2number

!pip install textstat

!pip install syllables



# Import python packages

import os, sys

import pandas as pd,numpy as np

import re

import spacy

from os import walk

import shutil

from shutil import copytree, ignore_patterns

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

import xml.etree.cElementTree as ET

from collections import OrderedDict

import json

from __future__ import unicode_literals, print_function

import plac

import random

from pathlib import Path

from spacy.util import minibatch, compounding

from spacy.matcher import Matcher

from word2number import w2n

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from datetime import date

import calendar

from sklearn.feature_extraction.text import CountVectorizer

from itertools import takewhile, tee

import itertools

import nltk, string

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.cluster.util import cosine_distance

import networkx as nx

from PIL import Image,ImageFilter

import textstat

from textstat.textstat import textstatistics, easy_word_set, legacy_round 

import syllables

from IPython.display import display, HTML, Javascript



bulletin_dir = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins"

additional_data_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data'

STOP_WORDS = stopwords.words('english')



%matplotlib inline
data_dictionary = pd.read_csv(os.path.join(additional_data_dir,"kaggle_data_dictionary.csv"))

jobs_list = []

for file_name in os.listdir(bulletin_dir):

    with open(os.path.join(bulletin_dir,file_name), encoding = "ISO-8859-1") as f:

        content = f.read()

        jobs_list.append([file_name, content])

jobs_df = pd.DataFrame(jobs_list)

jobs_df.columns = ["FileName", "Content"]



job_titles = pd.read_csv(additional_data_dir+'/job_titles.csv', header=None)



job_titles_str = ','.join(job_titles[0])



configfile = r'''

<Config-Specifications>

<Term name="Requirements">

        <Method name="section_value_extractor" section="RequirementSection">

            <SchoolType>College or University,High School,Apprenticeship</SchoolType>

            <JobTitle>'''+job_titles_str+'''</JobTitle>

        </Method>

    </Term>

</Config-Specifications>

'''
print(jobs_df['Content'].values[0][:1000])
print(jobs_df['Content'].values[1][:1000])
def createDataFrameAndDictionary(jobs_df, job_titles, data_dictionary):

    headingsFrame = createHeadingsFrame()

    jobClassDetailsFrame = createJobClassDetailsFrame()

    requirementsFrame = createRequirementsFrame(headingsFrame)

    df_salary_dwp,df_salary_gen = createSalaryFrame(headingsFrame)

    

    print('Adding EDUCATION_MAJOR')

    requirementsFrame['EDUCATION_MAJOR']=requirementsFrame.apply(getEducationMajor, axis=1)

    requirementsFrame['EDUCATION_MAJOR']=requirementsFrame.apply(getApprenticeshipMajor, axis=1)



    print('Adding JOB_DUTIES')

    df_duties = getValues('duties','JOB_DUTIES', headingsFrame)

    

    requirementsFrame = addColumnsFromConfig(df_requirements=requirementsFrame)

    

    requirementsFrame['EDUCATION_YEARS'] = requirementsFrame.apply(getEducationYears, axis=1)

    print('Adding experience details...')

    requirementsFrame['REQUIREMENT_TEXT_WITH_K1'] = requirementsFrame.apply(getRequirementTextWithK1, axis=1)

    requirementsFrame['EXPERIENCE_TEXT'] = requirementsFrame.apply(getExperienceText, axis=1)

    requirementsFrame['REQUIREMENT_WITHOUT_EXPERIENCE_TEXT'] = requirementsFrame.apply(getTextWithoutExperienceText, axis=1)

    requirementsFrame['EXP_JOB_CLASS_ALT_RESP'] = requirementsFrame.apply(lambda x : getJobFunction(x,'K1b'), axis=1)

    requirementsFrame['EXP_JOB_CLASS_FUNCTION'] = requirementsFrame.apply(lambda x : getJobFunction(x,'K2'), axis=1)

    requirementsFrame['EXPERIENCE_LENGTH'],requirementsFrame['EXPERIENCE_LEN_UNIT'] = zip(*requirementsFrame.apply(getExperienceLength, axis=1))

    requirementsFrame['FULL_TIME_PART_TIME'],requirementsFrame['PAID_VOLUNTEER'] = zip(*requirementsFrame.apply(getExperienceType, axis=1))

    

    requirementsFrame.loc[requirementsFrame['EXPERIENCE_LEN_UNIT']=='year','EXPERIENCE_LEN_UNIT']='years'

    requirementsFrame.loc[requirementsFrame['EXPERIENCE_LEN_UNIT']=='month','EXPERIENCE_LEN_UNIT']='months'



    print('Adding course details...')

    requirementsFrame['COURSE_COUNT'] = requirementsFrame.apply(getCourseCount, axis=1)

    requirementsFrame['COURSE_LENGTH'],requirementsFrame['COURSE_LENGTH_TEXT'] = zip(*requirementsFrame.apply(getCourseLength, axis=1))

    requirementsFrame['COURSE_SUBJECT'] = requirementsFrame.apply(getCourseSubjects, axis=1)

    requirementsFrame['MISC_COURSE_DETAILS']= requirementsFrame.apply(lambda x: x['REQUIREMENT_TEXT'] if x['COURSE_LENGTH']=='' and 'course' in x['REQUIREMENT_TEXT'] else '', axis=1)

    

    print('Adding license details...')

    processNotesFrame = getValues('process note', 'PROCESS_NOTES', headingsFrame)

    requirementsFrame = requirementsFrame.merge(right=processNotesFrame,how='left',left_on='FILE_NAME',right_on='FILE_NAME')

    requirementsFrame['DRIVERS_LICENSE_REQ']=''

    requirementsFrame['DRIV_LIC_TYPE']=''

    requirementsFrame['DRIVERS_LICENSE_REQ'],requirementsFrame['DRIV_LIC_TYPE'] = zip(*requirementsFrame.apply(

        lambda x : getLicenseRequired(x, 'PROCESS_NOTES'), axis=1))

    requirementsFrame['DRIVERS_LICENSE_REQ'],requirementsFrame['DRIV_LIC_TYPE'] = zip(*requirementsFrame.apply(

        lambda x : getLicenseRequired(x, 'REQUIREMENT_TEXT'), axis=1))

    requirementsFrame['ADDTL_LIC'] = requirementsFrame.apply(lambda x: getAdditionalLic(x,'PROCESS_NOTES'), axis=1)

    

    jobs_df['ExamTypeContent'] = jobs_df.apply(getExamTypeContent, axis=1)

    jobs_df['EXAM_TYPE'] = jobs_df.apply(setExamTypeContent, axis=1)

    jobs_df.loc[jobs_df['EXAM_TYPE']=='','EXAM_TYPE']='OPEN'



    appDeadlineFrame = getValues('deadline', 'APPLICATION_DEADLINE_TEXT', headingsFrame)

    selectionProcessFrame = getValues(searchText='selection', COL_NAME='SELECTION_PROCESS', headingsFrame=headingsFrame)

    whereToApplyFrame = getValues(COL_NAME='WHERE_TO_APPLY',searchText='where to', headingsFrame=headingsFrame)



    appDeadlineFrame['APPLICATION_DEADLINE'] = appDeadlineFrame.apply(getAppDeadlineDate, axis=1)

    

    print('Adding selection criteria details...')

    selectionProcessFrame['SELECTION_CRITERIA'] = selectionProcessFrame['SELECTION_PROCESS'].apply(getSelection)

    whereToApplyFrame['WHERE_TO_APPLY'] = whereToApplyFrame['WHERE_TO_APPLY'].apply(getWhereToApplyUrl)

    

    #Merge all frames to create a single resultset for final submission

    job_class_df = pd.merge(jobClassDetailsFrame, requirementsFrame, how='inner', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, df_salary_dwp, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, df_salary_gen, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, df_duties, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, jobs_df, how='left', left_on='FILE_NAME', right_on='FileName', sort=True)

    job_class_df = pd.merge(job_class_df, appDeadlineFrame, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, selectionProcessFrame, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)

    job_class_df = pd.merge(job_class_df, whereToApplyFrame, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)



    job_class_df.drop(columns=['FileName', 'Content', 'ExamTypeContent','REQUIREMENT_TEXT_WITH_K1', 'EXPERIENCE_TEXT',

                    'REQUIREMENT_WITHOUT_EXPERIENCE_TEXT','COURSE_LENGTH_TEXT','REQUIREMENT_TEXT','PROCESS_NOTES',

                    'APPLICATION_DEADLINE_TEXT','SELECTION_PROCESS'],inplace=True)

    

    print('Updating data dictionary...\n')

    

    #Update allowed values for DRIV_LIC_TYPE

    data_dictionary.loc[data_dictionary['Field Name']=='DRIV_LIC_TYPE','Allowable Values'] = 'A,B,C,I'



    data_dictionary.loc[-1] = ['EXPERIENCE_LEN_UNIT','','The unit of experience length(e.g hours, years)','String','','Yes','']

    data_dictionary.index = data_dictionary.index+1



    data_dictionary.loc[-1] = ['PAID_VOLUNTEER','','Whether the required experience is paid or volunteer or both.','String','PAID, VOLUNTEER','Yes','In case if both allowed, PAID|VOLUNTEER is used.']

    data_dictionary.index = data_dictionary.index+1



    data_dictionary.loc[-1] = ['APPLICATION_DEADLINE','','Date on which applications will be closed.','String','','Yes','']

    data_dictionary.index = data_dictionary.index+1



    data_dictionary.loc[-1] = ['SELECTION_CRITERIA','','Selection Criteria','String','','Yes','Some classes have multiple selection criteria, in that case criteria are separated by |.']

    data_dictionary.index = data_dictionary.index+1



    data_dictionary.loc[-1] = ['REQUIREMENT_CONJ','','Requirement/subrequirement conjuction','String','','Yes','If requirement_subset_id is blank then this will represent conjuction value for requirement else for sub requirement.']

    data_dictionary.index = data_dictionary.index+1



    data_dictionary.loc[-1] = ['WHERE_TO_APPLY','','Online Url for job application','String','','Yes','Any available url in where to apply section of the job class']

    data_dictionary.index = data_dictionary.index+1

    data_dictionary = data_dictionary.sort_index()



    data_dictionary.loc[data_dictionary['Field Name']=='REQUIREMENT_SUBSET_ID','Accepts Null Values?'] = 'Yes'

    data_dictionary.loc[data_dictionary['Field Name']=='REQUIREMENT_SET_ID','Accepts Null Values?'] = 'Yes'

    data_dictionary.loc[data_dictionary['Field Name']=='JOB_CLASS_NO','Accepts Null Values?'] = 'Yes'



    print('Final list of columns:\n')

    print(job_class_df.columns)

    return job_class_df, data_dictionary
def createHeadingsFrame():

    headings = {}

    for filename in os.listdir(bulletin_dir):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            for line in f.readlines():

                line = line.replace("\n"," ").replace("\t"," ").replace(":","").replace("  "," ").strip()



                if line.isupper():

                    if line not in headings.keys():

                        headings[line] = 1

                    else:

                        count = int(headings[line])

                        headings[line] = count+1



    del headings['$103,606 TO $151,484'] #This is not a heading, it's an Annual Salary component

    headingsFrame = []

    for i,j in (sorted(headings.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)):

        headingsFrame.append([i,j])

    headingsFrame = pd.DataFrame(headingsFrame)

    headingsFrame.columns = ["Heading","Count"]

    return headingsFrame



def createJobClassDetailsFrame():

    print('Adding FILE_NAME, JOB_CLASS_TITLE, JOB_CLASS_NO ,OPEN_DATE')

    data_list = []

    for filename in os.listdir(bulletin_dir):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            job_class_title = ''

            job_class_no=''

            job_bulletin_date=''

            for line in f.readlines():

                #Insert code to parse job bulletins

                if "Open Date:" in line and job_bulletin_date=='':

                    job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

                    try:

                        job_bulletin_date = pd.to_datetime(job_bulletin_date, format="%m-%d-%y")

                    except:

                        job_bulletin_date = pd.to_datetime(job_bulletin_date, format="%m-%d-%Y")

                elif "Open date:" in line and job_bulletin_date=='':

                    job_bulletin_date = line.split("Open date:")[1].split("(")[0].strip()

                    try:

                        job_bulletin_date = pd.to_datetime(job_bulletin_date, format="%m-%d-%y")

                    except:

                        job_bulletin_date = pd.to_datetime(job_bulletin_date, format="%m-%d-%Y")

                if "Class Code:" in line and job_class_no=='':

                    job_class_no = line.split("Class Code:")[1].strip()[:4]

                elif "Class  Code:" in line and job_class_no=='':

                    job_class_no = line.split("Class  Code:")[1].strip()[:4]

                if len(job_class_title)<2 and len(line.strip())>1 and line.strip() != 'CAMPUS INTERVIEWS ONLY':

                    if 'Class Code:' in line:

                        job_class_title = line.split("Class Code:")[0].strip()

                    else:

                        job_class_title = line.strip()

            data_list.append([filename, job_bulletin_date, job_class_title, job_class_no])



    df = pd.DataFrame(data_list)

    df.columns = ["FILE_NAME", "OPEN_DATE", "JOB_CLASS_TITLE", "JOB_CLASS_NO"]

    return df



def getRequirements(headingsFrame, COL_NAME='REQUIREMENT_TEXT'):

    data_list = []

    dataHeadings = [k for k in headingsFrame['Heading'].values if 'requirement' in k.lower()]



    for filename in os.listdir(bulletin_dir):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            readNext = 0 

            datatxt = ''

            for line in f.readlines():

                clean_line = line.replace("\n"," ").replace("\t"," ").replace(":","").replace("  "," ").strip()

                if readNext == 0:                         

                    if clean_line in dataHeadings:

                        readNext = 1

                elif readNext == 1:

                    if clean_line in headingsFrame['Heading'].values:

                        break

                    else:

                         datatxt = datatxt + ' ' + line

            data_list.append([filename,datatxt.strip()])

    result = pd.DataFrame(data_list)

    result.columns = ['FILE_NAME',COL_NAME]

    return result



def getRequirement_Conj(txt):

    """Search for conjuction in the end of the passed text value

    

    Parameters

    ----------

    txt : str

        text to be searched for conjuction

    """

    rqmnt_conj = ''

    allowed_conjs = ['or','and']



    txt = txt.translate(str.maketrans('', '', string.punctuation))

    li = list(map(str.strip,list(filter(None,txt.split()))))

    if len(li)>0:

        rqmnt_conj = li[-1].lower()

    if rqmnt_conj not in allowed_conjs:

        rqmnt_conj = ''

    return rqmnt_conj



def createRequirementsFrame(headingsFrame):

    print('Adding REQUIREMENT_SET_ID,REQUIREMENT_SUBSET_ID,REQUIREMENT_TEXT,REQUIREMENT_CONJ')

    requirements = getRequirements(headingsFrame)



    requirementsAll = []

    for index,row in requirements.iterrows():

        filename = row['FILE_NAME']

        rqmntTxt=requirements[requirements['FILE_NAME']==filename]['REQUIREMENT_TEXT'].values

        lines = rqmntTxt[0].split('\n')

        rid = ''

        rsid = ''

        rlineTxt = ''

        rslineTxt = ''



        for line in lines:

            l = line.strip()

            arr = re.split('\. |\)', l.replace("\n"," ").replace("\t"," ").replace(":","").replace("  "," ").strip())

            if l.replace("\n"," ").replace("\t"," ").strip() == '':

                continue



            if arr[0].isdigit():

                if rlineTxt != '':

                    if rslineTxt != '':

                        requirementsAll.append([filename,rid,rsid,rlineTxt + ' - ' + rslineTxt])

                    else:

                        requirementsAll.append([filename,rid,rsid,rlineTxt])

                rid = arr[0]

                rlineTxt = l

                rsid = ''

                rslineTxt = ''

            elif re.match('^[a-z]$',arr[0]):

                if rlineTxt != '' and rslineTxt != '':

                    if rid == '':

                        rid = '1'

                    requirementsAll.append([filename,rid,rsid,rlineTxt + ' - ' + rslineTxt])

                rsid = arr[0]

                rslineTxt = l

            else:

                if rsid == '':

                    rlineTxt = rlineTxt + ' ' + l

                else:

                    rslineTxt = rslineTxt + ' ' + l

                continue

        if rid == '':

            rid = '1'

        if rslineTxt != '':

            requirementsAll.append([filename,rid,rsid,rlineTxt + ' - ' + rslineTxt])

        else:

            requirementsAll.append([filename,rid,rsid,rlineTxt])



    df_requirements = pd.DataFrame(requirementsAll)

    df_requirements.columns = ['FILE_NAME','REQUIREMENT_SET_ID','REQUIREMENT_SUBSET_ID','REQUIREMENT_TEXT']



    df_requirements['REQUIREMENT_CONJ'] = df_requirements['REQUIREMENT_TEXT'].apply(getRequirement_Conj)

    

    return df_requirements



def createSalaryFrame(headingsFrame):

    #Check for salary components

    salHeadings = [k for k in headingsFrame['Heading'].values if 'salary' in k.lower()]

    sal_list = []

    files = []

    for filename in os.listdir(bulletin_dir):

        files.append(filename)

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            readNext = 0

            for line in f.readlines():

                clean_line = line.replace("\n"," ").replace("\t"," ").replace(":","").replace("  "," ").strip()

                if clean_line in salHeadings:

                    readNext = 1

                elif readNext == 1:

                    if clean_line in headingsFrame['Heading'].values:

                        break

                    elif len(clean_line)<2:

                        continue

                    else:

                        sal_list.append([filename, clean_line])



    df_salary = pd.DataFrame(sal_list)

    df_salary.columns = ['FILE_NAME','SALARY_TEXT']      



    #Adding 'ENTRY_SALARY_GEN','ENTRY_SALARY_DWP'

    pattern = r'\$?([0-9]{1,3},([0-9]{3},)*[0-9]{3}|[0-9]+)(.[0-9][0-9])?'

    dwp_salary_list = {}

    gen_salary_list = {}

    for filename in files:

        for sal_text in df_salary.loc[df_salary['FILE_NAME']==filename]['SALARY_TEXT']:

            if 'department of water' in sal_text.lower():

                if filename in dwp_salary_list.keys():

                    continue

                matches = re.findall(pattern+' to '+pattern, sal_text) 

                if len(matches)>0:

                    salary_dwp = ' - '.join([x for x in matches[0] if x and not x.endswith(',')])

                else:

                    matches = re.findall(pattern, sal_text)

                    if len(matches)>0:

                        salary_dwp = matches[0][0]

                    else:

                        salary_dwp = ''

                dwp_salary_list[filename]= salary_dwp

            else:

                if filename in gen_salary_list.keys():

                    continue

                matches = re.findall(pattern+' to '+pattern, sal_text)

                if len(matches)>0:

                    salary_gen = ' - '.join([x for x in matches[0] if x and not x.endswith(',')])

                else:

                    matches = re.findall(pattern, sal_text)

                    if len(matches)>0:

                        salary_gen = matches[0][0]

                    else:

                        salary_gen = ''

                if len(salary_gen)>1:

                    gen_salary_list[filename]= salary_gen





    df_salary_dwp = pd.DataFrame(list(dwp_salary_list.items()), columns=['FILE_NAME','ENTRY_SALARY_DWP'])

    df_salary_gen = pd.DataFrame(list(gen_salary_list.items()), columns=['FILE_NAME','ENTRY_SALARY_GEN'])

    return df_salary_dwp,df_salary_gen
#Idea here is to first create part of speech tags

#And then find Noun/Pronoun tags following the words like majoring/major/apprenticeship



def preprocess(txt):

    txt = nltk.word_tokenize(txt)

    txt = nltk.pos_tag(txt)

    return txt



def getEducationMajor(row):

    """Returns education majors from the text supplied in the DataFrame row"""

    

    txt = row['REQUIREMENT_TEXT']

    txtMajor = ''

    if 'major in' not in txt.lower() and ' majoring ' not in txt.lower():

        return txtMajor

    result = []

    

    istart = txt.lower().find(' major in ')

    if istart!=-1:

        txt = txt[istart+10:]

    else:

        istart = txt.lower().find(' majoring ')

        if istart==-1:

            return txtMajor

        txt = txt[istart+12:]

    

    txt = txt.replace(',',' or ').replace(' and/or ',' or ').replace(' a closely related field',' related field').replace(' a ',' ').replace(' an ',' ')

    sent = preprocess(txt)

    pattern = """

            NP: {<DT>? <JJ>* <NN.*>*}

           BR: {<W.*>|<V.*>} 

        """

    cp = nltk.RegexpParser(pattern)

    cs = cp.parse(sent)

    #print(cs)

    checkNext = 0

    for subtree in cs.subtrees():

        if subtree.label()=='NP':

            result.append(' '.join([w for w, t in subtree.leaves()]))

            checkNext=1

        elif checkNext==1 and subtree.label()=='BR':

            break

    return '|'.join(result)



def getApprenticeshipMajor(row):

    """Returns education majors from the text supplied in the DataFrame row

    

    Checks for apprenticeship in the text and will process only if Education major is blank.

    """

    

    txt = row['REQUIREMENT_TEXT']

    txtMajor = row['EDUCATION_MAJOR']

    if 'apprenticeship' not in txt:

        return txtMajor

    if txtMajor != '':

        return txtMajor

    result = []

    

    istart = txt.lower().find(' apprenticeship program')

    if istart!=-1:

        txt = txt[istart+23:]

    else:

        istart = txt.lower().find(' apprenticeship ')

        if istart==-1:

            return txtMajor

        txt = txt[istart+15:]

    

    txt = txt.replace(',',' or ').replace(' full-time ',' ').replace(' a ',' ').replace(' an ',' ')

    sent = preprocess(txt)

    pattern = """

            NP: {<DT>? <JJ>* <NN.*>*}

           BR: {<W.*>|<V.*>} 

        """

    cp = nltk.RegexpParser(pattern)

    cs = cp.parse(sent)

    #print(cs)

    checkNext = 0

    for subtree in cs.subtrees():

        if subtree.label()=='NP':

            result.append(' '.join([w for w, t in subtree.leaves()]))

            checkNext=1

        elif checkNext==1 and subtree.label()=='BR':

            break

    return '|'.join(result)





def getValues(searchText, COL_NAME, headingsFrame):

    """Search for possible values for searchText in headings from job class file, and returns the content related to the heading

    

    Parameters

    ----------

    searchText : str

        text value to be searched for in possible Headings list

    COL_NAME : str

        name of the column to represent the content in the resulting dataframe

    

    Returns

    -------

    DataFrame

        dataframe with two columns ['FILE_NAME',COL_NAME].

        FILE_NAME - name of the job class file

        COL_NAME - content for the heading searched (searchText)

    """

    

    data_list = []

    dataHeadings = [k for k in headingsFrame['Heading'].values if searchText in k.lower()]



    for filename in os.listdir(bulletin_dir):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            readNext = 0 

            datatxt = ''

            for line in f.readlines():

                clean_line = line.replace("\n"," ").replace("\t"," ").replace(":","").replace("  "," ").strip()

                if readNext == 0:                         

                    if clean_line in dataHeadings:

                        readNext = 1

                elif readNext == 1:

                    if clean_line in headingsFrame['Heading'].values:

                        break

                    else:

                         datatxt = datatxt + ' ' + clean_line

            data_list.append([filename,datatxt.strip()])

    result = pd.DataFrame(data_list)

    result.columns = ['FILE_NAME',COL_NAME]

    return result



#Functions to find out text from a set of pre-defined list

#e.g. To search for 'High School' or 'Apprenticeship' or any job class title

def section_value_extractor( document, section, subterms_dict, parsed_items_dict ):

    retval = OrderedDict()

    single_section_lines = document.lower()

    

    for node_tag, pattern_string in subterms_dict.items():

        pattern_list = re.split(r",|:", pattern_string[0])#.sort(key=len)

        pattern_list=sorted(pattern_list, key=len, reverse=True)

        #print (pattern_list)

        matches=[]

        for pattern in pattern_list:

            if pattern.lower() in single_section_lines:

                matches.append(pattern)

                single_section_lines = single_section_lines.replace(pattern.lower(),'')

                #print (single_section_lines)

        #matches = [pattern for pattern in pattern_list if pattern.lower() in single_section_lines.lower()]

        #print (matches)

        if len(matches):

            info_string = ", ".join(list(matches)) + " "

            retval[node_tag] = info_string

    return retval



def read_config( configfile ):



    #tree = ET.parse(configfile) #Use this if configfile is path for xml

    tree = ET.fromstring(configfile)

    #root = tree.getroot()

    root = tree

    config = []

    for child in root:

        term = OrderedDict()

        term["Term"] = child.get('name', "")

        for level1 in child:

            term["Method"] = level1.get('name', "")

            term["Section"] = level1.get('section', "")

            for level2 in level1:

                term[level2.tag] = term.get(level2.tag, []) + [level2.text]



        config.append(term)

    json_result = json.dumps(config, indent=4)

    #print("Specifications:\n {}".format(json_result))

    return config



def parse_document(document, config):

    parsed_items_dict = OrderedDict()



    for term in config:

        term_name = term.get('Term')

        extraction_method = term.get('Method')

        extraction_method_ref = globals()[extraction_method]

        section = term.get("Section")

        subterms_dict = OrderedDict()

        

        for node_tag, pattern_list in list(term.items())[3:]:

            subterms_dict[node_tag] = pattern_list

        parsed_items_dict[term_name] = extraction_method_ref(document, section, subterms_dict, parsed_items_dict)



    return parsed_items_dict



def addColumnsFromConfig(df_requirements):

    print('Adding columns from config.')

    config = read_config(configfile.replace('&','&#38;').replace('\'','&#39;'))

    result = df_requirements['REQUIREMENT_TEXT'].apply(lambda k: parse_document(k,config))



    i=0

    df_requirements['EXP_JOB_CLASS_TITLE']=''

    df_requirements['SCHOOL_TYPE']=''

    for item in (result.values):

        for requirement,dic in list(item.items()):        

            if 'JobTitle' in dic.keys():

                df_requirements.loc[i,'EXP_JOB_CLASS_TITLE'] = dic['JobTitle']

            if 'SchoolType' in dic.keys():

                df_requirements.loc[i,'SCHOOL_TYPE'] = dic['SchoolType']

        i=i+1

    return df_requirements
#Functions used for finding education & experience details for a job class



def getEducationYears(row):

    txt = row['REQUIREMENT_TEXT'].lower()

    txtSchoolType = row['SCHOOL_TYPE'].strip().lower()

    

    if txtSchoolType == '':

        return ''

    

    txt = txt.replace(txtSchoolType,txtSchoolType.replace(' ','_'))

    txtSchoolType = txtSchoolType.replace(' ','_')



    doc = nlp(txt)

    matchResult = ''

    pattern = [{'POS': 'NUM'},

               {'TEXT': {'IN':['-','year','years','month','months','of']},'OP': '*'},

               {'TEXT': txtSchoolType}

               ]

    matcher = Matcher(nlp.vocab)



    matcher.add("EducationYears", None, pattern)

    

    matches = matcher(doc)

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        matchResult = span.text

        break

    if matchResult!='':

        matchResult = w2n.word_to_num(matchResult.split()[0])

    

    return matchResult



def getRequirementTextWithK1(row):

    rqmntTxt = row['REQUIREMENT_TEXT']

    jobClass = row['EXP_JOB_CLASS_TITLE']

    rqmntTxt = rqmntTxt.replace('fulltime','full-time').replace('.','. ').replace('#','').replace('  ',' ')

    if jobClass == '':

        return rqmntTxt

    else:

        jobClass = jobClass.split(',')

        for job in jobClass:

            job = job.strip()

            rqmntTxt = rqmntTxt.lower().replace(job.lower(),'K1 ')

    return rqmntTxt.replace('  ',' ')



def getExperienceText(row):

    txt = row['REQUIREMENT_TEXT_WITH_K1'].lower()

    matcher = Matcher(nlp.vocab)

    doc = nlp(txt)



    pattern2 = [{'POS': 'NUM'},

               {'POS': 'ADJ', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': 'NUM', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': 'NOUN'},

               {'IS_STOP': True, 'OP': '?'},

               {'LEMMA': {'IN': ['full','part']}, 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'LEMMA': 'time', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': 'NUM', 'OP': '*'},

               {'LEMMA': 'pay', 'OP': '?'},

               {'IS_STOP': True, 'OP': '?'},

               {'LEMMA': 'volunteer', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': {'IN': ['NOUN','ADJ','VERB']}, 'OP': '*'},

               {'ORTH': 'k1', 'OP': '?'},

               {'ORTH': 'experience'}]

    

    pattern3 = [{'LEMMA': {'IN': ['full','part']}},

               {'IS_PUNCT': True, 'OP': '?'},

               {'LEMMA': 'time', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'LEMMA': 'pay', 'OP': '?'},

               {'IS_STOP': True, 'OP': '?'},

               {'LEMMA': 'volunteer', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': {'IN': ['NOUN','ADJ','VERB']}, 'OP': '*'},

               {'ORTH': 'k1', 'OP': '?'},

               {'ORTH': 'experience'}]

    

    #matcher.add("Experience1", None, pattern1)

    matcher.add("Experience2", None, pattern2)

    matcher.add("Experience3", None, pattern3)

    

    matches = matcher(doc)

    i=0

    result = ''

    docs = {}

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end] 

        if string_id not in docs.keys():

            docs[string_id] = span.text

        if ')' in span.text and i==0:

            i=1

            del docs[string_id]



    if 'Experience2' in docs.keys():

        result = docs['Experience2']

    elif 'Experience3' in docs.keys():

        result = docs['Experience3']

        

    return result



def getTextWithoutExperienceText(row):

    rqmntTxt = row['REQUIREMENT_TEXT_WITH_K1'].lower()

    expTxt = row['EXPERIENCE_TEXT'].lower()

    if expTxt == '':

        return rqmntTxt

    istart = rqmntTxt.find(expTxt)

    istart = istart + len(expTxt)

    return rqmntTxt[istart:]



def getJobClassFunction(row, txt):

    doc = nlp(txt)

    verbTxt = ''

    sentence = ''

    for sent in doc.sents:

        sentence = sent.text

        for token in sent:

            if token.pos_ == 'VERB' and not token.is_stop:

                verbTxt = token.text

                break

        break

    if verbTxt == '':

        return ''

    istart = sentence.find(verbTxt)

    return sentence[istart:]



def getJobFunction(row, col):

    if row['EXPERIENCE_TEXT'] == '':

        return ''

    txt = row['REQUIREMENT_TEXT_WITH_K1'].lower()

    doc = nlp(txt)

    pattern = [{'ORTH': 'k1'},

               {'IS_STOP': True, 'OP': '*'},

               {'IS_PUNCT': True, 'OP': '*'},

               {'POS': 'NOUN', 'OP': '*'},

               {'ORTH': 'or'},

               {'IS_STOP': True, 'OP': '*'},

               {'ORTH': 'class', 'OP': '?'},

               {'IS_STOP': True, 'OP': '*'},

               {'ORTH': 'level'}]



    matcher = Matcher(nlp.vocab)



    matcher.add("K1Class", None, pattern)



    matches = matcher(doc)

    result = ''

    

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end] 

        result = span.text

    if result != '':

        if col == 'K1b':

            istart = txt.find(result)

            istart = istart + len(result)

            result = txt[istart:]

            return getJobClassFunction(row, result) #K1b

        else:

            return ''

    elif col=='K2':

        txt = row['REQUIREMENT_WITHOUT_EXPERIENCE_TEXT'].lower()

        return getJobClassFunction(row, txt) #K2

    else:

        return ''

    

def getExperienceLength(row):

    expTxt = row['EXPERIENCE_TEXT'].lower()

    if expTxt == '':

        return expTxt,expTxt

    matcher = Matcher(nlp.vocab)

    doc = nlp(expTxt)



    patExpLength = [{'POS': 'NUM'},

               {'POS': 'ADJ', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': 'NUM', 'OP': '?'},

               {'IS_PUNCT': True, 'OP': '?'},

               {'POS': 'NOUN'},

               ]

    

    matcher.add("ExperienceLength", None, patExpLength)

    

    matches = matcher(doc)

    matchResult=''

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        matchResult = span.text

        break

    doc = nlp(matchResult)

    expLength = ''

    expUnit = ''

    for token in doc:

        if token.pos_ == 'NUM':

            print

            expLength = w2n.word_to_num(token.text.replace(',',''))

        if token.pos_ == 'NOUN':

            expUnit = token.text

    return expLength,expUnit



def getExperienceType(row):

    expTxt = row['EXPERIENCE_TEXT'].lower().replace('fulltime','full time').replace('-',' ')

    if expTxt == '':

        return expTxt,expTxt

    paid_volu = ''

    full_part = ''

    if 'paid' in expTxt and 'volunteer' in expTxt:

        paid_volu = 'PAID|VOLUNTEER'

    elif 'paid' in expTxt:

        paid_volu = 'PAID'

    elif 'volunteer' in expTxt:

        paid_volu = 'VOLUNTEER'

    

    if 'full time' in expTxt and 'part time' in expTxt:

        full_part = 'FULL_TIME|PART_TIME'

    elif 'full time' in expTxt:

        full_part = 'FULL_TIME'

    elif 'part time' in expTxt:

        full_part = 'PART_TIME'

    

    return full_part,paid_volu
def getCourseCount(row):

    txt = row['REQUIREMENT_TEXT']

    doc = nlp(txt.lower())

    matchResult = ''

    pattern = [{'POS': 'NUM'},

                {'LEMMA': 'course'}

               ]

    matcher = Matcher(nlp.vocab)



    matcher.add("CourseCount", None, pattern)

    

    matches = matcher(doc)

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        matchResult = span.text

        break

    if matchResult!='':

        matchResult = w2n.word_to_num(matchResult.split()[0])

    

    return matchResult



def getCourseLength(row):

    txt = row['REQUIREMENT_TEXT']

    txt = txt.replace(')',') ').replace('-',' ').replace('  ',' ')

    doc = nlp(txt.lower())

    matchResult = ''

    pattern = [{'POS': 'NUM', 'OP': '+'},

               {'ORTH': '(', 'OP': '?'},

               {'POS': 'NUM', 'OP': '?'},

               {'ORTH': ')', 'OP': '?'},

               {'LEMMA': 'semester', 'OP': '?'},

               {'LEMMA': 'unit', 'OP': '?'},

               {'POS': 'CCONJ', 'OP': '?'},

               {'POS': 'NUM', 'OP': '*'},

               {'ORTH': '(', 'OP': '?'},

               {'POS': 'NUM', 'OP': '?'},

               {'ORTH': ')', 'OP': '?'},

               {'LEMMA': {'IN':['quarter','semester']}},

               ]

    matcher = Matcher(nlp.vocab)



    matcher.add("CourseLength", None, pattern)

    

    matches = matcher(doc)

    index = -1

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        matchResult = span.text



    if matchResult=='':

        return matchResult,matchResult

    

    doc = nlp(matchResult)

    numSem = ''

    numQtr = ''

    txtSem = ''

    txtQtr = ''

    isAfterConj = False

    readNext = True

    for token in doc:

        if readNext:

            if token.pos_ == 'CCONJ':

                isAfterConj = True

            elif token.text == '(':

                readNext = False

            elif not isAfterConj:

                if token.pos_ == 'NUM':

                    numSem = numSem + ' ' + token.text

                elif token.pos_ == 'NOUN' and token.lemma_.lower() != 'unit':

                    txtSem = token.text[0].upper()

            elif isAfterConj:

                if token.pos_ == 'NUM':

                    numQtr = numQtr + ' ' + token.text

                elif token.pos_ == 'NOUN' and token.lemma_.lower() != 'unit':

                    txtQtr = token.text[0].upper()

        else:

            if token.text == ')':

                readNext = True

    

    if numSem != '':

        numSem = w2n.word_to_num(numSem.strip())

        txtSem = str(numSem) + ' ' + txtSem

        if numQtr == '':

            return txtSem,matchResult

    if numQtr != '':

        numQtr = w2n.word_to_num(numQtr.strip())

        txtQtr = txtSem + '|' + str(numQtr) + ' ' + txtQtr

        return txtQtr,matchResult



def getCourseSubjects(row):

    txt = row['REQUIREMENT_TEXT']

    txt = txt.replace(')',') ').replace('and/or','or').replace('/',' or ').replace('-',' ').replace('  ',' ').lower()

    courseLengthTxt = row['COURSE_LENGTH_TEXT']

    if courseLengthTxt=='':

        return ''

    

    txt = txt[txt.find(courseLengthTxt)+len(courseLengthTxt)+1:]

    doc = nlp(txt)



    for sent in doc.sents:

        docSent = nlp(sent.text)

        isAfterAnd = False

        sentText = ''

        for token in docSent:

            if token.text == 'and':

                isAfterAnd = True

                continue

            else:

                if isAfterAnd:

                    if token.pos_ == 'NOUN' or token.pos_ == 'VERB':

                        sentText = sentText + ' & ' + token.text

                    else:

                        sentText = sentText + ' and ' + token.text

                else:

                    sentText = sentText + ' ' + token.text

                isAfterAnd = False

        break



    doc = nlp(sentText)



    pattern2 = [{'ORTH': {'IN':['in','on','of']}},

                {'POS': {'IN':['NOUN','VERB','ADJ','PUNCT','CCONJ','DET']}, 'OP': '*'},

                {'ORTH': {'IN': ['of','in']}, 'OP': '*'},

                {'POS': {'IN':['NOUN','VERB']}},

                ]

    matcher = Matcher(nlp.vocab)



    matcher.add("CourseSubject2", None, pattern2)



    matches = matcher(doc)

    result = ''

    index = -1

    prevMatch = ''

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        if index == -1:

            result = span.text

            index = sentText.find(span.text)

        elif index!=sentText.find(span.text):

            continue

        else:

            result = span.text

    

    result = result.replace('an accredited college or university','')

    

    if len(result.split(' ', 1))>1:

        result = result.split(' ', 1)[1]



    char_list = [', or ', ' or ', ',', ';']

    return re.sub("|".join(char_list), "|", result)



#Functions for getting license details

def getLicenseRequired(row, fromCol):

    if row['DRIVERS_LICENSE_REQ'] != '':

        return row['DRIVERS_LICENSE_REQ'],row['DRIV_LIC_TYPE']

    blank = ''

    txt = row[fromCol]

    txt = re.sub('\d\.','',txt)

    doc = nlp(txt)

    matchResult = ''

    licReq = ''

    patLicenseP = [{'LOWER': 'may'},

                   {'LOWER': 'require'},

                   {'LOWER': 'a', 'OP': '?'},

                   {'LOWER': 'class', 'OP': '?'},

                    {'POS': {'IN': ['PROPN','NOUN','ADJ','PUNCT']}, 'OP': '*'},

                    {'POS': 'PART','OP': '?'},

                    {'POS': 'CCONJ', 'OP': '?'},

                    {'LOWER': 'class', 'OP': '?'},

                    {'POS': {'IN': ['PROPN','NOUN','ADJ','PUNCT']}},

                    {'POS': 'PART','OP': '?'},

                    {'LEMMA': 'driver'},

                    {'POS': 'PART','OP': '?'},

                    {'LOWER': 'license'}

                   ]

    patLicenseR = [{'LOWER': 'class', 'OP': '?'},

                    {'POS': {'IN': ['PROPN','NOUN']}, 'OP': '?'},

                    {'POS': 'PART','OP': '?'},

                    {'POS': 'CCONJ', 'OP': '?'},

                    {'LOWER': 'class', 'OP': '?'},

                    {'POS': {'IN': ['PROPN','NOUN','ADJ']}},

                    {'POS': 'PART','OP': '?'},

                    {'LEMMA': 'driver'},

                    {'POS': 'PART','OP': '?'},

                    {'LOWER': 'license'}

                   ]

    matcher = Matcher(nlp.vocab)



    matcher.add("patLicenseP", None, patLicenseP)

    matcher.add("patLicenseR", None, patLicenseR)



    pattern = ''

    for sent in doc.sents:

        docSent = nlp(sent.text)

        matches = matcher(docSent)

        for match_id, start, end in matches:

            pattern = nlp.vocab.strings[match_id]

            span = docSent[start:end]

            matchResult = span.text

            break



        if matchResult != '':

            if pattern == 'patLicenseP':

                licReq = 'P'

            else:

                licReq = 'R'

            doc1 = nlp(matchResult)

            #print(matchResult)

            matchResult = [y.text for y in doc1 if (y.pos_ == 'NOUN' or y.pos_=='PROPN') and 

                           (y.text.lower() not in ['class','driver','license','california','commercial',

                                                   'state','drivers','firefighter'])]

            matchResult = '|'.join(matchResult)

            if matchResult != '':

                break

    if matchResult != '':       

        return licReq,matchResult

    elif licReq != '':

        return licReq,blank

    else:

        return blank,blank

    

def getAdditionalLic(row, fromCol):

    txt = row[fromCol]

    doc = nlp(txt)

    matcher = Matcher(nlp.vocab)

    

    pattern = [{'LOWER': 'valid'},

               {'POS': {'IN': ['PROPN','NOUN','ADJ','VERB']},'OP':'*'},

               {'IS_STOP': True, 'OP':'*'},

               {'LOWER': 'required'}]

    matcher.add("pattern", None, pattern)

    

    matchResult=''

    for sent in doc.sents:

        sentTxt = sent.text

        if 'license' in sentTxt.lower():

            sentTxt = sentTxt[sentTxt.lower().find('license'):]

        docSent = nlp(sentTxt)

        matches = matcher(docSent)

        for match_id, start, end in matches:

            pattern = nlp.vocab.strings[match_id]

            span = docSent[start:end]

            matchResult = span.text

            break

    return matchResult



def getExamTypeContent(row):

    content = row['Content']

    txtToMatch = 'THIS EXAMINATION IS TO BE GIVEN'

    if txtToMatch not in content:

        txtToMatch = 'THIS EXAM IS TO BE GIVEN'

        if txtToMatch not in content:

            return ''

        else:

            istart = content.find(txtToMatch)+25

    else:

        istart = content.find(txtToMatch)+32

    content = content[istart:]

    content = content.split('\n')

    content = list(filter(None, content))

    content = ','.join(content[:2])

    return content



def setExamTypeContent(row):

    content = row['ExamTypeContent'].lower()

    

    if ('open' in content) and ('interdepartmental' in content):

        return 'OPEN_INT_PROM'

    elif 'open' in content:

        return 'OPEN'

    elif ('interdepartmental' in content) or ('interdeparmental' in content):

        return 'INT_DEPT_PROM'

    elif 'departmental' in content:

        return 'DEPT_PROM'

    else:

        return ''



def getAppDeadlineDate(row):

    """Finds date pattern in application deadline text and returns date

    

    Parameters

    ----------

    row

        row of the dataframe to be searched

    

    Returns

    -------

    date

        1900-01-01 in case application will close without prior notice

        1901-01-01 in case no application deadline found in text

        1902-01-01 in case date found is not in correct format(i.e. %B %d %Y)

        Actual date in %Y-%m%d format in case a date is found and is in correct format

    """

    

    txt = row['APPLICATION_DEADLINE_TEXT']

    doc = nlp(txt)

    days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

    

    for sent in doc.sents:

        txtSent = sent.text.title().replace('Am','a.m.').replace('Pm','p.m.').replace('A.M.','a.m.').replace('P.M.','p.m.').replace('Septemeber','September')

        txtSent = re.sub("|".join(days), " ", txtSent)

        if 'without prior notice' in txtSent.lower():

            return pd.to_datetime('January 1, 1900', format="%B %d, %Y")

        docSent = nlp(txtSent)

        for ent in docSent.ents:

            if ent.label_ == "DATE":

                try:

                    return pd.to_datetime(ent.text.replace(',','').replace('  ',' '), format="%B %d %Y")

                except:

                    try:

                        toIndex = ent.text.find('To')

                        return pd.to_datetime(ent.text[:toIndex-1], format="%B %d, %Y")

                    except:

                        #print(ent.text)

                        return pd.to_datetime('January 1, 1902', format="%B %d, %Y")

        

    return pd.to_datetime('January 1, 1901', format="%B %d, %Y")



def getSelection(txt):

    matcher = Matcher(nlp.vocab)

    doc = nlp(txt)

    pattern = [{'LOWER': {'IN': ['weight','weights']}},

               {"TEXT": {"REGEX": ".*"},'OP':'*'},

#                {'POS': 'NUM'},

               {'TEXT': '%'}]



    matcher.add("Selection", None, pattern)



    matches = matcher(doc)



    spanTxt = ''

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]

        span = doc[start:end]

        spanTxt = span.text

    if spanTxt=='':

        return ''

    txt = spanTxt.lower().replace('weights','').replace('weight','')

    txt = txt.replace('. .','|').split('|')

    txt = [k.replace('.','').strip() for k in txt if ('%' not in k.strip()) & (k.strip()!='')]



    return '|'.join(txt) 



def getWhereToApplyUrl(txt):

    url = re.findall('http[s]?:?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)

    return url
job_class_df, data_dictionary = createDataFrameAndDictionary(jobs_df, job_titles, data_dictionary)
job_class_df.to_csv('job_class_submission.csv', index=False)

data_dictionary.to_csv('data_dictionary_submission.csv', index=False)
competency_df = pd.read_csv('../input/job-title-wise-competency-list-cityofla/df_competency.csv')



print('There are '+str(len(competency_df['job_class'].unique()))+' job classes for which competency data is collected and used in this analysis.')
competency_df[competency_df['job_class']=='Accounting Clerk Competencies.pdf'][['job_class','SubCompetency','Competency']]
def group_lower_ranking_values(pie_raw, column):

    """Converts pie_raw dataframe with multiple categories to a dataframe with fewer categories

    

    Calculate the 75th quantile and group the lesser values together.

    Lesser values will be labelled as 'Other'

    

    Parameters

    ----------

    pie_raw : DataFrame

        dataframe with the data to be aggregated

    column : str

        name of the column based on which dataframe values will be aggregated

    """

    pie_counts = pie_raw.groupby(column).agg('count')

    pct_value = pie_counts[lambda df: df.columns[0]].quantile(.75)

    values_below_pct_value = pie_counts[lambda df: df.columns[0]].loc[lambda s: s < pct_value].index.values

    def fix_values(row):

        if row[column] in values_below_pct_value:

            row[column] = 'Other'

        return row 

    pie_grouped = pie_raw.apply(fix_values, axis=1).groupby(column).agg('count')

    return pie_grouped



subcompetency_df = group_lower_ranking_values(competency_df, 'SubCompetency')



plt.figure(1, figsize=(10,10))

plt.pie(subcompetency_df['Id'],labels=subcompetency_df.index, autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.show()
plt.figure(1,(13,8))

plt.title('Missing Values in job_class file')

sns.heatmap(job_class_df.replace(r'^\s*$', np.nan, regex=True).isnull(), cbar=False)
def add_datepart(df, fldname, drop=True):

    fld = df[fldname]

    if not np.issubdtype(fld.dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','Quarter','Weekday_name',

            #'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'

             ):

        df[targ_pre+n] = getattr(fld.dt,n.lower())

    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis=1, inplace=True)



def showHistogram(data, title):

    """Display Histogram

    

    Parameters

    ----------

    data (Series)

        

    title

        title to dispay for the plot    

    """

    plt.figure(1,(6,6))

    data = data.value_counts(sort=False)

    sns.barplot(y=data.values,x=data.index)

    plt.gca().set_xticklabels(data.index,rotation='45')

    plt.title(title)

    plt.show()



job_class_df.fillna('', inplace=True)



#Adding some features using OPEN_DATE & APPLICATION_DEADLINE



job_class_df['APPLICATION_DEADLINE'] = pd.to_datetime(job_class_df['APPLICATION_DEADLINE'], format="%Y-%m-%d")

job_class_df['OPEN_DATE'].fillna('',inplace=True)

job_class_df['OPEN_DATE'] = pd.to_datetime(job_class_df['OPEN_DATE'], format="%Y-%m-%d")

job_class_df['no_of_days_to_apply'] = job_class_df.apply(lambda x: 

                                                         (x['APPLICATION_DEADLINE'] - x['OPEN_DATE']).days 

                                                         if (x['APPLICATION_DEADLINE'] - x['OPEN_DATE']).days >= 0 and 

                                                         x['APPLICATION_DEADLINE'].year>1903

                                                         else 0, axis=1)



open_date_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','OPEN_DATE']).size().reset_index(name='Freq'))



#Adding Date parts for OPEN_DATE

add_datepart(open_date_df, 'OPEN_DATE')
fig = plt.figure(4,(15,15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



data = open_date_df['OPEN_DATEWeekday_name'].value_counts(sort=False)

fig.add_subplot(2,2,1)

sns.barplot(y=data.values,x=data.index)

plt.title('Day wise job post openings')



data = open_date_df['OPEN_DATEMonth'].value_counts(sort=False)

fig.add_subplot(2,2,2)

sns.barplot(y=data.values,x=data.index)

plt.title('Month wise job post openings')



data = open_date_df['OPEN_DATEQuarter'].value_counts(sort=False)

fig.add_subplot(2,2,3)

sns.barplot(y=data.values,x=data.index)

plt.title('Quarter wise job post openings')



data = open_date_df['OPEN_DATEYear'].value_counts(sort=False)

fig.add_subplot(2,2,4)

sns.barplot(y=data.values,x=data.index)

plt.title('Year wise job post openings')



plt.show()
app_deadline_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','APPLICATION_DEADLINE']).size().reset_index(name='Freq'))

no_of_days_to_apply_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','no_of_days_to_apply']).size().reset_index(name='Freq'))

no_of_days_to_apply_df = no_of_days_to_apply_df[no_of_days_to_apply_df['no_of_days_to_apply']<50] ##removing outliers to get a better view



#Adding Date parts for APPLICATION_DEADLINE

add_datepart(app_deadline_df, 'APPLICATION_DEADLINE')
fig = plt.figure(2,(18,8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



data = app_deadline_df['APPLICATION_DEADLINEWeekday_name'].value_counts(sort=False)

fig.add_subplot(1,2,1)

sns.barplot(y=data.values,x=data.index)

plt.title('Day wise job application deadlines')



data = no_of_days_to_apply_df['no_of_days_to_apply'].value_counts(sort=False)

fig.add_subplot(1,2,2)

sns.barplot(y=data.values,x=data.index)

plt.title('No of days to apply')



plt.show()
def split_into_sentences(text):

    alphabets= "([A-Za-z])"

    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"

    suffixes = "(Inc|Ltd|Jr|Sr|Co)"

    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"

    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"

    websites = "[.](com|net|org|io|gov)"



    text = " " + text + "  "

    text = text.replace("\n"," ")

    text = re.sub(prefixes,"\\1<prd>",text)

    text = re.sub(websites,"<prd>\\1",text)

    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")

    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)

    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)

    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)

    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)

    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)

    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)

    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)

    if "”" in text: text = text.replace(".”","”.")

    if "\"" in text: text = text.replace(".\"","\".")

    if "!" in text: text = text.replace("!\"","\"!")

    if "?" in text: text = text.replace("?\"","\"?")

    text = text.replace(".",".<stop>")

    text = text.replace("?","?<stop>")

    text = text.replace("!","!<stop>")

    text = text.replace("<prd>",".")

    sentences = text.split("<stop>")

    sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]

    return sentences



# Returns Number of Words in the text 

def word_count(text):

    words = 0

    sentences = split_into_sentences(text)

    for sentence in sentences:

        wordList = re.sub("[^\w]", " ",  sentence).split()

        words += len(wordList)

    return words 

  

# Returns the number of sentences in the text 

def sentence_count(text):  

    sentences = split_into_sentences(text)

    return len(sentences)



def syllables_in_text(text):

    sentences = split_into_sentences(text)

    syllableCount = 0

    for sentence in sentences:

        wordList = re.sub("[^\w]", " ",  sentence).split()

        #syllables += sum([syllables_in_word(word.strip(string.punctuation)) for word in wordList])

        syllableCount += sum([syllables.estimate(word) for word in wordList]) 

    return syllableCount



def avg_sentence_length(text): 

    words = word_count(text) 

    sentences = sentence_count(text) 

    average_sentence_length = float(words / sentences) 

    return average_sentence_length 



def syllables_count(word):

    #print(textstatistics().syllable_count(word))

    return syllables_in_text(word)



# Returns the average number of syllables per 

# word in the text 

def avg_syllables_per_word(text): 

    syllable = syllables_count(text) 

    words = word_count(text)

    ASPW = float(syllable) / float(words) 

    return legacy_round(ASPW, 1)
job_class_df['title_len'] = job_class_df['JOB_CLASS_TITLE'].apply(lambda x: len(x))

jobs_df['content_word_len'] = jobs_df['Content'].apply(lambda x: word_count(x))

df_requirements = createRequirementsFrame(headingsFrame=createHeadingsFrame())

df_requirements['REQUIREMENT_TEXT_LEN'] = df_requirements['REQUIREMENT_TEXT'].apply(lambda x: len(x))



job_title_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','title_len']).size().reset_index(name='Freq'))
fig = plt.figure(2,(15,15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



fig.add_subplot(3,1,1)

sns.boxplot(x=job_title_df['title_len'])

plt.title('Job Title Length(in characters)')



fig.add_subplot(3,1,2)

sns.boxplot(x=jobs_df['content_word_len'])

plt.title('Job Class Content Length(in words)')



fig.add_subplot(3,1,3)

sns.boxplot(x=df_requirements['REQUIREMENT_TEXT_LEN'])

plt.title('Requirement Text Length(in characters, set wise)')



plt.show()
#source : https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

def getGradeLevelText(flesch_reading_ease):

    resultText = ''

    if 90.00 < flesch_reading_ease <= 100.00:

        resultText = '5th grade'

    elif 80.0 < flesch_reading_ease <= 90.00:

        resultText = '6th grade'

    elif 70.0 < flesch_reading_ease <= 80.00:

        resultText = '7th grade'

    elif 60.0 < flesch_reading_ease <= 70.00:

        resultText = '8th & 9th grade'

    elif 50.0 < flesch_reading_ease <= 60.00:

        resultText = '10th to 12th grade'

    elif 30.0 < flesch_reading_ease <= 50.00:

        resultText = 'College'

    elif 0.0 < flesch_reading_ease <= 30.00:

        resultText = 'College graduate'

    else:

        resultText = 'College graduate'

    return resultText

    

def flesch_reading_ease_grade(text): 

    """ 

        Implements Flesch Formula: 

        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW) 

        Here, 

          ASL = average sentence length (number of words  

                divided by number of sentences) 

          ASW = average word length in syllables (number of syllables  

                divided by number of words) 

    """

    ASL = avg_sentence_length(text)

    ASW = avg_syllables_per_word(text)

    FRE = 206.835 - float(1.015 * ASL) - float(84.6 * ASW) 

    FGL = (0.39 * ASL) + (11.8 * ASW) - 15.59

    gradeText = getGradeLevelText(legacy_round(FRE, 2))

    return legacy_round(FRE, 2), legacy_round(FGL, 2), gradeText
jobs_df['flesch_reading_ease_score'],jobs_df['flesch_grade_level'], jobs_df['grade_text'] = zip(*jobs_df['Content'].apply(flesch_reading_ease_grade))
plt.figure(1,(10,3))



sns.boxplot(x=jobs_df['flesch_reading_ease_score'])

plt.title('Readability Score for Bulletins')



plt.show()
jobs_grade_df = jobs_df['grade_text'].value_counts()

jobs_grade_df = pd.DataFrame({'code' : jobs_grade_df.index, 'values' : jobs_grade_df.values})



#explode = (0.1, 0, 0, 0, 0.1)

plt.figure(1,(8,8))

plt.pie(jobs_grade_df['values'],labels=jobs_grade_df['code'],#explode=explode, 

        autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.show()
salary_gen_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','ENTRY_SALARY_GEN']).size().reset_index(name='Freq'))

salary_gen_df['Is_Flat_Rated'] = salary_gen_df['ENTRY_SALARY_GEN'].apply(lambda x: 'No' if '-' in x else 'Yes')

salary_gen_df['Start_Range'] = salary_gen_df['ENTRY_SALARY_GEN'].apply(lambda x: 0 if x=='' else int(x.split('-')[0].replace(',','')))
fig = plt.figure(2,(15,8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



data = salary_gen_df['Is_Flat_Rated'].value_counts(sort=False)

fig.add_subplot(1,2,1)

sns.barplot(y=data.values,x=data.index)

plt.title('Is Flat Rated?')



fig.add_subplot(1,2,2)

sns.distplot(salary_gen_df['Start_Range'])

plt.title('Salary Distribution Plot')



plt.show()
exam_type_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','EXAM_TYPE']).size().reset_index(name='Freq'))



showHistogram(data=exam_type_df['EXAM_TYPE'], title='Exam Type Distribution')
selection_process_df = getValues(searchText='selection', COL_NAME='SELECTION_PROCESS', headingsFrame=createHeadingsFrame())

selection_process_df['SELECTION_CRITERIA'] = selection_process_df['SELECTION_PROCESS'].apply(getSelection)

s = selection_process_df["SELECTION_CRITERIA"].str.split('|', expand=True).stack()

i = s.index.get_level_values(0)

selection_process_df = selection_process_df.loc[i].copy()

selection_process_df["SELECTION_CRITERIA"] = s.values



selection_process_df = group_lower_ranking_values(selection_process_df, 'SELECTION_CRITERIA')



plt.figure(1, figsize=(8,8))

plt.pie(selection_process_df['SELECTION_PROCESS'],labels=selection_process_df.index, autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.show()
def getWordCloud(corpus):

    wordcloud = WordCloud(

                              background_color='white',

                              stopwords=STOPWORDS,

                              max_words=100,

                              max_font_size=50, 

                              random_state=42

                             ).generate(str(corpus))

    return wordcloud
job_duties_df=pd.DataFrame(job_class_df.groupby(['FILE_NAME','JOB_DUTIES']).size().reset_index(name='Freq'))

df_processnotes = getValues('process note','PROCESS_NOTES', headingsFrame=createHeadingsFrame())
fig = plt.figure(6,(15,15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



wordcloud = getWordCloud(' '.join(jobs_df['Content'].values))

fig.add_subplot(3,2,1)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Job Class Content')

plt.axis('off')



wordcloud = getWordCloud(' '.join(job_duties_df['JOB_DUTIES'].values))

fig.add_subplot(3,2,2)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Job Duties')

plt.axis('off')



wordcloud = getWordCloud(' '.join(df_requirements['REQUIREMENT_TEXT'].values))

fig.add_subplot(3,2,3)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Requirements')

plt.axis('off')



wordcloud = getWordCloud(' '.join(job_class_df['EDUCATION_MAJOR'].values))

fig.add_subplot(3,2,4)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Education Major')

plt.axis('off')



wordcloud = getWordCloud(' '.join(job_class_df['COURSE_SUBJECT'].values))

fig.add_subplot(3,2,5)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Course Subject')

plt.axis('off')



wordcloud = getWordCloud(' '.join(df_processnotes['PROCESS_NOTES'].values))

fig.add_subplot(3,2,6)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Process Notes')

plt.axis('off')



plt.show()
def getCorpus(data, COL_NAME):

    corpus = []

    for i in range(0, data.shape[0]):

        #Remove punctuations

        text = re.sub('[^a-zA-Z]', ' ', data[COL_NAME][i])



        #Convert to lowercase

        text = text.lower()



        #remove tags

        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)



        # remove special characters and digits

        text=re.sub("(\\d|\\W)+"," ",text)



        ##Convert to list from string

        text = text.split()



        ##Stemming

        ps=PorterStemmer()

        #Lemmatisation

        lem = WordNetLemmatizer()

        text = [lem.lemmatize(word) for word in text if not word in  

                set(stopwords.words("english"))] 

        text = " ".join(text)

        #if len(text)>2:

        corpus.append(text)

    return corpus



def lambda_unpack(f):

    return lambda args: f(*args)

def extract_key_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):



    # exclude words that are stop words or entirely punctuation

    punct = set(string.punctuation)

    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize and POS-tag words

    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)

                                                                    for sent in nltk.sent_tokenize(text)))

    # filter on certain POS tags and lowercase all words

    keywords = [word.lower() for word, tag in tagged_words

                  if tag in good_tags and word.lower() not in stop_words

                  and not all(char in punct for char in word)]



    return keywords



def extract_key_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):

    

    # exclude candidates that are stop words or entirely punctuation

    punct = set(string.punctuation)

    stop_words = set(nltk.corpus.stopwords.words('english'))

    # tokenize, POS-tag, and chunk using regular expressions

    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))

                                                    for tagged_sent in tagged_sents))

    # join constituent chunk words into a single chunked phrase

    keychunks = [' '.join(word for word, pos, chunk in group).lower()

                  for key, group in itertools.groupby(all_chunks, lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]



    return [cand for cand in keychunks

            if cand not in stop_words and not all(char in punct for char in cand)]
requirements = getRequirements(headingsFrame=createHeadingsFrame(),COL_NAME='REQUIREMENT_TEXT')

requirements['REQUIREMENT_WORDS'] = requirements['REQUIREMENT_TEXT'].apply(extract_key_words)

requirements['REQUIREMENT_CHUNKS'] = requirements['REQUIREMENT_TEXT'].apply(extract_key_chunks)

requirements['REQUIREMENT_SENTS'] = requirements['REQUIREMENT_WORDS'].apply(lambda x: ' '.join(x))
#Most frequently occuring words

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]



#Most frequently occuring Bi-grams

def get_top_n3_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3),  

            max_features=2000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]
requirements_corpus = getCorpus(COL_NAME='REQUIREMENT_SENTS', data=requirements)



top3_words = get_top_n3_words(requirements_corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Requirement sentences(n-gram)", "Freq"]



sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Requirement sentences(n-gram)", y="Freq", data=top3_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
df_processnotes['PROCESS_NOTE_WORDS'] = df_processnotes['PROCESS_NOTES'].apply(extract_key_words)

df_processnotes['PROCESS_NOTE_CHUNKS'] = df_processnotes['PROCESS_NOTES'].apply(extract_key_chunks)

df_processnotes['PROCESS_NOTE_SENTS'] = df_processnotes['PROCESS_NOTE_WORDS'].apply(lambda x: ' '.join(x))



process_notes_corpus = getCorpus(COL_NAME='PROCESS_NOTE_SENTS', data=df_processnotes)



top3_words = get_top_n3_words(process_notes_corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Process notes(n-gram)", "Freq"]



sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Process notes(n-gram)", y="Freq", data=top3_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
job_duties_df['JOB_DUTIES_WORDS'] = job_duties_df['JOB_DUTIES'].apply(extract_key_words)

job_duties_df['JOB_DUTIES_CHUNKS'] = job_duties_df['JOB_DUTIES'].apply(extract_key_chunks)

job_duties_df['JOB_DUTIES_SENTS'] = job_duties_df['JOB_DUTIES_WORDS'].apply(lambda x: ' '.join(x))



job_duties_corpus = getCorpus(COL_NAME='JOB_DUTIES_SENTS', data=job_duties_df)



top3_words = get_top_n3_words(job_duties_corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Job Duties(n-gram)", "Freq"]



sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Job Duties(n-gram)", y="Freq", data=top3_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
feminine_coded_words = [

    "agree",

    "affectionate",

    "child",

    "cheer",

    "collab",

    "commit",

    "communal",

    "compassion",

    "connect",

    "considerate",

    "cooperat",

    "co-operat",

    "depend",

    "emotiona",

    "empath",

    "feel",

    "flatterable",

    "gentle",

    "honest",

    "interpersonal",

    "interdependen",

    "interpersona",

    "inter-personal",

    "inter-dependen",

    "inter-persona",

    "kind",

    "kinship",

    "loyal",

    "modesty",

    "nag",

    "nurtur",

    "pleasant",

    "polite",

    "quiet",

    "respon",

    "sensitiv",

    "submissive",

    "support",

    "sympath",

    "tender",

    "together",

    "trust",

    "understand",

    "warm",

    "whin",

    "enthusias",

    "inclusive",

    "yield",

    "share",

    "sharin"

]



masculine_coded_words = [

    "active",

    "adventurous",

    "aggress",

    "ambitio",

    "analy",

    "assert",

    "athlet",

    "autonom",

    "battle",

    "boast",

    "challeng",

    "champion",

    "compet",

    "confident",

    "courag",

    "decid",

    "decision",

    "decisive",

    "defend",

    "determin",

    "domina",

    "dominant",

    "driven",

    "fearless",

    "fight",

    "force",

    "greedy",

    "head-strong",

    "headstrong",

    "hierarch",

    "hostil",

    "impulsive",

    "independen",

    "individual",

    "intellect",

    "lead",

    "logic",

    "objective",

    "opinion",

    "outspoken",

    "persist",

    "principle",

    "reckless",

    "self-confiden",

    "self-relian",

    "self-sufficien",

    "selfconfiden",

    "selfrelian",

    "selfsufficien",

    "stubborn",

    "superior",

    "unreasonab"

]



hyphenated_coded_words = [

    "co-operat",

    "inter-personal",

    "inter-dependen",

    "inter-persona",

    "self-confiden",

    "self-relian",

    "self-sufficien"

]
def clean_up_word_list(ad_text):

    cleaner_text = ''.join([i if ord(i) < 128 else ' '

        for i in ad_text])

    cleaner_text = re.sub("[\\s]", " ", cleaner_text, 0, 0)

    cleaned_word_list = re.sub(u"[\.\t\,“”‘’<>\*\?\!\"\[\]\@\':;\(\)\./&]",

        " ", cleaner_text, 0, 0).split(" ")

    word_list = [word.lower() for word in cleaned_word_list if word != ""]

    return de_hyphen_non_coded_words(word_list)



def de_hyphen_non_coded_words(word_list):

    for word in word_list:

        if word.find("-"):

            is_coded_word = False

            for coded_word in hyphenated_coded_words:

                if word.startswith(coded_word):

                    is_coded_word = True

            if not is_coded_word:

                word_index = word_list.index(word)

                word_list.remove(word)

                split_words = word.split("-")

                word_list = (word_list[:word_index] + split_words +

                    word_list[word_index:])

    return word_list



def assess_coding(row):

    coding = ''

    coding_score = row["feminine_ad_word_count"] - row["masculine_ad_word_count"]

    if coding_score == 0:

        if row["feminine_ad_word_count"]>0:

            coding = "neutral"

        else:

            coding = "empty"

    elif coding_score > 3:

        coding = "strongly feminine-coded"

    elif coding_score > 0:

        coding = "feminine-coded"

    elif coding_score < -3:

        coding = "strongly masculine-coded"

    else:

        coding = "masculine-coded"

    return coding



def assess_coding_txt(fem_word_count, masc_word_count):

    coding = ''

    coding_score = fem_word_count - masc_word_count

    if coding_score == 0:

        if fem_word_count>0:

            coding = "neutral"

        else:

            coding = "empty"

    elif coding_score > 3:

        coding = "strongly feminine-coded"

    elif coding_score > 0:

        coding = "feminine-coded"

    elif coding_score < -3:

        coding = "strongly masculine-coded"

    else:

        coding = "masculine-coded"

    return coding



def find_and_count_coded_words(advert_word_list, gendered_word_list):

    gender_coded_words = [word for word in advert_word_list

        for coded_word in gendered_word_list

        if word.startswith(coded_word)]

    return (",").join(gender_coded_words), len(gender_coded_words)



def assessBias(txt):

    words = clean_up_word_list(txt)

    txt_masc_coded_words, masc_word_count = find_and_count_coded_words(words, masculine_coded_words)

    txt_fem_coded_words, fem_word_count = find_and_count_coded_words(words, feminine_coded_words)

    coding = assess_coding_txt(fem_word_count, masc_word_count)

    print('List of masculine words found:')

    print(txt_masc_coded_words)

    print('\nList of feminine words found:')

    print(txt_fem_coded_words)

    return coding
print('Cleaning job bulletins words...')

jobs_df["clean_ad_words"] = jobs_df["Content"].apply(clean_up_word_list)

print('Finding masculine coded words...')

jobs_df["masculine_coded_ad_words"], jobs_df["masculine_ad_word_count"] = zip(*jobs_df["clean_ad_words"].apply(

    lambda x: find_and_count_coded_words(x,masculine_coded_words)))

print('Finding feminine coded words...')

jobs_df["feminine_coded_ad_words"], jobs_df["feminine_ad_word_count"] = zip(*jobs_df["clean_ad_words"].apply(

    lambda x: find_and_count_coded_words(x,feminine_coded_words)))

print('Assessing coding...')

jobs_df['coding'] = jobs_df.apply(assess_coding, axis=1)

print('Done')
jobs_bias_df = jobs_df['coding'].value_counts()

jobs_bias_df = pd.DataFrame({'code' : jobs_bias_df.index, 'values' : jobs_bias_df.values})



explode = (0.1, 0, 0, 0, 0.1)

plt.figure(1,(8,8))

plt.pie(jobs_bias_df['values'],labels=jobs_bias_df['code'],explode=explode, autopct='%1.1f%%', startangle=90)

plt.axis('equal')

plt.show()
fig = plt.figure(2,(15,15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



wordcloud = getWordCloud(' '.join(jobs_df["masculine_coded_ad_words"]))

fig.add_subplot(1,2,1)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Masculine-Coded Words in Bulletins')

plt.axis('off')



wordcloud = getWordCloud(' '.join(jobs_df["feminine_coded_ad_words"]))

fig.add_subplot(1,2,2)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Feminine-Coded Words in Bulletins')

plt.axis('off')



plt.show()
job_applicants = pd.read_csv('../input/la-applicants-gender-and-ethnicity/rows.csv')

job_applicants.head()
job_applicants['FemalePercentage'] = (job_applicants['Female'])*100/(job_applicants['Female']+job_applicants['Male'])

job_applicants['job_title'] = job_applicants['Job Description'].apply(lambda x : x[:-4].strip())



job_applicants = job_applicants.merge(job_class_df[['FILE_NAME','JOB_DUTIES','JOB_CLASS_TITLE']],

                    how = 'left', left_on = 'job_title', right_on = 'JOB_CLASS_TITLE')

job_applicants = job_applicants.drop_duplicates()

job_applicants.fillna('',inplace=True)



job_applicants['JOB_DUTIES_WORDS'] = job_applicants['JOB_DUTIES'].apply(extract_key_words)

job_applicants['JOB_DUTIES_CHUNKS'] = job_applicants['JOB_DUTIES'].apply(extract_key_chunks)

job_applicants['JOB_DUTIES_SENTS'] = job_applicants['JOB_DUTIES_WORDS'].apply(lambda x: ' '.join(x))



job_applicants_female = job_applicants[job_applicants['FemalePercentage']>60]

job_applicants_female.reset_index(inplace=True)

job_applicants_male = job_applicants[job_applicants['FemalePercentage']<30]

job_applicants_male.reset_index(inplace=True)
female_duties_corpus = getCorpus(job_applicants_female,'JOB_DUTIES_SENTS')

wordcloud = getWordCloud(female_duties_corpus)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Job Duties from female dominated jobs')

plt.axis('off')



plt.show()
male_duties_corpus = getCorpus(job_applicants_male,'JOB_DUTIES_SENTS')

wordcloud = getWordCloud(male_duties_corpus)

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Job Duties from male dominated jobs')

plt.axis('off')



plt.show()
def sentence_similarity(sent1, sent2, stopwords=None):

    if stopwords is None:

        stopwords = []

 

    sent1 = [w.lower() for w in sent1]

    sent2 = [w.lower() for w in sent2]

 

    all_words = list(set(sent1 + sent2))

 

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)

 

    # build the vector for the first sentence

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # build the vector for the second sentence

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2)



def build_similarity_matrix(sentences, stopwords=None):

    # Create an empty similarity matrix

    S = np.zeros((len(sentences), len(sentences)))

 

 

    for idx1 in range(len(sentences)):

        for idx2 in range(len(sentences)):

            if idx1 == idx2:

                continue

 

            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords) 

    return S



def showSimilarJobs(data, job_title, COL_NAME, similarity_matrix, n=3):

    i = data[data[COL_NAME]==job_title].index.values.astype(int)

    if len(i)>0:

        i = i[0]

        ind = np.argpartition(similarity_matrix[i], -n)[-n:]

        print(data.loc[ind][COL_NAME].values)

    else:

        print('Job class not found in data.')
#Creating similarity matrix using words in requirements section of bulletins

requirements_s_matrix = build_similarity_matrix(requirements['REQUIREMENT_WORDS'], STOP_WORDS)



#Removing bulletins where job duties section is missing

job_duties_df['WORDS_LEN'] = job_duties_df['JOB_DUTIES_WORDS'].apply(lambda x: len(x))

job_duties_words = job_duties_df[job_duties_df['WORDS_LEN']>0]

job_duties_words = job_duties_words.reset_index()



#Creating similarity matrix using words in job duties section of bulletins

job_duties_s_matrix = build_similarity_matrix(job_duties_words['JOB_DUTIES_WORDS'], STOP_WORDS)
showSimilarJobs(requirements,'ANIMAL KEEPER 4304 083118.txt','FILE_NAME', requirements_s_matrix, 3)
showSimilarJobs(job_duties_words,'ANIMAL KEEPER 4304 083118.txt','FILE_NAME', job_duties_s_matrix, 3)
def createEdges (job_class_df):

    edge_cols = ['JOB_CLASS_TITLE','EXP_JOB_CLASS_TITLE','EXPERIENCE_LENGTH','EXPERIENCE_LEN_UNIT']



    df_edges = job_class_df[edge_cols]



    #EXP_JOB_CLASS_TITLE consists of job classes separated with ',' (e.g. SYSTEMS ANALYST, MANAGEMENT ANALYST)

    #Spliting these values and creating rows for each job class

    s = df_edges["EXP_JOB_CLASS_TITLE"].str.split(',', expand=True).stack()

    i = s.index.get_level_values(0)

    df_edges = df_edges.loc[i].copy()

    df_edges["EXP_JOB_CLASS_TITLE"] = s.values



    df_edges['EXP_JOB_CLASS_TITLE'] = df_edges['EXP_JOB_CLASS_TITLE'].apply(lambda x: x.strip())



    df_edges = pd.DataFrame(df_edges.groupby(edge_cols).size().reset_index(name='Freq'))

    df_edges.drop(columns='Freq', inplace=True)



    #remove exp_job_class_title if it is same as job_class_title

    df_edges.loc[df_edges['JOB_CLASS_TITLE']==df_edges['EXP_JOB_CLASS_TITLE'],'EXP_JOB_CLASS_TITLE']=''



    #To find implicit links

    #For job classes like 'Senior Systems Analyst' - junior level job class will be like 'Systems Analyst', if the title exists

    #Find senior job_class that doesn't have junior level exp_job_class_title 

    job_without_juniors = []

    for job_class in df_edges['JOB_CLASS_TITLE'].unique():

        if job_class.startswith('SENIOR'):

            exist = False

            junior_level = job_class.replace('SENIOR','').strip()

            for exp_job in df_edges[df_edges['JOB_CLASS_TITLE']==job_class]['EXP_JOB_CLASS_TITLE']:

                if exp_job.strip()==junior_level:

                    exist = True

                    break

            if not exist:

                #add only those titles that are actually job_titles

                if junior_level in map(str.strip,job_titles[0].values):

                    job_without_juniors.append([job_class,junior_level,'',''])



    df_edges = df_edges.append(pd.DataFrame(job_without_juniors,

                                           columns = edge_cols), ignore_index=True)



    #df_edges.head()

    return df_edges



def getLowerJobEdges(job_title, data):

    """Lower edges or the job titles that can be promoted to job_title.

    

    Parameters

    ----------

    job_title : str

        job class title for which lower job classes are to be searched in data

    data : DataFrame

        dataframe used to search lower job classes

    

    Returns

    -------

    list : [tuple,str]

        tuple - edge tuple (source, target)

        str - edge label

    """

    

    result = []

    for index,e in data.iterrows():

        if e['JOB_CLASS_TITLE'] == job_title and e['EXP_JOB_CLASS_TITLE']!='':

            result.append([(e['EXP_JOB_CLASS_TITLE'],e['JOB_CLASS_TITLE']),

                          str(e['EXPERIENCE_LENGTH'])+' '+e['EXPERIENCE_LEN_UNIT']])

            result.extend(getLowerJobEdges(e['EXP_JOB_CLASS_TITLE'], data))

    return result



def getUpperJobEdges(job_title, data):

    """Upper edges or the job titles that a job_title can be promoted to.

    

    Parameters

    ----------

    job_title : str

        job class title for which upper job classes are to be searched in data

    data : DataFrame

        dataframe used to search upper job classes

    

    Returns

    -------

    list : [tuple,str]

        tuple - edge tuple (source, target)

        str - edge label

    """

    

    result = []

    for index,e in data.iterrows():

        if e['EXP_JOB_CLASS_TITLE'] == job_title:

            result.append([(e['EXP_JOB_CLASS_TITLE'],e['JOB_CLASS_TITLE']),

                          str(e['EXPERIENCE_LENGTH'])+' '+e['EXPERIENCE_LEN_UNIT']])

            result.extend(getUpperJobEdges(e['JOB_CLASS_TITLE'], data))

    return result



def getEdges(job_title, data):

    """Edges or job titles that a job_title can be promoted to/from.

    

    Parameters

    ----------

    job_title : str

        job title to search for edges

    data : DataFrame

        dataframe used to search upper job classes

    

    Returns

    -------

    list : [tuple,str]

        tuple - edge tuple (source, target)

        str - edge label

    """

    

    edges = []

    edges = getLowerJobEdges(job_title, data)

    edges.extend(getUpperJobEdges(job_title, data))

    return edges



def plotGraph(G):

    """Plots a networkx graph

    

    Parameters

    ----------

    G : networkx Graph

    """

    

    plt.figure(figsize=(12, 12)) 

    plt.axis('off')

    pos = nx.circular_layout(G)

    nx.draw_networkx(G, pos, with_labels=True, 

                    node_color='blue', font_size=8, node_size=10000, width = 2)

    

    #Uncomment below to draw edge labels

    #nx.draw_networkx_edge_labels(G, label_pos=.5, pos=pos)

    

    plt.show()



def showPromotionalPaths(df_edges, job_class_df, job_title, candidateDataDf=pd.DataFrame):

    """Displays eligible/proposed promotions. Future promotion paths are also displayed as a directed graph.

    

    Parameters

    ----------

    candidateDatadf : DataFrame

        Dataframe consisting of data for a candidate to be searched for available promotions. Only first row will be considered.

    df_edges : DataFrame

        Dataframe consisting of all available data edges from all job classes

    """

    

    if 'JOB_CLASS_TITLE' in candidateDataDf.columns:

        #only first row will be considered and searched for promotional paths

        job_title = candidateDataDf['JOB_CLASS_TITLE'].iloc[0]

    if job_title == '':

        print('No job title assigned.')

        return

    

    job_nodes = job_class_df[job_class_df['JOB_CLASS_TITLE']==job_title][['JOB_CLASS_TITLE',

                                                                          'EXP_JOB_CLASS_TITLE']]

    job_node_names = [n[0] for index,n in job_nodes.iterrows()]

    job_edges = getEdges(job_title, df_edges)

    

    for edge in job_edges:

        if edge[0][0] == job_title:

            checkEligibility(job_title=edge[0][1], job_class_df = job_class_df, candidateDataDf=candidateDataDf) #check and print eligiblity for explicit promotions

    

    job_edge_names = [e[0] for e in job_edges]

    

    #set edge labels

    experience_dict={}

    for edge in job_edges:

        experience_dict[edge[0]] = edge[1]



    #networkx directed graph for promotion path visuals

    G = nx.DiGraph()

    G.add_nodes_from(job_node_names)

    G.add_edges_from(job_edge_names)

    nx.set_edge_attributes(G, experience_dict, 'Experience')



    #print(nx.info(G))

    plotGraph(G)
class Experience:

    """A class to represent experience details

    

    Attributes

    ----------

    expLength : float

        experience length

    expLengthUnit : str

        unit for experience length (e.g. year/month/hour)

    fullTimePartTime : str

        type of experience (FULL_TIME/PART_TIME)

    paidVolunteer : str

        type of experience (PAID/VOLUNTEER)

    experience : str

        Represents the experience text by concatenating other attributes

    errMsg : str

        Represents the errors/mismatch occured while comparing two experiences. Blank if matched.

    

    Methods

    -------

    compare(objCandidateExperience)

        Compares self with another experience object(i.e. objCandidateExperience) and returns errMsg accordingly

    getExperience

        Outputs a string representing the experience text

    getErrMsg

        Outputs a message to be displayed in case of comparison mismatch

    """

    

    def __init__(self, dfJobClassRow):

        """

        

        Parameters

        ----------

        dfJobClassRow : Series

            Row of Job Class dataframe, containing experience related columns

        """

        

        if dfJobClassRow['EXPERIENCE_LENGTH']=='':

            self.expLength = 0

        else:

            self.expLength = float(dfJobClassRow['EXPERIENCE_LENGTH'])

        self.expLengthUnit = dfJobClassRow['EXPERIENCE_LEN_UNIT']

        self.fullTimePartTime = dfJobClassRow['FULL_TIME_PART_TIME']

        self.paidVolunteer = dfJobClassRow['PAID_VOLUNTEER']

        self.experience = self.getExperience()

        self.errMsg = ''

    

    def compare(self, objCandidateExperience):

        """Compares self with another experience class object

        

        Parameters

        ----------

        objCandidateExperience : Experience

            object for Experience class created for a candidate

        

        Returns

        -------

        errMsg : str

            blank if matched, else mismatched experience requirement string

        """

        

        if self.experience == objCandidateExperience.experience:

            self.errMsg = ''

        else:

            if self.expLength == objCandidateExperience.expLength and self.expLengthUnit == objCandidateExperience.expLengthUnit:

                if self.fullTimePartTime == objCandidateExperience.fullTimePartTime:

                    if objCandidateExperience.paidVolunteer in self.paidVolunteer:

                        self.errMsg = ''

                    else:

                        self.errMsg = self.getErrorMsg()

                else:

                    if objCandidateExperience.fullTimePartTime in self.fullTimePartTime:

                        if objCandidateExperience.paidVolunteer in self.paidVolunteer.contains:

                            self.errMsg = ''

                        else:

                            self.errMsg = self.getErrorMsg()

                    else:

                        self.errMsg = self.getErrorMsg()

            elif self.expLengthUnit == objCandidateExperience.expLengthUnit:

                if self.expLength < objCandidateExperience.expLength:

                    if self.fullTimePartTime == objCandidateExperience.fullTimePartTime:

                        if objCandidateExperience.paidVolunteer in self.paidVolunteer:

                            self.errMsg = ''

                        else:

                            self.errMsg = self.getErrorMsg()

                    else:

                        if objCandidateExperience.fullTimePartTime in self.fullTimePartTime:

                            if objCandidateExperience.paidVolunteer in self.paidVolunteer.contains:

                                self.errMsg = ''

                            else:

                                self.errMsg = self.getErrorMsg()

                        else:

                            self.errMsg = self.getErrorMsg()

                else:

                    self.errMsg = self.getErrorMsg()

            else:

                self.errMsg = self.getErrorMsg()

        #print(self.experience, objCandidateExperience.experience)

        return self.errMsg

    

    def getExperience(self):

        """Outputs a string representing the experience text

        

        Returns

        -------

        str - string representing the experience text

        """

        

        return ' '.join([str(float(self.expLength) if self.expLength != '' else self.expLength),

                         self.expLengthUnit,

                         self.fullTimePartTime.replace('|','/'),

                         self.paidVolunteer.replace('|','/')])

    

    def getErrorMsg(self):

        """Outputs a message to be displayed in case of comparison mismatch"""

        

        return self.experience + ' experience is required for this job class.'



class License:

    """A class to represent license details

    

    Attributes

    ----------

    driverLicReq : str

        whether driver's license is required for a job class or not. In case of candidate, it should be 'R' if one holds a license

    driverLicType : str

        license types hold by candidate/ license types required for a job class

    additionalLic : str

        additional licenses hold by candidate/ required for a job class

    license : str

        Represents the license text by concatenating other attributes

    errMsg : str

        Represents the errors/mismatch occured while comparing two licenses. Blank if matched.

    

    Methods

    -------

    compare(objCandidateLicense)

        Compares self with another license object(i.e. objCandidateLicense) and returns errMsg accordingly

    getLicense

        Outputs a string representing the license text

    getErrMsg

        Outputs a message to be displayed in case of comparison mismatch

    """

    

    def __init__(self, dfJobClassRow):

        """

        

        Parameters

        ----------

        dfJobClassRow : Series

            Row of Job Class dataframe, containing license related columns

        """

        

        if 'DRIVERS_LICENSE_REQ' in dfJobClassRow:

            self.driverLicReq = dfJobClassRow['DRIVERS_LICENSE_REQ']

        else:

            self.driverLicReq = 'R'

        self.driverLicType = dfJobClassRow['DRIV_LIC_TYPE']

        self.additionalLic = dfJobClassRow['ADDTL_LIC']

        self.license = self.getLicense()

        self.errMsg = ''

    

    def compare(self, objCandidateLicense):

        """Compares self with another license class object

        

        If DRIVERS_LICENSE_REQ is 'P', then this will consider it as a match

        Additional licenses are not compared in this method.

        

        Parameters

        ----------

        objCandidateLicense : License

            object for License class created for a candidate

        

        Returns

        -------

        errMsg : str

            blank if matched, else mismatched license requirement string

        """

        

        if self.driverLicReq == 'P' or self.driverLicReq == '':

            self.errMsg = ''

        else:

            if self.driverLicType == '' and objCandidateLicense.driverLicReq == 'R':

                self.errMsg = ''            

            elif objCandidateLicense.driverLicType in self.driverLicType:

                self.errMsg = ''

            else:

                self.errMsg = self.getErrorMsg()

        

        return self.errMsg

    

    def getLicense(self):

        """Outputs a string representing the license text

        

        Returns

        -------

        str - string representing the license text

        """

        

        return self.driverLicType

    

    def getErrorMsg(self):

        """Outputs a message to be displayed in case of comparison mismatch"""

        

        if self.driverLicType != '':

            return self.driverLicType + ' license is required for this job class.'

        else:

            return 'A valid California driver\'s license is required for this job class.'



class JobClass:

    """A class to represent job class details

    

    Attributes

    ----------

    jobClassTitle : str

        job class title/ current job class of candidate

    examType : str

        OPEN, INT_DEPT_PROM, DEPT_PROM, OPEN_INT_PROM ('' if not provided)

    selectionCriteria : str

        selection criteria for a job class ('' if not provided)

    requirementSetId : str

        requirement set id ('' if not provided)

    requirementSubSetId : str

        requirement sub set id ('' if not provided)

    experience : Experience

        experience details for job class

    license : License

        license details for job class

    errMsg : str

        Represents the errors/mismatch occured while comparing two licenses. Blank if matched.

    

    Methods

    -------

    compare(candidateJobClass)

        Compares self with another JobClass object(i.e. candidateJobClass) and returns errMsg accordingly

    """

    

    def __init__(self, dfJobClassRow):

        """

        

        Parameters

        ----------

        dfJobClassRow : Series

            Row of Job Class dataframe, containing job class columns

        """

        

        if 'JOB_CLASS_TITLE' in dfJobClassRow:

            self.jobClassTitle = dfJobClassRow['JOB_CLASS_TITLE']

        else:

            self.jobClassTitle = 'CandidateJobClass'

        if 'EXAM_TYPE' in dfJobClassRow:

            self.examType = dfJobClassRow['EXAM_TYPE']

        else:

            self.examType = ''

        if 'SELECTION_CRITERIA' in dfJobClassRow:

            self.selectionCriteria = dfJobClassRow['SELECTION_CRITERIA']

        else:

            self.selectionCriteria = ''

        self.expJobClassTitle = dfJobClassRow['EXP_JOB_CLASS_TITLE']

        if 'REQUIREMENT_SET_ID' in dfJobClassRow:

            self.requirementSetId = dfJobClassRow['REQUIREMENT_SET_ID']

        else:

            self.requirementSetId = ''

        if 'REQUIREMENT_SUBSET_ID' in dfJobClassRow:

            self.requirementSubSetId = dfJobClassRow['REQUIREMENT_SUBSET_ID']

        else:

            self.requirementSubSetId = ''

        self.experience = Experience(dfJobClassRow)

        self.license = License(dfJobClassRow)

        self.errMsg = ''

    

    def compare(self, candidateJobClass):

        """Compares education, experience, license details of a candidate job class with self

        

        Parameters

        ----------

        candidateJobClass : JobClass

            JobClass object for candidate to be compared

        Returns

        -------

        errMsg : str

            blank if matched, else mismatched requirements string

        """

        

        self.errMsg = self.errMsg + ' ' + self.experience.compare(candidateJobClass.experience)

        self.errMsg = self.errMsg + ' ' + self.license.compare(candidateJobClass.license)

        return self.errMsg.strip()
def checkRequirements(data, candidateJobClass):

    """Matches the requirements of the job class with supplied candidate job class data

    

    Parameters

    ----------

    data : DataFrame

        job class data with the data_dictionary fields (requirements to match to be eligible for the promotion)

    candidateJobClass : JobClass

        candidate data with which the requirements will be matched

    

    Returns

    -------

    list [errMsg, conj]

        a row for each requirement in data

        errMsg - blank if matches the requirements, else contains not matched requirement texts

        conj - conjuction to be checked with other requirements (i.e.or/and)

    """

    

    conj = ''

    result = []



    for index,row in data.iterrows():

        conj = row['REQUIREMENT_CONJ']

        if conj == '':

            conj = 'and'

        

        jobClass = JobClass(row)

        errMsg = jobClass.compare(candidateJobClass)

        result.append([errMsg, conj])

    return result



def matchRequirements(requirementsResult):

    """Applies conjuctions on multiple requirements(checkRequirements result) 

    

    Parameters

    ----------

    requirementsResult : list

        [errMsg, conj]

        errMsg - blank for matched requirements, else contains not matched requirement texts

        conj - conjuction to be checked with other requirements (i.e.or/and)

    

    Returns

    -------

    errMsg, conj

        errMsg - blank if all requirements matched, else ';' separated message for all unmatched requirements

        conj - last conjuction to be matched with other requirements(if any). Will be used if first called for sub-requirements

    """

    

    resultErrList = [] 

    resultRequirementsMatch = False

    conj = ''

    for row in requirementsResult:

        errMsg = row[0]

        conj = row[1]

        currentRequirementsMatch = False

        if len(errMsg)==0:

            currentRequirementsMatch = True

        else:

            resultErrList.append(errMsg)

            currentRequirementsMatch = False

        if conj=='or':

            if not resultRequirementsMatch:

                if currentRequirementsMatch:

                    resultErrList = []

                    resultRequirementsMatch = True

        elif conj=='and':

            if not currentRequirementsMatch:

                resultRequirementsMatch = False

        else:

            resultRequirementsMatch = currentRequirementsMatch

    return ';'.join(resultErrList),conj



def checkEligibility(job_title, job_class_df, candidateDataDf):

    """For a job title, compares all the requirements for all the explicitly mentioned promotions in job class dataframe.

    

    Prints messages based on requirements match/mismatch.

    

    Parameters

    ----------

    job_title : str

        job title of the candidate to be searched and matched with the requirements

    """

    

    job_title = job_title

    single_job_class_df = job_class_df[job_class_df['JOB_CLASS_TITLE'] == job_title]

    single_job_class_df = single_job_class_df.iloc[::-1] #reverse the dataframe

    candidate_job_class = JobClass(candidateDataDf.iloc[0])

    prevRqmntId = ''

    result = []

    i = 0

    for index,row in single_job_class_df.iterrows(): 

        rqmntId = ''

        if row['REQUIREMENT_SUBSET_ID'] != '':

            rqmntId = row['REQUIREMENT_SET_ID']

            if prevRqmntId == '':

                prevRqmntId = rqmntId

                data = single_job_class_df[single_job_class_df['REQUIREMENT_SET_ID'] == rqmntId]

                rqmntChk = checkRequirements(data, candidate_job_class)

                errMsg,conj = matchRequirements(rqmntChk)

                conj = data['REQUIREMENT_CONJ'].iloc[0]

                result.append([errMsg,conj])

            if rqmntId != prevRqmntId:

                prevRqmntId = rqmntId  

                data = single_job_class_df[single_job_class_df['REQUIREMENT_SET_ID'] == rqmntId]

                rqmntChk = checkRequirements(data, candidate_job_class)

                errMsg,conj = matchRequirements(rqmntChk)

                conj = data['REQUIREMENT_CONJ'].iloc[0]

                result.append([errMsg,conj])

        else:

            rqmntId = ''

            data = pd.DataFrame(row).transpose()

            rqmntChk = checkRequirements(data, candidate_job_class)

            errMsg,conj = rqmntChk[0][0],rqmntChk[0][1]

            result.append([errMsg,conj])

    #print(result)

    errMsg,conj = matchRequirements(result)

    if errMsg.strip() == '':

        print('Candidate is eligible for promotion for job class ' + job_title + '\n')

    else:

        print('For job class '+ job_title +' you need to fulfill below requirements: ')

        print(errMsg + '\n')
def getTreeFormattedChildren(df, title):

    if title=='':

        print('No job title provided')

        return

    resultList = []

    result = {}

    result['name'] = str(title.title())

    children = df[df['name']==title]['children'].values

    if len(children)>0:

        for child in children:

            resultList.append(getTreeFormattedChildren(df, child))

        result['children'] = resultList

    else:

        return result

    return result



def getTreeHTML(data, title):

    treeHtml = """

    <!DOCTYPE html>

    <meta charset="utf-8">

    <style>

    .node circle {

      fill: #fff;

      stroke: steelblue;

      stroke-width: 3px;

    }



    .node text { font: 8px sans-serif; }



    .node--internal text {

      text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;

    }



    .link {

      fill: none;

      stroke: #ccc;

      stroke-width: 2px;

    }

    </style>

    <body>

    <div style="font: 30px sans-serif; color: #42374D"><b>Promotional Paths</b></div>

    <div id="fd"></div>

    <div style="font: 16px sans-serif; color: #42374D"><b>Above graph is interacitve. Click on nodes to view more detailed paths

    </b>

    </div>

    <script>

    require.config({

        paths: {

            d3: "https://d3js.org/d3.v4.min"

         }

     });



    require(["d3"], function(d3) {

    var treeData =

      """+json.dumps(getTreeFormattedChildren(data, title))+"""



    // Set the dimensions and margins of the diagram

    var margin = {top: 20, right: 90, bottom: 30, left: 90},

        width = 960 - margin.left - margin.right,

        height = 700 - margin.top - margin.bottom;



    // append the svg object to the body of the page

    // appends a 'group' element to 'svg'

    // moves the 'group' element to the top left margin

    var svg = d3.select("#fd").append("svg")

        .attr("width", width + margin.right + margin.left)

        .attr("height", height + margin.top + margin.bottom)

      .append("g")

        .attr("transform", "translate("

              + margin.left + "," + margin.top + ")");



    var i = 0,

        duration = 750,

        root;



    // declares a tree layout and assigns the size

    var treemap = d3.tree().size([height, width]);



    // Assigns parent, children, height, depth

    root = d3.hierarchy(treeData, function(d) { return d.children; });

    root.x0 = height / 2;

    root.y0 = 0;



    // Collapse after the second level

    root.children.forEach(collapse);



    update(root);



    // Collapse the node and all it's children

    function collapse(d) {

      if(d.children) {

        d._children = d.children

        d._children.forEach(collapse)

        d.children = null

      }

    }



    function update(source) {



      // Assigns the x and y position for the nodes

      var treeData = treemap(root);



      // Compute the new tree layout.

      var nodes = treeData.descendants(),

          links = treeData.descendants().slice(1);



      // Normalize for fixed-depth.

      nodes.forEach(function(d){ d.y = d.depth * 180});



      // ****************** Nodes section ***************************



      // Update the nodes...

      var node = svg.selectAll('g.node')

          .data(nodes, function(d) {return d.id || (d.id = ++i); });

    

      // Enter any new modes at the parent's previous position.

      var nodeEnter = node.enter().append('g')

          .attr('class', 'node')

          .attr("transform", function(d) {

            return "translate(" + source.y0 + "," + source.x0 + ")";

        })

        .on('click', click);



      // Add Circle for the nodes

      nodeEnter.append('circle')

          .attr('class', 'node')

          .attr('r', 1e-6)

          .style("fill", function(d) {

              return d._children ? "lightsteelblue" : "#fff";

          });



      // Add labels for the nodes

      nodeEnter.append('text')

          .attr("dy", ".35em")

          .attr("x", function(d) {

              return d.children || d._children ? -13 : 13;

          })

          .attr("text-anchor", function(d) {

              return d.children || d._children ? "end" : "start";

          })

          .text(function(d) { return d.data.name; });



      // UPDATE

      var nodeUpdate = nodeEnter.merge(node);



      // Transition to the proper position for the node

      nodeUpdate.transition()

        .duration(duration)

        .attr("transform", function(d) { 

            return "translate(" + d.y + "," + d.x + ")";

         });



      // Update the node attributes and style

      nodeUpdate.select('circle.node')

        .attr('r', 10)

        .style("fill", function(d) {

            return d._children ? "lightsteelblue" : "#fff";

        })

        .attr('cursor', 'pointer');





      // Remove any exiting nodes

      var nodeExit = node.exit().transition()

          .duration(duration)

          .attr("transform", function(d) {

              return "translate(" + source.y + "," + source.x + ")";

          })

          .remove();



      // On exit reduce the node circles size to 0

      nodeExit.select('circle')

        .attr('r', 1e-6);



      // On exit reduce the opacity of text labels

      nodeExit.select('text')

        .style('fill-opacity', 1e-6);



      // ****************** links section ***************************



      // Update the links...

      var link = svg.selectAll('path.link')

          .data(links, function(d) { return d.id; });



      // Enter any new links at the parent's previous position.

      var linkEnter = link.enter().insert('path', "g")

          .attr("class", "link")

          .attr('d', function(d){

            var o = {x: source.x0, y: source.y0}

            return diagonal(o, o)

          });



      // UPDATE

      var linkUpdate = linkEnter.merge(link);



      // Transition back to the parent element position

      linkUpdate.transition()

          .duration(duration)

          .attr('d', function(d){ return diagonal(d, d.parent) });



      // Remove any exiting links

      var linkExit = link.exit().transition()

          .duration(duration)

          .attr('d', function(d) {

            var o = {x: source.x, y: source.y}

            return diagonal(o, o)

          })

          .remove();



      // Store the old positions for transition.

      nodes.forEach(function(d){

        d.x0 = d.x;

        d.y0 = d.y;

      });



      // Creates a curved (diagonal) path from parent to the child nodes

      function diagonal(s, d) {



        path = `M ${s.y} ${s.x}

                C ${(s.y + d.y) / 2} ${s.x},

                  ${(s.y + d.y) / 2} ${d.x},

                  ${d.y} ${d.x}`



        return path

      }



      // Toggle children on click.

      function click(d) {

        if (d.children) {

            d._children = d.children;

            d.children = null;

          } else {

            d.children = d._children;

            d._children = null;

          }

        update(d);

      }

    }

    });

    </script>

    </body>

    """

    return treeHtml



def showInteractivePromotionalPaths(df_edges, job_title):

    treeEdges = getEdges(data=df_edges, job_title=job_title)

    edges = []

    for edge in treeEdges:

        edges.append((edge[0][0],edge[0][1]))

    

    treeEdges = pd.DataFrame(edges, columns=['name','children'])

    treeEdges = treeEdges.groupby(['name','children']).size().reset_index(name='Freq')

    return (getTreeHTML(treeEdges, job_title))
df_edges = createEdges(job_class_df)

HTML(showInteractivePromotionalPaths(df_edges, 'WATER UTILITY WORKER'))
#Test/Sample data for CandidateJobClass

data = {'JOB_CLASS_TITLE': 'WATER UTILITY WORKER',

        'EXP_JOB_CLASS_TITLE': 'WATER UTILITY WORKER',

        'EXPERIENCE_LENGTH': '3',

        'EXPERIENCE_LEN_UNIT': 'years',

        'FULL_TIME_PART_TIME': 'FULL_TIME',

        'PAID_VOLUNTEER': 'PAID',

        'DRIVERS_LICENSE_REQ': 'R',

        'DRIV_LIC_TYPE': '',

        'ADDTL_LIC': 'NA'}

candidate_job_class_df = pd.DataFrame(data=data, index=[0])

df_edges = createEdges(job_class_df)
showPromotionalPaths(df_edges, job_class_df, '', candidate_job_class_df)
#Requirements based

showSimilarJobs(requirements,'ANIMAL KEEPER 4304 083118.txt','FILE_NAME', requirements_s_matrix, 3)
text = jobs_df['Content'].values[0] #sample bulletin text

ease_score, grade_level, ease_score_based_grade_level = flesch_reading_ease_grade(text)

print('Reading Ease Score for ' + jobs_df['FileName'].values[0] + ' bulletin is ' + str(ease_score))

print('Reading Ease Score based grade level for the bulletin is ' + ease_score_based_grade_level)
text = jobs_df['Content'].values[0]

biasednessResultCode = assessBias(text)

print('\nJob bulletin text for "' + jobs_df['FileName'].values[0]+ '" file is ' + biasednessResultCode)
#Test CandidateJobClass

data = {'JOB_CLASS_TITLE': 'WATER UTILITY WORKER',

        'EXP_JOB_CLASS_TITLE': 'WATER UTILITY WORKER',

        'EXPERIENCE_LENGTH': '1',

        'EXPERIENCE_LEN_UNIT': 'years',

        'FULL_TIME_PART_TIME': 'FULL_TIME',

        'PAID_VOLUNTEER': 'PAID',

        'DRIVERS_LICENSE_REQ': 'R',

        'DRIV_LIC_TYPE': '',

        'ADDTL_LIC': 'NA'}

candidate_job_class_df = pd.DataFrame(data=data, index=[0])

df_edges = createEdges(job_class_df)

showPromotionalPaths(df_edges, job_class_df, '', candidate_job_class_df)