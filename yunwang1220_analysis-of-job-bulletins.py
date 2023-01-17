# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# install below 2 package as they are not installed by default

!pip install vaderSentiment

!pip install find_job_titles

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.



strBulletinsPath = '../input/cityofla/CityofLA/Job Bulletins/'

#strCSVPath="C:/Work/Bulletins.csv"

arrFileList  = os.listdir(strBulletinsPath)

strAlterReq = "REQUIREMENTS/MINIMUM QUALIFICATION"

dicSectionCounter = dict() #Count each section's occurance



def get_second_part(strLine):

    strPart = strLine.split(':')

    if len(strPart) > 1:

        return strPart[1].replace(' ','')

    else:

        return 0



arrCSV = []

strNextKeyWord = ''

arrKeyWord = ['ANNUAL SALARY',

              'DUTIES',

              'REQUIREMENT/MINIMUM QUALIFICATION',

              'REQUIREMENTS',

              'PROCESS NOTE',

              'WHERE TO APPLY',

              'NOTE',

              'APPLICATION DEADLINE',

              'SELECTION PROCESS',

              'QUALIFICATIONS REVIEW',

              'NOTICE'

              ]



for objFile in arrFileList:

    blnHeader = False #In case some files got first few lines empty

    blnFlag = False #For same file, once checked header empty lines, the rest empty lines in the middle of the file will not be processed specially

    blnClassCd = False #One file has many class code, only first one required

    strPath = os.path.join(strBulletinsPath, objFile)

    fileHandle = open(strPath, encoding = "ISO-8859-1")



    objItem = { 'FILE_NAME': objFile }

    for kw in arrKeyWord:

        objItem[kw] = ''  #Initial value blank

        

    for i, strLine in enumerate(fileHandle.readlines()):

        if not blnHeader and not blnFlag and strLine.strip() != "":

            blnHeader = True

            blnFlag = True

        if i == 0 or blnHeader:

            # JOB_CLASS_TITLE

            objItem['JOB_CLASS_TITLE'] = strLine

            blnHeader = False

            continue

        if 'Class Code' in strLine and not blnClassCd:

             # JOB_CLASS_CODE

            objItem['JOB_CLASS_CODE'] = get_second_part(strLine)

            blnClassCd = True

            continue

        if 'Open Date' in strLine or 'Open date' in strLine:

            objItem['OPEN_DATE'] = get_second_part(strLine)

            continue

        blnNewKeyword = False

        for kw in arrKeyWord:

            if kw in strLine:

                strNextKeyWord = kw

                blnNewKeyword = True

                if kw != 'NOTE':

                    if len(dicSectionCounter) == 0:

                        dicSectionCounter[kw] = 1

                    else:

                        if kw in dicSectionCounter:

                            dicSectionCounter[kw] = dicSectionCounter[kw]+1

                        else:

                            dicSectionCounter[kw] = 1

                break

            #in case in CSV, the title is not "REQUIREMENT/MINIMUM QUALIFICATIONS"    but "REQUIREMENTS/MINIMUM QUALIFICATION"

            if not blnNewKeyword:

                if strAlterReq in strLine:

                    strNextKeyWord = "REQUIREMENT/MINIMUM QUALIFICATION"               

                    blnNewKeyword = True

                    break

        if blnNewKeyword:

            continue

        if strNextKeyWord == '':

            continue

        objItem[strNextKeyWord] += strLine

    arrCSV.append(objItem)

    blnHeader = False

    blnFlag = False

    blnClassCd = False

                

df = pd.DataFrame(arrCSV)



def clean_line(objRow):

    for col in df.columns:

        objRow[col] = str(objRow[col]).replace('\n','').replace('\n52','')

    return objRow



df = df.apply(clean_line, axis=1)

#Here can generate CSV file.

#df.to_csv(strCSVPath)

df.head()
import textblob

import pandas as pd

import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, RegexpTokenizer

from wordcloud import WordCloud

from collections import Counter



analyzer = SentimentIntensityAnalyzer()



#Read CSV

#strCSVPath="C:/Work/Bulletins.csv"

#df = pd.read_csv(strCSVPath)

df1 = df

# Below fields are which we need to analyze

arrCols = ['DUTIES', 'NOTE', 'NOTICE', 'PROCESS NOTE', 'QUALIFICATIONS REVIEW', 'REQUIREMENT/MINIMUM QUALIFICATION', 'REQUIREMENTS', 'SELECTION PROCESS', 'WHERE TO APPLY']



arrNegative=[]

arrItem=[]

arrKeyWord = ['FILE_NAME', 

              'ITEM',

              'TEXT',

              'POLARITY',

              'SUBJECTIVITY'

              ]



def get_sentiment(row):

    row['polarity'] = 0

    row['subjectivity'] = 0



    for col in arrCols:

        row[col] = str(row[col]).replace('.','. ').replace('/',' / ') 

        polarity_col = col + '_Polarity'

        subjectivity_col = col + '_Subjectivity'

        #get each section's polarity and subjectivity value

        blob = textblob.TextBlob(row[col])

        row[polarity_col] = blob.sentiment.polarity

        row[subjectivity_col] = blob.sentiment.subjectivity

        if row[polarity_col] < 0:

            objItem = dict()

            for kw in arrKeyWord:

                objItem[kw] = ''  #Initial value blank        

            objItem['FILE_NAME'] = row['FILE_NAME']

            objItem['ITEM'] = col

            objItem['TEXT'] = row[col]

            objItem['POLARITY'] = row[polarity_col]

            objItem['SUBJECTIVITY'] = row[subjectivity_col]

            arrNegative.append(objItem)

            arrItem.append(col)



        row['polarity'] = row['polarity'] + blob.sentiment.polarity

        row['subjectivity'] = row['subjectivity'] + blob.sentiment.subjectivity

    return row



df1 = df1.apply(get_sentiment, axis=1)

Neg = pd.DataFrame(arrNegative)

Neg.head()

#Neg.to_csv('C:/Work/Negative.csv')
# lets see the histograms for a starter

#%matplotlib notebook

import seaborn as sns

import matplotlib.pyplot as plt

#plt.axis("on")

from scipy import stats



for col in arrCols:

    #Here we only show the graphs for "SELECTION PROCESS" as one example.

    if col == "SELECTION PROCESS":

        sns.set(color_codes=True)

        fig, ax = plt.subplots(nrows=1, ncols=2)

        sns.distplot(tuple(df1[col + '_Polarity']), ax=ax[0])

        sns.distplot(tuple(df1[col + '_Subjectivity']), ax=ax[1])



        g = sns.jointplot(tuple(df1[col + '_Polarity']), tuple(df1[col + '_Subjectivity']), kind="scatter", height=7, space=0)

        plt.show()
#This is the bar chart

arrSection = []

cnt = Counter(arrItem)

for key, value in cnt.items():

    dicTemp = dict()

    dicTemp['Section'] = key

    dicTemp['Counter'] = value

    arrSection.append(dicTemp)



Bar = pd.DataFrame(arrSection)

plt.figure(figsize=(7,5))

sns.barplot(x='Counter',y='Section',palette='rocket',data=Bar) 

plt.title('Negative Sections')

plt.xlabel("Count")

plt.ylabel('Section')

plt.gcf().subplots_adjust(left=0.3)
neg_word_list=[]

def get_word_sentiment(text):

    tokenized_text = nltk.word_tokenize(text)

    return tokenized_text

    



dicFreq = dict()



#use anoteher package SentimentIntensityAnalyzer to get the most negative words

for item in arrNegative:

    if item['ITEM'] == 'REQUIREMENTS' or item['ITEM'] == 'REQUIREMENT/MINIMUM QUALIFICATION':

        for word in get_word_sentiment(item['TEXT']):

            if (analyzer.polarity_scores(word)['compound']) <= -0.1:

                if len(dicFreq) == 0:

                    dicFreq[word]=1         

                else:

                    if word in dicFreq:

                        dicFreq[word] = dicFreq[word]+1

                    else:

                        dicFreq[word]=1

#use word cloud to draw the word board

wc = WordCloud(relative_scaling=1, background_color='white',

        max_words=50)



wordcloud = wc.generate_from_frequencies(dicFreq)

fig = plt.figure(1, figsize=(12, 12))

#fig.set_size_inches(18.5, 10.5)

plt.figure()

plt.axis('off')

#fig.suptitle('Most negative Words in Selection Process', fontsize=20)

fig.subplots_adjust(top=2.3)

plt.imshow(wordcloud, interpolation="bilinear")

plt.show()
neg_word_list=[]

dicFreq = dict()



#use anoteher package SentimentIntensityAnalyzer to get the most negative words

for item in arrNegative:

    for word in get_word_sentiment(item['TEXT']):

        if (analyzer.polarity_scores(word)['compound']) <= -0.1:

            if len(dicFreq) == 0:

                dicFreq[word]=1         

            else:

                if word in dicFreq:

                    dicFreq[word] = dicFreq[word]+1

                else:

                    dicFreq[word]=1



#use word cloud to draw the word board

wc = WordCloud(relative_scaling=1, background_color='white',

        max_words=50)



wordcloud = wc.generate_from_frequencies(dicFreq)

fig = plt.figure(1, figsize=(12, 12))

#fig.set_size_inches(18.5, 10.5)

plt.figure()

plt.axis('off')

#fig.suptitle('Most negative Words', fontsize=20)

fig.subplots_adjust(top=2.3)

plt.imshow(wordcloud, interpolation="bilinear")

plt.show()
arrSecCnt = []

for key, value in dicSectionCounter.items():

    dicTemp = dict()

    dicTemp['Section'] = key

    dicTemp['Counter'] = value

    arrSecCnt.append(dicTemp)



Bar = pd.DataFrame(arrSecCnt)

Bar.head()

plt.figure(figsize=(7,5))

plt.gcf().subplots_adjust(left=0.3)

sns.barplot(x='Counter',y='Section',palette='vlag',data=Bar) 

plt.title('Sections Counter')

plt.xlabel("Count")

plt.ylabel('Section')
!pip install geotext

from geotext import GeoText



dicCity = dict()

# use GeoText to get city name in requirement section

for item in arrCSV:

    places = GeoText(item['REQUIREMENT/MINIMUM QUALIFICATION'] + item['REQUIREMENTS'])

    lstCity = places.cities

    if len(lstCity) > 0:

        for city in lstCity:

            if len(dicCity) == 0:

                dicCity[city]=1         

            else:

                if city in dicCity:

                    dicCity[city] = dicCity[city]+1

                else:

                    dicCity[city]=1



print('City mentioned times')

print('------------------------------------')

for key, value in dicCity.items():

    print(key + '    '+str(value)+'\n')

print('Example:' + '\n')

for item in arrCSV:

    places = GeoText(item['REQUIREMENT/MINIMUM QUALIFICATION'])

    lstCity = places.cities

    if 'Los Angeles' in lstCity:

        print(item['REQUIREMENT/MINIMUM QUALIFICATION'])

        break
import pandas as pd

from find_job_titles import FinderAcora

from graphviz import Digraph



#Read CSV

#strCSVPath="C:/Work/Bulletins.csv"

dot = Digraph(comment='Promotions')



finder=FinderAcora()

strReq = "REQUIREMENT/MINIMUM QUALIFICATION"

arrRelation=[]

arrFinal=[]



def get_promotion(row):

    strLine = str(row [strReq]) #only check Requirement/minimum qualification section to get the promotion path

    # usually the promotion sentences start with "as a" and finish with "with"

    if strLine.find("as a") > 0:

        objItem = dict() 

        # only one previous position required

        #if strLine.find("or") < 0:

        strTemp = strLine[strLine.find("as a"):strLine.find("with")-1]

        strTemp = strTemp.replace("as an", "as a").replace("as a ","")

        if strTemp != "":

            for m in finder.findall(strTemp):

                objItem['Job1'] = row["JOB_CLASS_TITLE"].lstrip().rstrip().upper()

                objItem['Job2'] = m[2].lstrip().rstrip().upper()

                arrRelation.append(objItem)

    return row

    

df = df.apply(get_promotion, axis=1)



pro = pd.DataFrame(arrRelation)

#pro.head()

#pro.to_csv('C:/Work/Promotion.csv')

pro = pro.drop_duplicates()

#Job1 is higher role and Job1 is the role which could promote to Job1

for index, row in pro.iterrows():

# as the image is too large, here just take police related jobs as an example

    if row["Job2"].startswith("POLICE"):

        dot.edge(row["Job2"], row["Job1"], label='Promotion')

dot