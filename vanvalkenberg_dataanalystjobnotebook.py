# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DataFrame = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
DataFrame.info()
## lets have a look
## at first 10 values
DataFrame.loc[0 : 9]
## lets see which industry has the highest jobs to offer
import matplotlib.pyplot as plt
def plotter(param, titleOfGraph, xlbl, ylbl, threshold):
        uniqueVals = set(DataFrame[param])
        
        labels = []
        counts = []
        
        for mem in uniqueVals:
            if mem == '-1' or mem == -1:
                continue
            
            val = len (DataFrame[DataFrame[param] == mem])
            if val < threshold:
                continue
            
            labels.append(mem)
            counts.append(val)
        
        plt.rcParams.update({'font.size': 30})
        plt.figure(figsize = (50, 20))
        plt.bar(labels, counts)
        plt.title(titleOfGraph)
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        plt.xticks(rotation='vertical')
        plt.grid(True)
    
    
plotter('Industry', 'Distribution of Jobs IndustryWise', 'Industry type', 'Available Jobs', 0)
## From the above graph it easy to conclude more jobs oppurtunities are present in the below Industries
## 1.) IT services
## 2.) staffing and outSourcing 
## 3.) HealthCare Service and Hospitals
## 4.) Consulting
## 5.) Investment banking and Asset MAnagement
## 6.) Banks and Credit Unions
## 7.) EnterPrise software and Network solution
## and so on .......
## let see the job Distrinution Sector wise
plotter('Sector', 'Job Distribution SectorWise', 'Sector', 'Available jobs', 0)
## we can easily say the following 4 sector have most job openings
## 1.) Information and technology
## 2.) Business Services 
## 3.) Finance 
## 4.) HealthCare

## let us also see the jobs distribution location wise 
## Since there were to locations here we will only Display those
## locations that has atleat 5 jobs
plotter('Location', 'Job Distribution LocationWise', 'LOCATIONS', 'Available jobs', 5)
## clearly the following can be observed
## 1.) New York has highest number of jobs
## 2.) Choicago offers 2nd highest number of jobs
## 3.) San Francisco offer 3rd highest 
## and so on ...
## lets study how pay is distributed
plotter('Salary Estimate', 'Salary Distribution', 'Salaries', 'number of jobs in a given group', 0)
## lets also get insight about the distribution of companies acording to the number of employees employed
plotter('Size', 'Company Size', 'work force size', 'number of companies in a group', 0)
## lets plot wordClouds regarding job Posting and job descriptions
from wordcloud import WordCloud, STOPWORDS 


def WordCloundPlotter(param, title):
    stopwords = set(STOPWORDS)
    wordArr = ""
    wa  = np.array(DataFrame[param])
    
    for mem in wa:
        wordArr = wordArr + mem + " "
    
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(wordArr) 
    
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.title(title)
    plt.tight_layout(pad = 0) 

WordCloundPlotter('Job Description', 'Word Cloud for Job Description')
WordCloundPlotter('Job Title', 'WordCloud for Job Titles')