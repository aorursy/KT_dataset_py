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
!pip install bert-extractive-summarizer
!pip install transformers==2.2.0

!pip install spacy==2.0.12
paragraphs = pd.read_excel('/kaggle/input/task-uj/TASK.xlsx', headers=True) #read dataset 
paragraphs['introduction'] = paragraphs['Unnamed: 1'] #cleanup column
paragraphs = paragraphs.drop(columns=['TEST DATASET','Unnamed: 1'],axis=1)
paragraphs = paragraphs.drop(paragraphs.index[0])#cleanup rows
new_index = list(range(0,len(paragraphs)))#reorder index
paragraphs['index'] = new_index
paragraphs = paragraphs.set_index('index')
print(len(paragraphs))
paragraphs.head()
from summarizer import Summarizer,TransformerSummarizer
#get input for the desired word count of the summary
sum_len=int(input('Enter length of summary? For better performance min_lenght is 60 '))

#initialize variables
summary = [] # for storing summaries

model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")

#generating summaries using ransformerSummarizer 
for i in range(0,len(paragraphs)):
        summary.append(model(paragraphs.introduction[i],min_length=sum_len))

#adding the summary to dataframe
paragraphs['summary'] = summary


#print top 5 results
#paragraphs.to_excel("output.xlsx")
paragraphs.head()
#get input for the desired word count of the summary
sum_len=int(input('Enter length of summary? For better performance min_lenght is 60 '))

#initialize variables
summary = [] # for storing summaries

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")

#generating summaries using gensim summarizer #bert_summary = ''.join(bert_model(introduction, min_length=60))
for i in range(0,len(paragraphs)):
        summary.append(GPT2_model(paragraphs.introduction[i],min_length=sum_len))

#adding the summary to the main dataframe
paragraphs['summary'] = summary

#print top 5 results
#execute the following code to output the code as an excel file marked as 'output.xlsx'
#paragraphs.to_excel("output.xlsx")
paragraphs.head()
#get input for the desired word count of the summary
sum_len=int(input('What is the desired word count of the summary? Ideal length is around 50 '))

#initialize variables
summary = [] # for storing summaries

bert_model = Summarizer()

#generating summaries using gensim summarizer 
for i in range(0,len(paragraphs)):

        
        summary.append(bert_model(paragraphs.introduction[i],min_length=sum_len))

#adding the summary to the main dataframe
paragraphs['summary'] = summary

#execute the following code to output the code as an excel file marked as 'output.xlsx'
#paragraphs.to_excel("output.xlsx")
paragraphs.head()
paragraphs.to_excel("/kaggle/working/output11.xlsx")
