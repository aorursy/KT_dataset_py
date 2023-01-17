# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#let's imprt the necessary libraries

import numpy as np

import pandas as pd

import os

import json

import glob
#Creating an empty dataframe with only column names to fill it with files content

df = pd.DataFrame(columns=['Doc_ID', 'Title', 'Text', 'Source'])
#Grabbing the files from the repositories using glob library



json_filenames = glob.glob(f'/kaggle/input/CORD-19-research-challenge/2020-03-13/**/**/*.json', recursive=True)
#Taking a look at the first 10 filenames path 

json_filenames[:10]
def get_df(json_filenames, df):



    for file_name in json_filenames:



        row = {"Doc_ID": None, "Title": None, "Text": None, "Source": None}



        with open(file_name) as json_data:

            data = json.load(json_data)

            

            #getting the column values for this specific document

            row['Doc_ID'] = data['paper_id']

            row['Title'] = data['metadata']['title']            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                except:

                    pass



            body = " ".join(body_list)

            row['Text'] = body

            

            # Now just add to the dataframe. 

            row['Source'] = file_name.split("/")[5]

            

            df = df.append(row, ignore_index=True)

    

    return df
#converting json file to dataframe

corona_dataframe = get_df(json_filenames, df)
#reading first 10 head 

corona_dataframe.head(10)
#reading tail information

corona_dataframe.tail(5)
out = corona_dataframe.to_csv('covid-19_csv_format.csv')
#reading the csv with pandas

df=pd.read_csv('covid-19_csv_format.csv')
df
# spaCy based imports

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English
nlp = spacy.load('en_core_web_lg')
#NER extraction

doc = nlp(corona_dataframe["Text"][10])

spacy.displacy.render(doc, style='ent',jupyter=True)
pd.read_csv('covid-19_csv_format.csv').to_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/**/**/submission.csv')