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



# Any results you write to the current directory are saved as output.<span style="color:red">some **This is Red Bold.** text</span>
import numpy as np                             # linear algebra

import pandas as pd                            # Data processing, CSV file I/O (e.g. pd.read_csv)

import re                                      # for Data Cleaning 

import matplotlib.pyplot as plt                # For Visualization  

import re                                      # For data Cleaning 



from tqdm import tqdm                          # For ProgressBar 

import nltk                                    # For preprocessing

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer



from sklearn import metrics                    # For Accuracy Measure 

from sklearn.metrics import accuracy_score

# Function To Load Data from File into Pandas Datframe

def Load_data(Data_path,show_info = False):

    data = pd.read_csv(Data_path)

    if show_info:

        print(data.info())

    return data
Data_File_Location = "../input/mbti-type/mbti_1.csv"

data = Load_data(Data_File_Location,True)
# Function To check Classes in data 

def count_class(DataFrame,count = False ,plot = False):

    # considering 1st Column is for classes 

    Classes = list(data[data.columns[0]].unique())

    #print(Classes)

    if plot or count:

        count_type = data.groupby('type').count()

    if count : print(count_type)

    if plot:

        fig = plt.figure()

        ax = fig.add_axes([0,0,2,2])

        count_type_temp = count_type.sort_values('posts')

        ax.bar(count_type_temp.index,count_type_temp['posts'])

        plt.show()

    return Classes
classes = count_class(data,True,True)
data.iloc[0][1]
data.columns
data.shape[0]
# Function To replace "|||" from text with " " Join all texts written by 1 Person 

def replace_sep(text):

    """Remove '|||' which is used as seprator """

    text = text.replace("|||"," ")

    return text



# Function To remove Links from text and replace them with 'Link' 

def remove_link(text):

    """Replace Links from text to 'Link' """

    text = re.sub(r"http\S+", "Link", text, flags=re.MULTILINE)

    return text



# Function To Remove punctuation from Text 

def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words

    
replace_sep(data.iloc[0][1])
remove_link(data.iloc[0][1])
def pre_processing_stage_1(text):

    text = replace_sep(text)  # Calling Function to remove "|||" seprator and join all texts 

    text = remove_link(text) # calling function to removes Links 

    text = text.lower()     # To convert whole text To lower

    return text
pre_processing_stage_1(data.iloc[0][1])
def pre_processing_stage_2(text):

    tokenized_text = word_tokenize(text)

    for word in tokenized_text:

        if word in stopwords.words('english'):

            tokenized_text.remove(word)

    tokenized_text = remove_punctuation(tokenized_text)

    for i in range(len(tokenized_text)):

        tokenized_text[i] = stemmer.stem(tokenized_text[i])  # 

    final_text = " ".join(tokenized_text)

    return final_text
stemmer = PorterStemmer()                     # Defining Stemmer for Stemming in pre_processing_stage_2



def Clean_Data(df):

    print("PreProcessing----------- ")

    for i in tqdm(range(df.shape[0])):

        text = df.iloc[i][1]                      # Getting data from DataFrame to Text varibale to Preprocess

        text = pre_processing_stage_1(text)       # calling Function to merge texts and Do 1st level pre-processing  

        text = pre_processing_stage_2(text)

        df.set_value(i,'posts',text)

    return df
data = Clean_Data(data)
print(data.iloc[0][1])
#data.to_csv('Data.csv')
# This Function Will be used to make data equal for all Class 

def up_down_sampling(data,count):

    types = list(set(data.type))

    defined = False

    for tp in types :

        print(tp)

        if not defined:

            defined = True

            tp_class_count = data.type.value_counts()[tp]

            if tp_class_count > count :

                df = data[data['type'] == tp].sample(count)

            else:

                df = data[data['type'] == tp].sample(count,replace = True)

        else:

            tp_class_count = data.type.value_counts()[tp]

            if tp_class_count > count :

                df = pd.concat([df, data[data['type'] == tp].sample(count)], axis=0)

            else:

                df = pd.concat([df, data[data['type'] == tp].sample(count,replace = True)], axis=0)

    return df

            
# Using Up_down Sampling for preparing trainable data   

df = up_down_sampling(data,600)
# Checking Trainable data 

count_class = df.type.value_counts()

count_class
#Suffling DataFrame 

df = df.sample(frac = 1)
# Text Written By (Input)

text = df.posts



# Personality Type (OutPut)

cator = df.type
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

vector = CountVectorizer(ngram_range = (2,2))
vector.fit(text)
X = vector.transform(text)
Y = np.array(cator)#.reshape(-1,1)
tfidf_transformer = TfidfTransformer()
X_final =tfidf_transformer.fit_transform(X) 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_final,Y,test_size = 0.20)
from sklearn.naive_bayes import MultinomialNB
# Defining Model 

cls = MultinomialNB()
# training Model 

cls.fit(X_train,Y_train)
# Testing model on Test Set 

res = cls.predict(X_test)
print("Accuracy Of Model 1 is :",accuracy_score(res,Y_test)*100)
from sklearn import tree
# Defining Model 

classifier_2 = tree.DecisionTreeClassifier()
# Training Model 

classifier_2.fit(X_train,Y_train)
# Testing Model on Test Data 

result_2 = classifier_2.predict(X_test)
print("Accuracy Of Model 2 is :",accuracy_score(result_2,Y_test)*100)