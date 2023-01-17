!pip install transformers==3.0.2

# Hugging Face new library for datasets (https://huggingface.co/nlp/)

!pip install nlp
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import nlp # Hugginface extra datasets

from nlp import load_dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



np.random.seed(1234) 

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mnli = load_dataset(path='glue', name='mnli') # loading more data from the Huggin face dataset

snli   =  load_dataset("snli") # loading more data from the Huggin face dataset

xnli = load_dataset('xnli') # more data from the huggin face dataset
for i in range(len(xnli['test']['premise']) + len(xnli['validation']['premise'])):

    

    if i < len(xnli['test']['premise']):

        

        if i == 0:

            

            xnli_df = pd.concat([pd.Series(xnli['test']['premise'][i]), pd.Series(xnli['test']['hypothesis'][i]['translation'], index =\

                                                         xnli['test']['hypothesis'][i]['language'])],axis =1)

            

        else:

            

            xnli_df = pd.concat([xnli_df,\

                                pd.concat([pd.Series(xnli['test']['premise'][i]), pd.Series(xnli['test']['hypothesis'][i]['translation'], index =\

                                                         xnli['test']['hypothesis'][i]['language'])],axis =1)])

    else:

        

        xnli_df = pd.concat([xnli_df,\

                            pd.concat([pd.Series(xnli['validation']['premise'][i-len(xnli["test"]["premise"])]), pd.Series(xnli['validation']['hypothesis'][i-len(xnli["test"]["premise"]) ]['translation'], index =\

                                                         xnli['validation']['hypothesis'][i- len(xnli["test"]["premise"]) ]['language'])],axis =1)])





xnli_df.to_csv('xnli_org.csv')

        

            

            

        

        

# Loading Data



#import random



#random.seed(123) # setting random 



#train_df = pd.read_csv('../input/contradictory-my-dear-watson-augmented-dataset/only english original train.csv')

#print('Traning Data, the size of the dataset is: {} \n'.format(train_df.shape))

#display(train_df.head())

#test_df = pd.read_csv('../input/contradictory-my-dear-watson-augmented-dataset/TTA1 dear watson.csv')

#print('Test Data, the size of the dataset is: {} \n'.format(test_df.shape))

#display(test_df.head(10))

#print(train_df.shape)



#original_train_df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')
#import seaborn as sns

#import matplotlib.pyplot as plt



#fig = plt.figure(figsize = (15,5))



#plt.subplot(1,2,1)

#plt.title('Traning data language distribution')

#sns.countplot(data = train_df, x = 'lang_abv', order = train_df['lang_abv'].value_counts().index)



#plt.subplot(1,2,2)

#plt.title('Test data laguage distribution')

#sns.countplot(data = test_df, x = 'lang_abv', order = test_df['lang_abv'].value_counts().index)
def extract(dataframe, lang_abv):

    '''

    this functions takes a dataframe and returns a dataframe with the sentences only in the language selected

    '''

    specific_df = dataframe[dataframe['lang_abv'] == lang_abv]

    return specific_df



def trans_frame(dataframe, language):

    """

    translate a single language dataframe to a specific language

    

    """

    trans_data = []

    for i in range(dataframe.shape[0]):

        translator = Translator()

        

        trans_data.append([dataframe.iloc[i]['id'], translator.translate(dataframe.iloc[i]['premise'], dest = language).text \

                           , translator.translate(dataframe.iloc[i]['hypothesis'], dest = language).text,\

                           language, dataframe.iloc[i]['label']])

    

    return pd.DataFrame(trans_data, columns = list(dataframe.columns))

        

    



def round_trans(dataframe, language_list, number):



    '''

    this function takes a dataframe with several languages and augments it translating several data entries to other

    languages

    '''

    

    storage_list = [extract(dataframe, lang) for lang in language_list]

    

    for lang1 in language_list:

    

        except_list = [x for x in language_list if x != lang1]

    

        for lang2 in except_list:

           storage_list[language_list.index(lang2)] =  pd.concat([storage_list[language_list.index(lang2)],trans_frame(extract(dataframe, lang1).iloc[:number], lang2)])

    

    return storage_list    
#aug_list = round_trans(train_df, languages, 90)



#train_df = aug_list[0]



#for df in aug_list[1:]:

    

#   train_df = pd.concat([train_df, df])

    

#train_df.to_csv('augmentedata.csv', index = False)
#train_df = pd.read_csv('./augmentedata.csv')
#import json

#with open('../input/englishengen-synonyms-json-thesaurus/eng_synonyms.json') as json_file:  

#    synonyms_dict = json.load(json_file)
def syn_sentence(sentence, dictio):

    

    '''

    

    Function takes a sentence and substitues 4 random words from the sentece

    takes a sentence and a dictionary

    

    

    '''

    

    word_list = sentence.split()

    

    randomlist = random.sample(range(len(word_list)), 4)

    

    for ele in randomlist:

        

        syn = dictio.get(word_list[ele], [])

        

        if syn != []:

            

            word_list[ele] = syn[0]

            

    return ' '.join(str(i) for i in word_list)





def syn_df(df, dictio):

    

    '''

    

    this function will use the thesaurus augmentation technique to 

    make the english part of the dataset into a version with synonyms 

    

    

    '''

    

    columns = df.columns

    

    df = np.array(df).tolist()

    

    

    for i in range(len(df)):

        

        

        premise = df[i][0]

        hypothesis = df[i][1]

        

        if len(premise.split()) > 5 and len(hypothesis.split()) > 5:

        

            premise_syn = syn_sentence(premise, dictio)

            hypothesis_syn = syn_sentence(hypothesis, dictio)

        

            if premise != premise_syn or hypothesis != hypothesis_syn:

                

                df.append([premise_syn, hypothesis_syn, df[i][2], df[i][3]])

            

    return pd.DataFrame(df, columns = columns)    





def back_trans(df, aug_number, lang_list, random_seed, trans_rounds):

    

    """

    this function is used for backtranslation data aungmentation, the arguments are the dataframe,

    the number of extra datapoints that want to be generated, list of languages (except english) 

    that can be used for augmentation, random seed, how many back transaltion rounds are desired

    (keep in mind that english will always be the last language)

    """

    

    random.seed(random_seed)

    

    

    for i in range(aug_number):

        

        random_datapoint = df.iloc[random.sample(range(df.shape[0]), k = 1)]

    

        l_round = random.choices(lang_list, k = trans_rounds)

        

        l_round.append('en')

        

        for lan in l_round:

    

            translator = Translator()

        

            random_datapoint.iloc[0,0] = translator.translate(random_datapoint.iloc[0,0], dest = lan).text

            

            random_datapoint.iloc[0,1] = translator.translate(random_datapoint.iloc[0,1], dest = lan).text

            

        if i == 0:



            trans_df = random_datapoint



        else:



            trans_df = pd.concat([trans_df, random_datapoint])

            

    return trans_df



def TTA(df, lang_list, random_seed, trans_rounds):

    

    """

    Does TTA on a test dataset can take a multilingual dataset and the result will be only in english

    

    """

    

    random.seed(random_seed)

    

    

    for i in range(df.shape[0]):

        

        datapoint = df.iloc[i]

        

        l_round = random.choices(lang_list, k = trans_rounds)

        

        l_round.append('en')

        

        for lan in l_round:

            

            translator = Translator()

        

            datapoint[1] = translator.translate(datapoint[1], dest = lan).text

            

            datapoint[2] = translator.translate(datapoint[2], dest = lan).text

            

        if i == 0:



            trans_df = pd.DataFrame(datapoint).T



        else:



            trans_df = pd.concat([trans_df, pd.DataFrame(datapoint).T])

            

    return trans_df
#english_syn = syn_df(train_df[train_df['lang_abv'] == 'en'], synonyms_dict)



#train_df = train_df[train_df['lang_abv'] != 'en']



#train_df = pd.concat([train_df, english_syn])

    

#test_df4.to_csv('TTA4 dear watson.csv', index = False)


