# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import re
og_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

og_train = og_train.dropna()

test = test.dropna()



og_train = og_train.reset_index(drop=True)

test = test.reset_index(drop=True)



train_og_pos = og_train[og_train['sentiment'] == 'positive'] 

train_og_neg = og_train[og_train['sentiment'] == 'negative']



train_og_pos = train_og_pos.reset_index(drop=True)

train_og_neg = train_og_neg.reset_index(drop=True)
from sklearn.model_selection import train_test_split

train, val = train_test_split(og_train, test_size=0.25)

train = train.reset_index(drop=True)

val = val.reset_index(drop=True)



train_pos = train[train['sentiment'] == 'positive']

train_neg = train[train['sentiment'] == 'negative']



train_pos = train_pos.reset_index(drop=True)

train_neg = train_neg.reset_index(drop=True)
#lowercase the text

train_pos_text = train_pos['text'].str.lower()

train_neg_text = train_neg['text'].str.lower()



train_pos_target_text = train_pos['selected_text'].str.lower()

train_neg_target_text = train_neg['selected_text'].str.lower()





train_og_pos_text = train_og_pos['text'].str.lower()

train_og_neg_text = train_og_neg['text'].str.lower()



train_og_pos_target_text = train_og_pos['selected_text'].str.lower()

train_og_neg_target_text = train_og_neg['selected_text'].str.lower()
# removing special characters and numbers

train_pos_text = train_pos_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

train_neg_text = train_neg_text.apply(lambda x : re.sub("[^a-z\s]","",x) )



train_pos_target_text = train_pos_target_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

train_neg_target_text = train_neg_target_text.apply(lambda x : re.sub("[^a-z\s]","",x) )





train_og_pos_text = train_og_pos_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

train_og_neg_text = train_og_neg_text.apply(lambda x : re.sub("[^a-z\s]","",x) )



train_og_pos_target_text = train_og_pos_target_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

train_og_neg_target_text = train_og_neg_target_text.apply(lambda x : re.sub("[^a-z\s]","",x) )
# removing stopwords

from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))



train_pos_text = train_pos_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

train_neg_text = train_neg_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))



train_pos_target_text = train_pos_target_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

train_neg_target_text = train_neg_target_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))





train_og_pos_text = train_og_pos_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

train_og_neg_text = train_og_neg_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))



train_og_pos_target_text = train_og_pos_target_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

train_og_neg_target_text = train_og_neg_target_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))
#create the matrix of featurized sentences from data using spacy 

import spacy 

nlp = spacy.load('en_core_web_lg')



document_pos = nlp.pipe(train_pos_text)

pos_vector = np.array([tweet.vector for tweet in document_pos])



document_neg = nlp.pipe(train_neg_text)

neg_vector = np.array([tweet.vector for tweet in document_neg])





document_og_pos = nlp.pipe(train_og_pos_text)

og_pos_vector = np.array([tweet.vector for tweet in document_og_pos])



document_og_neg = nlp.pipe(train_og_neg_text)

og_neg_vector = np.array([tweet.vector for tweet in document_og_neg])
#iterate through words of text and create word feature, and append sentence feature to word ft., label (Y) is whether or the not the single word is part of the selected text

def featurize(text, selected_text, corpus_vect):

    labels = []

    featurized_data = []

    for i in range(len(text)):

        sent_vect = corpus_vect[i]

        target_text = selected_text[i]

        for word in text[i].split():

            word_vect = nlp(word).vector

            ft_vect = np.concatenate((word_vect, sent_vect))

            featurized_data.append(ft_vect.tolist())

            if word in target_text:

                labels.append(1)

            else:

                labels.append(0)

    return (featurized_data, labels)



(featurized_positive_X, featurized_positive_Y) = featurize(train_pos_text, train_pos_target_text, pos_vector)

(featurized_negative_X, featurized_negative_Y) = featurize(train_neg_text, train_neg_target_text, neg_vector)





(featurized_og_positive_X, featurized_og_positive_Y) = featurize(train_og_pos_text, train_og_pos_target_text, og_pos_vector)

(featurized_og_negative_X, featurized_og_negative_Y) = featurize(train_og_neg_text, train_og_neg_target_text, og_neg_vector)

from sklearn.linear_model import LogisticRegression

clf_pos = LogisticRegression(random_state=0, max_iter=1000).fit(featurized_positive_X, featurized_positive_Y)

clf_neg = LogisticRegression(random_state=0, max_iter=1000).fit(featurized_negative_X, featurized_negative_Y)



#clf_og_pos = LogisticRegression(random_state=0, max_iter=1000).fit(featurized_og_positive_X, featurized_og_positive_Y)

#clf_og_neg = LogisticRegression(random_state=0, max_iter=1000).fit(featurized_og_negative_X, featurized_og_negative_Y)
from sklearn.ensemble import RandomForestClassifier

clf_pos_RF = RandomForestClassifier().fit(featurized_positive_X, featurized_positive_Y)

clf_neg_RF = RandomForestClassifier().fit(featurized_negative_X, featurized_negative_Y)



clf_og_pos_RF = RandomForestClassifier().fit(featurized_og_positive_X, featurized_og_positive_Y)

clf_og_neg_RF = RandomForestClassifier().fit(featurized_og_negative_X, featurized_og_negative_Y)
#method to build the return string, uses the first word and last word of phrase to extract that portion from the original text (og_words)

def buildRetString(first_word, last_word, og_words):

    retStr = '';

    for i in range(len(og_words)):

        og_word = og_words[i]

        word = og_word.lower()

        word = re.sub("[^a-z\s]","", word)

        if word not in stopwords:

            if word != first_word:

                continue;

            else:

                temp_og_word = og_word

                temp_word = word

                retStr += temp_og_word

                i += 1

                while temp_word != last_word:

                    temp_og_word = og_words[i]

                    retStr += (' ' + temp_og_word)

                    temp_word = temp_og_word.lower()

                    temp_word = re.sub("[^a-z\s]","", temp_word)

                    i += 1

                return retStr
#preprocess the validation data

val_text = val['text'].str.lower()

val_text = val_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

val_text = val_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

#obtain feature vector for sentences

document_val = nlp.pipe(val_text)

val_vector = np.array([tweet.vector for tweet in document_val])



selected_text_clf_probs = []



#extra clfs

selected_text_RF = []





for i in range(len(val_text)):

    sent_vect = val_vector[i]

    if val['sentiment'][i] != 'neutral' and len(val_text[i].split()) > 2:

        temp_selected_text = []

        probabilities_words = {}

        

        #extra clfs

        probabilities_RF = {}        

        

        for word in val_text[i].split():

            word_vect = nlp(word).vector

            ft_vect = np.concatenate((word_vect, sent_vect))

            if val['sentiment'][i] == 'positive':

                clf = clf_pos

                clf_RF = clf_pos_RF

                

            else:

                clf = clf_neg

                clf_RF = clf_neg_RF

                

            probability_class_1 = clf.predict_proba([ft_vect])[:, 1]

            probabilities_words.update({word:(probability_class_1-0.5)})  

            

            #extra clfs                       

            probability_class_1 = clf_RF.predict_proba([ft_vect])[:, 1]

            probabilities_RF.update({word:(probability_class_1-0.5)})  



        

        words = val_text[i].split()

        subsets = [words[m:j+1] for m in range(len(words)) for j in range(m,len(words))]

        

        

        best_sum = 0;

        best_index = -1

        for j in range(len(subsets)):

            current_sum = 0

            for p in range(len(subsets[j])):

                current_sum += probabilities_words.get(subsets[j][p])

            if current_sum > best_sum:

                best_sum = current_sum

                best_index = j

        if best_index != -1:

            first_word = subsets[best_index][0]

            last_word = subsets[best_index][len(subsets[best_index])-1]

            og_words = val['text'][i].split()

            retStr = buildRetString(first_word, last_word, og_words)  

            #print(retStr)

            selected_text_clf_probs.append(retStr)

        else:

            selected_text_clf_probs.append(val['text'][i])

            

        #extra clfs                 

        best_sum = 0;

        best_index = -1

        for j in range(len(subsets)):

            current_sum = 0

            for p in range(len(subsets[j])):

                current_sum += probabilities_RF.get(subsets[j][p])

            if current_sum > best_sum:

                best_sum = current_sum

                best_index = j

        if best_index != -1:

            first_word = subsets[best_index][0]

            last_word = subsets[best_index][len(subsets[best_index])-1]

            og_words = val['text'][i].split()

            retStr = buildRetString(first_word, last_word, og_words)  

            #print(retStr)

            selected_text_RF.append(retStr)

        else:

            selected_text_RF.append(val['text'][i])           



        

    else:

        #neutral case

        selected_text_clf_probs.append(val['text'][i])

        

        #extra clfs

        selected_text_RF.append(val['text'][i])

        
#jacard score

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



ground_truth = val['selected_text']

prediction = pd.Series(selected_text_clf_probs)

jac_sum = 0

for i in range(len(prediction)):

    jac_sum += jaccard(prediction[i], ground_truth[i])



print('score(val) LR: ', (1/len(prediction)) * jac_sum)



#jacard score

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



ground_truth = val['selected_text']

prediction = pd.Series(selected_text_RF)

jac_sum = 0

for i in range(len(prediction)):

    jac_sum += jaccard(prediction[i], ground_truth[i])



print('score(val) RF: ', (1/len(prediction)) * jac_sum)
#preprocess the test data

test_text = test['text'].str.lower()

test_text = test_text.apply(lambda x : re.sub("[^a-z\s]","",x) )

test_text = test_text.apply(lambda x : " ".join(word for word in x.split() if word not in stopwords ))

#obtain feature vector for sentences

document_test = nlp.pipe(test_text)

test_vector = np.array([tweet.vector for tweet in document_test])



selected_text_clf_probs_og = []

for i in range(len(test_text)):

    sent_vect = test_vector[i]

    if test['sentiment'][i] != 'neutral' and len(test_text[i].split()) > 2:

        temp_selected_text = []

        probabilities_words = {}

        for word in test_text[i].split():

            word_vect = nlp(word).vector

            ft_vect = np.concatenate((word_vect, sent_vect))

            if test['sentiment'][i] == 'positive':

                clf = clf_og_pos_RF

            else:

                clf = clf_og_neg_RF

            probability_class_1 = clf.predict_proba([ft_vect])[:, 1]

            probabilities_words.update({word:(probability_class_1-0.5)})        

        

        words = test_text[i].split()

        subsets = [words[m:j+1] for m in range(len(words)) for j in range(m,len(words))]

        best_sum = 0;

        best_index = -1

        for j in range(len(subsets)):

            current_sum = 0

            for p in range(len(subsets[j])):

                current_sum += probabilities_words.get(subsets[j][p])

            if current_sum > best_sum:

                best_sum = current_sum

                best_index = j

        if best_index != -1:

            first_word = subsets[best_index][0]

            last_word = subsets[best_index][len(subsets[best_index])-1]

            og_words = test['text'][i].split()

            retStr = buildRetString(first_word, last_word, og_words)  

            selected_text_clf_probs_og.append(retStr)

        else:

            selected_text_clf_probs_og.append(test['text'][i])

        

    else:

        #neutral case

        selected_text_clf_probs_og.append(test['text'][i])

    
temp_series = pd.Series(selected_text_clf_probs_og)

submission['selected_text'] = temp_series

submission.to_csv('submission.csv', index=False)