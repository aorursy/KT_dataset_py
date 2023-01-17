# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

 # Any results you write to the current directory are saved as output.

#get data

#strange UnicodeDecodeError with provided debate file

text = []

debate = pd.read_csv('../input/debate.csv', encoding = "ISO-8859-1")



#only want Trump and Clinton

debate = debate.query("Speaker == 'Trump' or Speaker == 'Clinton'")

debate.head()
#function for cleaning words

from nltk.corpus import stopwords #stop word list

import re

def clean_text(text):

    if text is None:

        print("Text is None!")

        return None

    

    stops = set(stopwords.words("english")) #stopwords list

    

    #add proper nouns to stops

    stops.update(['anderson', 

                 'donald', 

                 'clinton', 

                 'trump', 

                 'hillary', 

                 'kaine', 

                 'pence', 

                 'raddatz', 

                 'martha'])

    letters_only = re.sub("[^a-zA-z]", " ", text)

    lower_case = letters_only.lower()

    split = lower_case.split() #gives list of words lower case



    #find meaningful words

    meaningful_words = [w for w in split if not w in stops] #takes out no

    

    if len(meaningful_words) == 0:

        return None

    

    #for scikit countvectorizer to work, must return string

    final_string = " ".join(meaningful_words)

    return final_string

    

#create bag of words and seperate things into a trump and clinton text    

from sklearn.feature_extraction.text import TfidfVectorizer



split_text = {'Trump': [], 'Clinton': []}

for index, row in debate.iterrows():

    meaningful_sentence = clean_text(row['Text'])

    if meaningful_sentence:

        split_text[row['Speaker']].append(meaningful_sentence) 



full_text = split_text['Trump'] + split_text['Clinton']



#make for top 1000 words

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,

                                   max_features=1000)

bag_words = vectorizer.fit_transform(full_text)



#features

full_debate_feature_names = vectorizer.get_feature_names()

print(full_debate_feature_names[0:3]) #in alphabetical order cool
'''

takes in the bag of words and runs NMF for the number of topics specified, also for the number of top words

'''

def nmf(bag_of_words, num_topics):



	from sklearn import decomposition



	nmf = decomposition.NMF(n_components=num_topics, random_state=1)

	nmf.fit(bag_of_words)

	return nmf



def top_words(nmf_model, num_top_words):

    #print words associated with topics

	topic_words = []



	for topic in clf.components_:

		word_idx = np.argsort(topic)[::-1][0:num_top_words]

		topic_words.append([vocab[i] for i in word_idx])



	print("show top 15 words")

	for t in range(len(topic_words)):

		print("topic {}: {}".format(t, ' '.join(topic_words[t][:15])))



	return topic_words



#train NMF on full 2 debate vocabulary and results

#limit to top 10 topics, 5 for each debator

full_debate_nmf = nmf(bag_words, 10)



def print_feature_names(nmf_model, feature_names, n_top_words):

    for index, topic in enumerate(nmf_model.components_):

        print("Topic #%d in order of most popular words:" % index)

        top_i = topic.argsort()[len(topic) - n_top_words - 1: len(topic)-1]

        print(str([feature_names[i] for i in top_i]))

        

#get full debate features



print_feature_names(full_debate_nmf, full_debate_feature_names, 20)
#LD analysis on topics for whole debate

count_vectorizer = CountVectorizer(max_df=0.95, min_df=2,

                                max_features=1000)
#word clouds suck, but hey.

from wordcloud import WordCloud
