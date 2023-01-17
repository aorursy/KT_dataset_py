import gensim

import nltk
from nltk.data import find

word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))

model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
model.most_similar(positive=['medium'], topn = 3)
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def similar_questions(qus):

    dict1 = {}

    pos_tag = nltk.pos_tag(word_tokenize(qus))

    for i in pos_tag:

        if(i[1] in ['JJ', 'JJR','JJS']):

            try:

                dict1.update({i[0]:[k for k in [j[0] for j in model.most_similar_cosmul(positive=[i[0]], topn = 10)] if(k!=lemmatizer.lemmatize(i[0], "v") and k[:3].lower()!=i[0][:3].lower() and k.lower()!=i[0].lower())][:2]})

            except:

                print("Not trained on this word")

        elif(i[1] in ['VB', 'VBD','VBG','VBN','VBP']):

            try:

                dict1.update({i[0]:[k for k in [j[0] for j in model.most_similar_cosmul(positive=[i[0]], topn = 10)] if(k!=lemmatizer.lemmatize(i[0], "v") and k[:3].lower()!=i[0][:3].lower() and k.lower()!=i[0].lower())][:2]})

            except:

                print("Not trained on this word")

        elif(i[1] in ['NN', 'NNS','NNP','NNPS']):

            try:

                dict1.update({i[0]:[k for k in [j[0] for j in model.most_similar_cosmul(positive=[i[0]], topn = 10)] if(k!=lemmatizer.lemmatize(i[0], "v") and k[:3].lower()!=i[0][:3].lower() and k.lower()!=i[0].lower())][:2]})

            except:

                print("Not trained on this word")

    list1=[]

    for i in range(2):

        for k in dict1:

            list1.append(qus.replace(k,dict1[k][i]))

    return list1
qus1 = "Where is the official home of Santa Claus?"

print("="*20, "Similar Questions", "="*20)

similar_questions(qus1)
qus2 = "What is the largest country in the world in terms of land area?"

print("="*20, "Similar Questions", "="*20)

similar_questions(qus2)
qus3 = "At which place in the world, Narendra Modi wax statue was unveiled in a museum?"

print("="*20, "Similar Questions", "="*20)

similar_questions(qus3)