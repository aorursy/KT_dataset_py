import numpy as np # linear algebra 

from math import log, sqrt # neperian logarithm, square root

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import nltk





## Read the train

train_df = pd.read_csv("../input/train.csv")



authors = {'EAP':0,'HPL':1,'MWS':2}

authors_rev = ['EAP','HPL','MWS']
nb_texts = len(train_df)

lambdas = (train_df.author.value_counts()/nb_texts).to_dict()

print("Lambdas :",lambdas)
N_train = len(train_df)



# cutting of the train set

cut = round(N_train * 0.95)



train = pd.DataFrame(train_df, index=range(cut))

test = pd.DataFrame(train_df, index=range(cut,N_train))
# built from a training corpus a dictionary of used words. For each word, we fill a dictionary that each author associates the number of times this word appeared in the author

def voca_Authors(train):

    voca = {}

    for i,line in train.iterrows():

        words = nltk.word_tokenize(line['text'])

        for word in words:

            word = word.lower()

            if not word in voca:

                voca[word] = {auth:0 for auth in authors}

                voca[word][line['author']] = 1

            else:

                voca[word][line['author']] += 1

    return voca



print('Computation of vocabulary dictionary.')

voca_authors = voca_Authors(train)

print('Dictionary size:',len(voca_authors),'\n20 examples from this dictionary :\n')

for i,word in enumerate(voca_authors):

    if i < 20:

        print(word,voca_authors[word])
voca_authors['of']
def vect_Voca(sentence, dict = voca_authors, zero = 0.01):

    words = nltk.word_tokenize(sentence)

    ret = [0]*len(authors)

    for word in words:

        word = word.lower()

        if word in dict:

            vect = [0]*len(authors)

            nb_appar = 0

            for auth in dict[word]:

                nb_appar += dict[word][auth]

            for auth in dict[word]:

                vect[authors[auth]] = dict[word][auth]/nb_appar/lambdas[auth] # vect[j] = x_j/lambda_j = p_j / (Sum lambda_j p_j)

            s = 1/sum(vect) # s = (Sum lambda_j p_j) 

            vect = [p_j_div_sum * s for p_j_div_sum in vect] # vect[j] = p_j



            for j,p_j in enumerate(vect):

                if p_j == 0:

                    ret[j] += log(zero)

                else:

                    ret[j] += log(p_j)

    return ret
# the matrix of sentences of the corpus of text transformed using vect_Voca

X = []

col = [] # the color according to the author: Poe => red, Lovecraft => green, Shelley => blue

# colours = {'EAP':'rgb(150, 5, 5)','HPL':'rgb(5, 150, 5)','MWS':'rgb(5, 5, 150)'}

colours = {'EAP':'r','HPL':'g','MWS':'b'}

# computation of X, the size matrix N_tests * 3

for i,line in test.iterrows():

    vect = vect_Voca(line['text'])

    X.append(vect)

    col.append(colours[line['author']])

X = np.array(X)



## The display



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt





fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X[:,0], X[:,1], X[:,2], c=col, marker='+')



from sklearn.neighbors import NearestNeighbors



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



voca_authors = voca_Authors(train)

# la matrice des phrases du corpus de texte transformées à l'aide de vect_Voca

M = [] # the train matrix

Sol = [] # the solutions of the train matrix

X = [] # the test matrix





for i,line in train.iterrows():

    vect = vect_Voca(line['text'])

    M.append(vect)

    Sol.append(line['author'])

M = np.array(M)



for i,line in test.iterrows():

    vect = vect_Voca(line['text'])

    X.append(vect)

X = np.array(X)





# The kNN algorithm

adjunction = 0.025

nb_ppv = 200 # number of nearest neighbors

M_ppv = NearestNeighbors(n_neighbors=nb_ppv)

M_ppv.fit(M)

tab_kneighbors = M_ppv.kneighbors(X, return_distance=False)



Probas = [] # the return



for i,line in test.iterrows():

    p = [0]*3

    kneighbors = tab_kneighbors[i]

    for kneighbor in kneighbors:

        p[authors[Sol[kneighbor]]] += 1

    s = sum(p)

    p = [x/s for x in p]

    Probas.append(p)



# harmonisation of the probabilities

for v in Probas:

    for i,prob in enumerate(v):

        v[i] = (prob + adjunction)/(1+3*adjunction)





submission = pd.read_csv('../input/sample_submission.csv')

submission.loc[:,['EAP', 'HPL', 'MWS']] = Probas

submission.to_csv("Log_likelihoof_on_vocabulary.csv", index=False)

submission.head()
nb_texts = len(train_df)

lambdas = (train_df.author.value_counts()/nb_texts).to_dict()

print("Lambdas :",lambdas)
N_train = len(train_df)



# découpe du train

cut = round(N_train * 0.95)



train = pd.DataFrame(train_df, index=range(cut))

test = pd.DataFrame(train_df, index=range(cut,N_train))
# construit à partir d'un corpus d'entraînement un dictionnaire des mots utilisés. Pour chaque mot, on remplit un dictionnaire qui à chaque auteur associe le nombre de fois que ce mot est apparu chez l'auteur

def voca_Authors(train):

    voca = {}

    for i,line in train.iterrows():

        words = nltk.word_tokenize(line['text'])

        for word in words:

            word = word.lower()

            if not word in voca:

                voca[word] = {auth:0 for auth in authors}

                voca[word][line['author']] = 1

            else:

                voca[word][line['author']] += 1

    return voca



print('Calcul du dictionnaire du vocabulaire.')

voca_authors = voca_Authors(train)

print('Taille du dictionnaire :',len(voca_authors),'\n20 exemples tirés de ce dictionnaire :\n')

for i,word in enumerate(voca_authors):

    if i < 20:

        print(word,voca_authors[word])
voca_authors['of']
def vect_Voca(sentence, dict = voca_authors, zero = 0.01):

    words = nltk.word_tokenize(sentence)

    ret = [0]*len(authors)

    for word in words:

        word = word.lower()

        if word in dict:

            vect = [0]*len(authors)

            nb_appar = 0

            for auth in dict[word]:

                nb_appar += dict[word][auth]

            for auth in dict[word]:

                vect[authors[auth]] = dict[word][auth]/nb_appar/lambdas[auth] # vect[j] = x_j/lambda_j = p_j / (Sum lambda_j p_j)

            s = 1/sum(vect) # s = (Sum lambda_j p_j) 

            vect = [p_j_div_sum * s for p_j_div_sum in vect] # vect[j] = p_j



            for j,p_j in enumerate(vect):

                if p_j == 0:

                    ret[j] += log(zero)

                else:

                    ret[j] += log(p_j)

    return ret
# la matrice des phrases du corpus de texte transformées à l'aide de vect_Voca

X = []

col = [] # la couleur en fonction de l'auteur : Poe=>rouge, Lovecraft=>vert, Shelley=>bleu



# calcul de X la matrice de taille N_tests * 3

for i,line in test.iterrows():

    vect = vect_Voca(line['text'])

    X.append(vect)

    col.append(colours[line['author']])

X = np.array(X)



## L'affichage



#import plotly

#from plotly.graph_objs import Scatter, Layout, Scatter3d

fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X[:,0], X[:,1], X[:,2], c=col, marker='+')

#plotly.iplot({

 #   "data": [Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', 

 #   marker=dict(size=4,

 #       color=col, 

  #      opacity=0.7

  #  ))],

  #  "layout": Layout(title="vect_Voca", scene=dict(camera= dict(

 #   up=dict(x=0, y=0, z=1),

  #  center=dict(x=0, y=0, z=0),

  #  eye=dict(x=1.5, y=0.75, z=0.475)

#)))

#})
from sklearn.neighbors import NearestNeighbors



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



voca_authors = voca_Authors(train)

# la matrice des phrases du corpus de texte transformées à l'aide de vect_Voca

M = [] # the train matrix

Sol = [] # the solutions of the train matrix

X = [] # the test matrix





for i,line in train.iterrows():

    vect = vect_Voca(line['text'])

    M.append(vect)

    Sol.append(line['author'])

M = np.array(M)



for i,line in test.iterrows():

    vect = vect_Voca(line['text'])

    X.append(vect)

X = np.array(X)





# The kNN algorithm

adjunction = 0.025

nb_ppv = 200 # number of nearest neighbors

M_ppv = NearestNeighbors(n_neighbors=nb_ppv)

M_ppv.fit(M)

tab_kneighbors = M_ppv.kneighbors(X, return_distance=False)



Probas = [] # the return



for i,line in test.iterrows():

    p = [0]*3

    kneighbors = tab_kneighbors[i]

    for kneighbor in kneighbors:

        p[authors[Sol[kneighbor]]] += 1

    s = sum(p)

    p = [x/s for x in p]

    Probas.append(p)



# harmonisation of the probabilities

for v in Probas:

    for i,prob in enumerate(v):

        v[i] = (prob + adjunction)/(1+3*adjunction)





submission = pd.read_csv('../input/sample_submission.csv')

submission.loc[:,['EAP', 'HPL', 'MWS']] = Probas

submission.to_csv("Log_likelihoof_on_vocabulary.csv", index=False)

submission.head()