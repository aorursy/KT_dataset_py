import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re
os.chdir('../input')

os.getcwd()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

data = pd.read_csv('/kaggle/input/haiwaiian-hotel-reviews/Hilton_Hawaiian_Village_Waikiki_Beach_Resort-Honolulu_Oahu_Hawaii__en.csv',engine='python',index_col=False, nrows = 100)

data.head()
import spacy

from tqdm import tqdm

nlp = spacy.load('en_core_web_lg', parse=True, tag=True, entity=True)
data.head(3)
data.shape
txt = 'This is a huge resort, capacity-wise (not necessarily in terms of the territory. It sits next to a commercial "village" of sorts (the usual array of overpriced restaurants and various shops catering to tourist clientele).There"s nothing special to it, I"d even call it "faceless" despite a stream of celebrities and dignitaries of all kinds that graced the hotel with their presence and even movies exposure - but, at the same time, there"s nothing to complain about. The hotel is being well-run and well-maintained. The rooms are rather standard for resorts of this kind. The beach is right there (not the best, though - large-grain sand and quite a bit of crushed shells). In the room, we were pleasantly surprised to find a PS-3 gaming console that can run both games and DVDs you may rent from automated kiosks (free if you have a status of certain level with Hilton).The hotel has a large, well-equipped gym (I counted 33 cardio machines plus HIIT/pulley rack, all typical main-muscle groups strength machines, a full rack of dumbbells running through 100 lbs, balls, batons, etc. There"s a nice spa, too. They also offer group classes in various sports.Breakfasts were ok, although I"d expect a bit higher variety of morning foods. In any case, there are options for dining within a walking distance from the hotel.All in all, a positive experience. By Hawaii standards, an excellent hotel'

doc = nlp(txt)

spacy.displacy.render(doc,style='dep',jupyter=True)
txt = 'We stayed at HHV every summer in the 1980"s. This is our first trip back since then. The Village is as amazing and as beautiful as almost 40 years ago. It is self-contained-ABC stores, jewelers, clothing stores, luggage stores, eateries of all stripes, etc. With 4000 rooms, the check-in desk is bustling 24-7. The beaches (public) are pristine and the water beautiful. There is even a Church on the Beach which has operated there for 49 years and is open to all faiths. We actually found the food competitive in taste and price. The only complaint I have is the soundproofing between rooms. Our neighbor played loud music night and day, beginning at 7.20 a.m.(perhaps they are having a staycation in their room). After security visited, they would lower the volume, but we could still hear the music and their voices. As soon as security left, they started up again. We finally asked to be moved. I feel bad for whoever ended with the room next. The next room was quiet, because we had nice neighbors. But because of the location and convenience, and customer service, I still rate them a 5.'

doc = nlp(txt)

spacy.displacy.render(doc,style='dep',jupyter=True)
txt =  "I love hilton, but for this hotel, I really cannot recommend for anyone. We booked the most expansive room of the hotel, but except bigger room, the whole room was old and feel like in a 3 stars hotel room. Not worthy for the price at all. Won't recommend for this hotel."

doc = nlp(txt)

spacy.displacy.render(doc,style='dep',jupyter=True)
txt =  "Me and my father really enjoyed everything about this hotel. Would recommend it to anyone and hopefully we can soon return for another vacation.Beds were big and really comfy, also nice and big rooms."

doc = nlp(txt)

spacy.displacy.render(doc,style='dep',jupyter=True)
competitors = ['Chevy','chevy','Ford','ford','Nissan','nissan','Honda','honda','Chevrolet','chevrolet','Volkswagen','volkswagen','benz','Benz','Mercedes','mercedes','subaru','Subaru','VW']
data.head(2)
aspect_terms = []

comp_terms = []

easpect_terms = []

ecomp_terms = []

enemy = []



def get_aspect_adj(data):

    for x in tqdm(range(len(data['review_body']))):

        amod_pairs = []

        advmod_pairs = []

        compound_pairs = []

        xcomp_pairs = []

        neg_pairs = []

        eamod_pairs = []

        eadvmod_pairs = []

        ecompound_pairs = []

        eneg_pairs = []

        excomp_pairs = []

        enemlist = []

        if len(str(data['review_body'][x])) != 0:

            lines = str(data['review_body'][x]).replace('*',' ').replace('-',' ').replace('so ',' ').replace('be ',' ').replace('are ',' ').replace('just ',' ').replace('get ','').replace('were ',' ').replace('When ','').replace('when ','').replace('again ',' ').replace('where ','').replace('how ',' ').replace('has ',' ').replace('Here ',' ').replace('here ',' ').replace('now ',' ').replace('see ',' ').replace('why ',' ').split('.')       

            for line in lines:

                enem_list = []

                for eny in competitors:

                    enem = re.search(eny,line)

                    if enem is not None:

                        enem_list.append(enem.group())

                if len(enem_list)==0:

                    doc = nlp(line)

                    str1=''

                    str2=''

                    for token in doc:

                        if token.pos_ is 'NOUN':

                            for j in token.lefts:

                                if j.dep_ == 'compound':

                                    compound_pairs.append((j.text+' '+token.text,token.text))

                                if j.dep_ is 'amod' and j.pos_ is 'ADJ': #primary condition

                                    str1 = j.text+' '+token.text

                                    amod_pairs.append(j.text+' '+token.text)

                                    for k in j.lefts:

                                        if k.dep_ is 'advmod': #secondary condition to get adjective of adjectives

                                            str2 = k.text+' '+j.text+' '+token.text

                                            amod_pairs.append(k.text+' '+j.text+' '+token.text)

                                    mtch = re.search(re.escape(str1),re.escape(str2))

                                    if mtch is not None:

                                        amod_pairs.remove(str1)

                        if token.pos_ is 'VERB':

                            for j in token.lefts:

                                if j.dep_ is 'advmod' and j.pos_ is 'ADV':

                                    advmod_pairs.append(j.text+' '+token.text)

                                if j.dep_ is 'neg' and j.pos_ is 'ADV':

                                    neg_pairs.append(j.text+' '+token.text)

                            for j in token.rights:

                                if j.dep_ is 'advmod'and j.pos_ is 'ADV':

                                    advmod_pairs.append(token.text+' '+j.text)

                        if token.pos_ is 'ADJ':

                            for j,h in zip(token.rights,token.lefts):

                                if j.dep_ is 'xcomp' and h.dep_ is not 'neg':

                                    for k in j.lefts:

                                        if k.dep_ is 'aux':

                                            xcomp_pairs.append(token.text+' '+k.text+' '+j.text)

                                elif j.dep_ is 'xcomp' and h.dep_ is 'neg':

                                    if k.dep_ is 'aux':

                                            neg_pairs.append(h.text +' '+token.text+' '+k.text+' '+j.text)



                else:

                    enemlist.append(enem_list)

                    doc = nlp(line)

                    str1=''

                    str2=''

                    for token in doc:

                        if token.pos_ is 'NOUN':

                            for j in token.lefts:

                                if j.dep_ == 'compound':

                                    ecompound_pairs.append((j.text+' '+token.text,token.text))

                                if j.dep_ is 'amod' and j.pos_ is 'ADJ': #primary condition

                                    str1 = j.text+' '+token.text

                                    eamod_pairs.append(j.text+' '+token.text)

                                    for k in j.lefts:

                                        if k.dep_ is 'advmod': #secondary condition to get adjective of adjectives

                                            str2 = k.text+' '+j.text+' '+token.text

                                            eamod_pairs.append(k.text+' '+j.text+' '+token.text)

                                    mtch = re.search(re.escape(str1),re.escape(str2))

                                    if mtch is not None:

                                        eamod_pairs.remove(str1)

                        if token.pos_ is 'VERB':

                            for j in token.lefts:

                                if j.dep_ is 'advmod' and j.pos_ is 'ADV':

                                    eadvmod_pairs.append(j.text+' '+token.text)

                                if j.dep_ is 'neg' and j.pos_ is 'ADV':

                                    eneg_pairs.append(j.text+' '+token.text)

                            for j in token.rights:

                                if j.dep_ is 'advmod'and j.pos_ is 'ADV':

                                    eadvmod_pairs.append(token.text+' '+j.text)

                        if token.pos_ is 'ADJ':

                            for j in token.rights:

                                if j.dep_ is 'xcomp':

                                    for k in j.lefts:

                                        if k.dep_ is 'aux':

                                            excomp_pairs.append(token.text+' '+k.text+' '+j.text)

            pairs = list(set(amod_pairs+advmod_pairs+neg_pairs+xcomp_pairs))

            epairs = list(set(eamod_pairs+eadvmod_pairs+eneg_pairs+excomp_pairs))

            for i in range(len(pairs)):

                if len(compound_pairs)!=0:

                    for comp in compound_pairs:

                        mtch = re.search(re.escape(comp[1]),re.escape(pairs[i]))

                        if mtch is not None:

                            pairs[i] = pairs[i].replace(mtch.group(),comp[0])

            for i in range(len(epairs)):

                if len(ecompound_pairs)!=0:

                    for comp in ecompound_pairs:

                        mtch = re.search(re.escape(comp[1]),re.escape(epairs[i]))

                        if mtch is not None:

                            epairs[i] = epairs[i].replace(mtch.group(),comp[0])



        aspect_terms.append(pairs)

        comp_terms.append(compound_pairs)

        easpect_terms.append(epairs)

        ecomp_terms.append(ecompound_pairs)

        enemy.append(enemlist)

    data['compound_nouns'] = comp_terms

    data['aspect_keywords'] = aspect_terms

    data['competition'] = enemy

    data['competition_comp_nouns'] = ecomp_terms

    data['competition_aspects'] = easpect_terms

    return data
data.shape
data1 = get_aspect_adj(data)
!pip install vaderSentiment
my_df = pd.DataFrame(columns = ['row_index','aspect_adj_string','pos_score','neg_score','compound_score','neutral_score'])

my_df
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

my_df = pd.DataFrame(columns = ['row_index','aspect_adj_string','pos_score','neg_score','compound_score','neutral_score'])

global_sentiment = {}

for aspect_adj_tuple in data1.aspect_keywords.iteritems():

    row_sentiment = {}

    for text in aspect_adj_tuple[1]:

        sentiment_array = []

        polarity = analyser.polarity_scores(text)

#         print(polarity)

#         sentiment_array.append(polarity['pos'])

#         sentiment_array.append(polarity['neg'])

#         row_sentiment[text] = sentiment_array

        my_df = my_df.append({'row_index':aspect_adj_tuple[0],'aspect_adj_string':text,'pos_score':polarity['pos'],'neg_score':polarity['neg'],'compound_score':polarity['compound'],'neutral_score':polarity['neu']},ignore_index=True)

#     global_sentiment[aspect_adj_tuple[0]] = row_sentiment

    

# print(global_sentiment)

my_df[(my_df['compound_score']<0) & (my_df['pos_score']>0) & (my_df['neg_score']>0)]
my_df.shape
final_df = my_df[((my_df['compound_score']>0.1) | (my_df['compound_score']< -0.1))]

final_df[(final_df['compound_score']<0) & (final_df['pos_score']> final_df['neg_score'])]
final_df['final_sentiment'] = ['pos' if i>0 else 'neg' for i in final_df['compound_score']]
final_df.head(30)
# import spacy



# nlp = spacy.load("en_core_web_sm")

# # adj =[]

# # aspect = []

# temp['adj']='nan'

# temp['aspect']='nan'



# for row_txt in final_df.aspect_adj_string.iteritems():

#     print(row_txt)

#     doc = nlp(row_txt[1])

#     adj =[]

#     aspect = []

#     print(doc)

#     for token in doc:

#         if (token.tag_ == 'ADV') or (token.tag_ == 'ADJ') or (token.tag_ == 'JJ'):

#             print(token.text)

#             adj.append(token.text)



        

#         elif (token.tag_ == 'NN') or (token.tag_ == 'NNP') or (token.tag_ == 'VBG') or (token.tag_ == 'VB'):

#             print(token.text)

#             aspect.append(token.text)

    

#     temp=temp.append({'aspect':aspect,'adj':adj},ignore_index=True)

            

doc = 'well done'

doc = nlp(doc)

for token in doc:

    print(token.tag_)

    print(token.text)
import spacy



nlp = spacy.load("en_core_web_sm")



for row_txt in final_df.aspect_adj_string.iteritems():

    doc = nlp(row_txt[1])

    adj = []

    aspect = []

    for token in doc:



        if (token.tag_ == 'ADV') or (token.tag_ == 'ADJ') or (token.tag_ == 'UH') or (token.tag_ == 'JJS') or (token.tag_ == 'RB') or (token.tag_ == 'JJ') :

            adj.append(token.text)

            

        elif (token.tag_ == 'NN') or (token.tag_ == 'NNP') or (token.tag_ == 'VBG') or (token.tag_ == 'VB')  or (token.tag_ == 'VBN') or (token.tag_ == 'NNS'):

            aspect.append(token.text)



    if (len(adj) == len(aspect)) & (len(adj)>0):

        final_df.at[row_txt[0],'adj']= adj

        final_df.at[row_txt[0],'aspect']= aspect

    elif len(adj) > len(aspect):

        final_df.at[row_txt[0],'adj']= adj[:len(aspect)]

        final_df.at[row_txt[0],'aspect']= aspect

    elif  len(adj) < len(aspect):

        final_df.at[row_txt[0],'adj']= adj

        final_df.at[row_txt[0],'aspect']= aspect[:len(adj)]

    else:

        final_df.at[row_txt[0],'adj']= adj

        final_df.at[row_txt[0],'aspect']= aspect

            
final_df
import csv

word_vec = pd.read_csv('/kaggle/input/glove6b50dtxt/glove.6B.50d.txt',sep=' ',index_col=0,header=None,quoting=csv.QUOTE_NONE)
word_vec.head()
final_df.reset_index(inplace=True,drop=True)
final_df.head()
import numpy as np

aspect_vec = np.zeros((final_df.shape[0],50))

for aspect in final_df.aspect.iteritems():

    if len(aspect[1]):

        embedding_vector = word_vec[word_vec.index == aspect[1][0]].iloc[:,:].values

        if len(embedding_vector):

            aspect_vec[aspect[0]] = embedding_vector

            
# pd.DataFrame(aspect_vec)
from sklearn.cluster import KMeans

kmeans= KMeans(n_clusters=60, random_state=0)

kmeans.fit(aspect_vec)
# print(kmeans.cluster_centers_)
print(kmeans.labels_)
final_df['aspect_cluster_labels'] = kmeans.labels_
final_df.sort_values(by=['aspect_cluster_labels'],ascending=True)[:30]
 # clustering dataset

# determine k using elbow method



from sklearn import metrics

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt



# create new plot and data

plt.plot()

X = aspect_vec

colors = ['b', 'g', 'r']

markers = ['o', 'v', 's']



# k means determine k

distortions = []

K = range(1,100)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(X)

    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000

pd.options.display.max_colwidth = 1000
data[['review_body', 'aspect_keywords']].head(10)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
import operator

sentiment = []

for i in range(len(data)):

    score_dict={'pos':0,'neg':0,'neu':0}

    if len(data['aspect_keywords'][i])!=0: 

        for aspects in data['aspect_keywords'][i]:

            sent = analyser.polarity_scores(aspects)

            score_dict['neg'] += sent['neg']

            score_dict['pos'] += sent['pos']

        #score_dict['neu'] += sent['neu']

        sentiment.append(max(score_dict.items(), key=operator.itemgetter(1))[0])

    else:

        sentiment.append('NaN')

data['sentiment'] = sentiment

data.head()
int_sent = []

for sent in data['sentiment']:

    if sent is 'NaN':

        int_sent.append('NaN')

    elif sent is 'pos':

        int_sent.append('1')

    else:

        int_sent.append('0')

data['int_sent'] = int_sent

data.head()
d = {'sent':toy_rev['Positive Review'],'sent_pred':toy_rev['int_sent']}

metric_df = pd.DataFrame(data=d)

metric_df.head()
len(metric_df.sent)
metric_df = metric_df[metric_df.sent_pred != 'NaN']

len(metric_df.sent)
from sklearn.metrics import accuracy_score,auc,f1_score,recall_score,precision_score

print('accuracy')

print(accuracy_score(metric_df.sent, metric_df.sent_pred))

print('f1 score')

print(f1_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))

print('recall')

print(recall_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))

print('precision')

print(precision_score(metric_df.sent, metric_df.sent_pred,pos_label='1'))