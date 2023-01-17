import pandas as pd

import numpy as np

import nltk
nltk.download("popular")
import re

import nltk

from nltk.util import ngrams

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

import random
ps=PorterStemmer()

wordnet=WordNetLemmatizer()

corpus=[]
df = pd.read_csv('../input/sample.csv')

df.info()
df.head()
#===data cleaning======

df['FeedText'].isnull().sum(axis = 0)

df.dropna(subset=['FeedText'],inplace=True)

df.reset_index(inplace = True, drop = True) 

df.info()
#Working on words

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import wordnet as wn

import nltk

def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wn.ADJ,

                "N": wn.NOUN,

                "V": wn.VERB,

                "R": wn.ADV}



    return tag_dict.get(tag, wn.NOUN)
total_words=[]

for i in range(len(df['Message'])):

    print(i)

    text_for=df['Message'][i]

    review = re.sub('[^a-zA-Z]',' ',str(text_for))

    review = review.lower()

    #review = review.split()

    all_word=[wordnet.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(review)]

    #print(all_word)

    #review=" ".join(all_word)

    review = [wordnet.lemmatize(word,'v') for word in all_word if not word in stopwords.words('english')]

    #print(review)

    total_words.append(review)

    review1 = ' '.join(review)

    corpus.append(review1)

print(len(corpus))

print(type(corpus))

print(len(total_words))
#=========================================working on word2vec==========================================

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer

from sklearn import cluster

from sklearn import metrics
model = Word2Vec(total_words,min_count=1,size=32)



X=model[model.wv.vocab]
seller=['Seller','sell',"offer","pay","Quick","shopgirl","saleswoman","salesman",

        "salesperson","shopkeeper","discount","supplier","off","interested",

        "price","percentage","percentage-off","retail","stock",

        "never-again","last-chance",

        "running-out","horse-trader",

        "merchant","merchandise","market","handle","bring-around","convince",

        "prevail-on","salableness","salability","marketableness","marketability",

        "sell-for","sell-out","sell-off","double-cross"

        

        ]
#====================================find all seller synonyms=======================================

import nltk 

from nltk.corpus import wordnet 

seller_synonyms = [] 

for i in seller:

    for syn in wordnet.synsets(i): 

        for l in syn.lemmas(): 

            seller_synonyms.append(l.name()) 

seller_list=list(set(seller_synonyms))           

print(set(seller_synonyms))
#==========================================find top2 words most similar of seller===========================================

seller_w2v_list=[]

for i in seller_list:

    try:

        hk=model.most_similar(positive=[i], negative=[],topn=2)

        print(hk)

        for j in range(len(hk)):

            seller_w2v_list.append(hk[j][0])

    except:

        pass

    

print(seller_w2v_list)

print(len(seller_w2v_list))
#================================================buyer list======================================================

buyer=["buy","buyer","customer","consumer","investment","acquisition","client","go-shopping","purchaser","patron","customer","client","bull","user","patron","representive","prospect","let-the-buyer-beware"

       "buyers-premium","pig-in-a-poke","shopper","purchasing-agent","purchase","accept","order","redeem","hire"]



print(len(buyer))

print(len(seller))
#==============================================find all buyer synonyms ========================================== 

import nltk 

from nltk.corpus import wordnet 

buyer_synonyms = [] 



for i in buyer:

    for syn in wordnet.synsets(i): 

        for l in syn.lemmas(): 

            buyer_synonyms.append(l.name()) 

buyer_list=list(set(buyer_synonyms))

print(buyer_list)

print(len(set(buyer_synonyms)))
#==========================================find top2 words most similar buyer===========================================

buyer_w2v_list=[]

for i in buyer_list:

    try:

        mk=model.most_similar(positive=[i], negative=[],topn=2)

        print(mk)

        for j in range(len(mk)):

            buyer_w2v_list.append(mk[j][0])

    except:

        pass

    

print(buyer_w2v_list)

print(len(buyer_w2v_list))
#============================================extend both list===========================================================

seller_w2v_list.extend(seller)

buyer_w2v_list.extend(buyer)

seller_w2v_list=set(seller_w2v_list)

buyer_w2v_list=set(buyer_w2v_list)



seller_w2v_list=list(seller_w2v_list)

buyer_w2v_list=list(buyer_w2v_list)



seller_w2v_list=list(seller_w2v_list[::-1])

buyer_w2v_list=list(buyer_w2v_list[::-1])

print(len(seller_w2v_list))



print(len(seller_w2v_list))

print(len(buyer_w2v_list))
#==========================================check freq of words in both clusters========================================

seller_word =list(set(seller_w2v_list.copy()))

print(len(seller_word))

buyer_word = list(set(buyer_w2v_list.copy()))

seller_word.extend(buyer_word)

print(len(seller_word))

count_df={}

for i in seller_word:

    if i in count_df:

        count_df[i] +=1

    else:

        count_df[i] =1
#=================================================remove elements from seller==========================================

'''

seller_w2v_list.remove("buy")

seller_w2v_list.remove("go")

seller_w2v_list.remove("customer")

seller_w2v_list.remove("purchase")

seller_w2v_list.remove("choose")

seller_w2v_list.remove("get")'''
#=================================================remove elements from buyer==========================================

'''

buyer_w2v_list.remove("sell")

buyer_w2v_list.remove("rate")

buyer_w2v_list.remove("charge")

buyer_w2v_list.remove("cost")

buyer_w2v_list.remove("product")

buyer_w2v_list.remove("price")

buyer_w2v_list.append("looking")

buyer_w2v_list.append("looking")

buyer_w2v_list.append("look")

buyer_w2v_list.append("new")'''
#==============================================remove elements from both clusters====================================

'''seller_w2v_list.remove("keyword")

buyer_w2v_list.remove("keyword")

seller_w2v_list.remove("mother")

buyer_w2v_list.remove("mother")

seller_w2v_list.remove("suicide")

buyer_w2v_list.remove("suicide")

seller_w2v_list.remove("without")

buyer_w2v_list.remove("without")'''
#===========================================shuffle elements of seller and buyer list===============================

random.shuffle(seller_w2v_list) 

random.shuffle(buyer_w2v_list)
#=========================================create wordscloud for seller=============================================

from wordcloud import WordCloud

import matplotlib.pyplot as plt

seller_text = list(seller_w2v_list)

seller_unique_string=(" ").join(seller_w2v_list)

seller_wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(seller_unique_string)

plt.figure(figsize=(15,8))

plt.imshow(seller_wordcloud)

plt.axis("off")

#plt.savefig("seller_review"+".png")

plt.show()

plt.close()
#=======================================create wordsclouds for buyer===============================================

'''from wordcloud import WordCloud

import matplotlib.pyplot as plt

buyer_text = list(buyer_w2v_list)

convert list to string and generate

buyer_unique_string=(" ").join(buyer_w2v_list)

buyer_wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(buyer_unique_string)

plt.figure(figsize=(15,8))

plt.imshow(buyer_wordcloud)

plt.axis("off")

#plt.savefig("buyer_review"+".png",bbox_inches='tight')

plt.show()

plt.close()'''
#================================================predict labels===================================================

seller_flag=1

buyer_flag=1

label=[]

def pre(text):

    check_str=[]

    global seller_flag

    global buyer_flag

    wordnet=WordNetLemmatizer()

    check = re.sub('[^a-zA-Z]',' ',str(text))

    check = check.lower()

    check_all_word=[wordnet.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(check)]

    check_word = [wordnet.lemmatize(word) for word in check_all_word if not word in stopwords.words('english')]

    check_str.append(check_word)

    output = list(ngrams(check_word, 2))

    for i in range(len(output)):

        new_word=output[i][0]+"-"+output[i][1]

        check_str[0].append(new_word)



    for i in range(len(check_str)):

        seller_flag=0

        buyer_flag=0

        for j in range(len(check_str[i])-1,-1,-1):

            if check_str[i][j] in seller_w2v_list:

                label.append("seller")

                #global seller_flag

                seller_flag=2

                #print("Seller",check_str[i][j])

                break

            elif check_str[i][j] in buyer_w2v_list:

                label.append("Buyer")

                #global buyer_flag

                buyer_flag=2

                #print("buyer",check_str[i][j])

                break

        else:

            label.append("Neutral")

            pass

            

    if seller_flag==2 and buyer_flag==2:

        print("Both")

    elif seller_flag ==2:

        print("Seller")

    elif buyer_flag ==2:

        print("Buyer")

    else:

        print("Neutral")
#========================check for every input string==============================================

while(True):

    user_input=input("Enter your string : ")

    if user_input !='axaxa':

        pre(user_input)

    else:

        break
#=======================check for dataframe slice=================================================

for i in range(len(df['Message'][0:101])):

    text_for=df['Message'][i]

    pre(text_for)
#========================save label and message as a dataframe csv file================================

pre_text2=pd.DataFrame(label,columns=['label'])

my_df2 = pd.DataFrame(label)

my_df2[0].value_counts()





new_pre_df3 = pd.DataFrame({'Message':df['Message'][0:101],'label':pre_text2['label'][0:101]})

new_pre_df3.to_csv('pre.csv',index=False)

#=====================================================end============================================