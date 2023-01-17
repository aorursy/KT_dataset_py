import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS

from operator import itemgetter

import matplotlib as mpl

%matplotlib inline

from os import path

import matplotlib.pyplot as plt

import os

print(os.getcwd())

print ("file exist:"+str(path.exists("../input/")))









yelp_business=pd.read_csv("../input/yelp.csv")

plt.rcParams['figure.figsize']=(12,5)     

Categories={}



for x in yelp_business.categories:

    all_categories=x.split(";")

    for cat in all_categories:

        if cat not in Categories:

            Categories[cat]=1

        else:

            Categories[cat]+=1

All_categories=list(Categories.keys())

Cat_list=[[x,Categories[x]] for x in All_categories]



Cat_list=sorted(Cat_list, key=lambda x: x[1], reverse=True)

#LETS find the top 40 Categories of business

Cat_list=Cat_list[:40]

plt.bar(range(len(Cat_list)),[x[1] for x in Cat_list] ,align="center", color="bkmcgr")

plt.xticks(range(len(Cat_list)), [x[0] for x in Cat_list], rotation="vertical")

plt.show()
Only_stars=[]

Categories_star={}

for i,x in yelp_business.iterrows():

    all_categories=x["categories"].split(";")

    Only_stars.append(int(round(x["stars"])))

    for cat in all_categories:

        if cat not in Categories_star:

            Categories_star[cat]=[]

        Categories_star[cat].append(x["stars"])

Star_list=[]

for x in list(Categories_star.keys()):

    Star_list.append([x, np.mean(Categories_star[x])])

    

Star_list=sorted(Star_list, key=lambda x: x[1], reverse=True)

Star_list=Star_list[:20] + Star_list[len(Star_list)-20:]



plt.bar(range(len(Cat_list)),[x[1] for x in Star_list] ,align="center",color="rgbkmc")

plt.xticks(range(len(Cat_list)), [x[0] for x in Star_list], rotation="vertical")

plt.show()

Only_stars=pd.DataFrame(Only_stars)

Only_stars.columns=["STARS"]
Only_stars["STARS"].groupby(Only_stars["STARS"]).count().plot(kind="bar", sort_columns=True,color=[plt.cm.Paired(np.arange(len(Only_stars)))])
yelp_business.city.groupby(yelp_business.city).count().sort_values()[::-1][:35].plot(kind="bar",color=[plt.cm.Paired(np.arange(len(yelp_business)))])
mpl.rcParams['font.size']=10

mpl.rcParams['figure.subplot.bottom']=.1 

word_string=" ".join(yelp_business["name"]).replace('"','').lower()

wordcloud = WordCloud(

                          background_color='black',

                          stopwords=set(STOPWORDS),

                          max_words=2500,

                          max_font_size=400, 

                          random_state=42

                         ).generate(word_string)

plt.imshow(wordcloud)

plt.axis('off')



plt.show()
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import time

stop_words = set(stopwords.words('english'))

CountDictionary={}

for i in range(1,6):

    CountDictionary[i]={}

CountDictionary["all"]={}

def process(yelp_review):

    for i,x in yelp_review.iterrows():

        text=x["text"]

        tokenizer = RegexpTokenizer(r'\w+')

        text = tokenizer.tokenize(text.lower())

        text=[x for x in text if x not in stop_words]

        star=x["stars"]

        for val in text:

            if val in CountDictionary[star]:

                CountDictionary[star][val]+=1

            else:

                CountDictionary[star][val]=1

            if val in CountDictionary["all"]:

                CountDictionary["all"][val]+=1

            else:

                CountDictionary["all"][val]=1



chunksize = 10000

filename="../input/yelp_review.csv"

count=1

beg_ts = time.time()

avg_time_chunk=[]

for chunk in pd.read_csv(filename, chunksize=chunksize):

    ch_start=time.time()

    process(chunk)

    ch_end=time.time()

    avg_time_chunk.append(ch_end-ch_start)

    count+=1

end_ts=time.time()

print ("Total time taken to read 3.53 GB file is " + str(end_ts - beg_ts))

print ("Average time for processing one chunk of"+ str(count) + "chunks is "+ str( np.mean(avg_time_chunk)))

print ("Sucessfully Populated Dictionary")
def CreateCloud(attr):

    

    wordcloud = WordCloud(background_color='black',

                              stopwords=set(STOPWORDS),

                              random_state=42).generate_from_frequencies(frequencies=CountDictionary[attr])

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
CreateCloud("all")
for i in range(1,6):

    print (("_")*90)

    print ("WORD CLOUD FOR "+ str(i)+" stars")

    CreateCloud(i)
yelp_checkin=pd.read_csv("../input/yelp_checkin.csv")

yelp_checkin.weekday.groupby(yelp_checkin.weekday).count().sort_values()[::-1].plot(kind="bar",color=[plt.cm.Paired(np.arange(len(yelp_checkin)))])
yelp_checkin=yelp_checkin.sort_values(by="checkins", ascending=False)[:15]

s1 = pd.merge(yelp_business, yelp_checkin, how='inner', on=['business_id'])

s1=s1.sort_values(by="checkins", ascending=False)

plt.style.use('ggplot')

ax = s1[['checkins']].plot(kind='bar',figsize=(15,10),legend=True, fontsize=12, color=[plt.cm.Paired(np.arange(len(yelp_checkin)))])

ax.set_xticklabels(s1.name, rotation=90)

ax.set_xlabel("Business Names",fontsize=12)

ax.set_ylabel("Checkins",fontsize=12)

plt.show()