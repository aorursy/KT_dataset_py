# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.corpus import twitter_samples as ts

from nltk.tokenize import word_tokenize as wtoken

from nltk.tokenize import wordpunct_tokenize as wptk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
samples_tw = ts.strings('tweets.20150430-223406.json') #loading sample tweets
wtoken(samples_tw[20]) #word tokenizer
wptk(samples_tw[20]) #word_punct tokenizer
from nltk import regexp_tokenize

patn= '\w+'

regexp_tokenize(samples_tw[20],patn) #regexp tokenizer
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

for i in wtoken(samples_tw[20]):

    lemmed = lem.lemmatize(i) 

    print('{}: {}'.format(i, lemmed))
from nltk.stem import PorterStemmer

stemming = PorterStemmer()

for i in wtoken(samples_tw[20]):

    stemmed = stemming.stem(i) 

    print('{}: {}'.format(i, stemmed))
from nltk.corpus import stopwords

sw = stopwords.words('english')

sentence = "The sentence you're reading is a sentence that contains stop words"

def nostopwords(sentence):

    arr = sentence.split(' ')

    sentence_wo_stopwords=''

    for i in arr:

        if i not in sw:

            sentence_wo_stopwords+=(str(i)+' ')

    print(sentence_wo_stopwords)

print(sentence)

nostopwords(sentence)
lyrics = '''The dark horizontal

Ahead, a panting land approaches

Flaring vibrant coastal glimmers

Mend the distant fraying seams

Of a violent waters restless bedding

Panic indented in the floating web of wicked mental horrors

Thought up by the crew's subconscious fears

Come alive to haunt and drain us



The sea is a man that's trying to kill me

The wind is his repugnant voice

The air his fouled breath

A stench of salt of blood and rot

The sky is his mouth, the stars his open sores

The shoreline is his jaw, grinding for meat



Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear

Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear



Mountains unstable and hideous

The guardians of a contorting dimensional paragon

A.U.M

A.U.M. sent us to find and retrieve the artifact

In the house of change

Where the king came back

Through spontaneous regeneration

Reliving Kairos in the king's hell

Becoming the mimics from the triplets' well

Exhuming infinity to find a hand

A child made out of teeth, feeding on blood and sand



The sea is a man that's trying to kill me

The air his fouled breath

A stench of salt, of blood and rot

The sky, his mouth, the stars open sores

The shoreline is his jaw, grinding for meat

The sea is a man that's trying to kill me

The sky is his mouth, the stars open sores

The shoreline is his jaw, grinding for...



Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear

Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear

A dark horizontal

Flooding the linear land with the sand of its own

In the parallel dunes of its hourglass

We are buried

Mountains unstable and hideous

The guardians of a contorting dimensional paragon



Raised blades of land stab skyward

Through the ocean's skin

Storm winds true their edge to scalp

The curve of Earth

Peeling off the rind, revealing a house made of living tissue

Yelling out its mouth for me to regenerate, in it



In dark horizontal

Ahead, a panting land approaches

Flaring vibrant coastal glimmers

Mend the distant fraying seams

Of a violent waters restless bedding

In this house the disease of dream

Course in the veins and arteries of the walls

Imagined infections thrive

They blister and breed under the floor

Their many roots are breaking through

Saplings sprout and secure their place

Growing into trunks of flesh and boil

The pillars of man's deepest fears



Enclosing me in a forest of unforgiving mental genesis

I mold this house of clay bacterium

Where I sculpt my own organ prison

In this house beyond our time, this house at the end of the Earth

The artifact's resting place



A dark horizontal fear

Defying the linear



Lost in this storm and exhausted of options

I crawled to the bow of the boat and leaped off it

The figurehead came alive

And it clawed and it grabbed at me

Desperately wailing in agony

Failing to capture my fall

Diving through the fog

Through the gray and green

My hands bound together as if in prayer

I cut through the mesh of black liquid surrounding us

Fighting against the live current of thousands of

Thousands of eyes and hands

Miles I swam

Through skin and limb



The sea is a man that's trying to kill me

The air his fouled breath

A stench of salt of blood and rot

The sky his mouth, the stars his open sores

The shoreline is his jaw grinding for meat



The sea is a man that's trying to kill me

The sky is his mouth, the stars open sores

The shoreline is his jaw, grinding for...



Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear

Time freezes death inside this altering dark horizontal

Imaginable realities constructed out of fear

'''
lyrics_words = wtoken(lyrics.lower())

lyrics_words[:20]
frequencies = nltk.FreqDist(lyrics_words)

longer_than_2 = dict([(i, j) for i, j in frequencies.items() if len(i)>3])

freq_dist=nltk.FreqDist(longer_than_2)

freq_dist.plot(50, cumulative=False)
from wordcloud import WordCloud

import matplotlib.pyplot as plt



wcloud = WordCloud().generate_from_frequencies(freq_dist)

plt.figure(figsize=(10, 10))

plt.imshow(wcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
no_stop_lyrics = [i for i in lyrics_words if (i not in sw and len(i)>2)]

freq_dist_2 = nltk.FreqDist(no_stop_lyrics)

wcloud_2 = WordCloud().generate_from_frequencies(freq_dist_2)

plt.figure(figsize=(12, 12))

plt.imshow(wcloud_2, interpolation='bilinear')

plt.axis('off')

plt.show()
nltk.pos_tag(lyrics_words[:30], tagset='universal')