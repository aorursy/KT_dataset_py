'''

lyrics_table = pd.DataFrame({'album':[], 'name':[], 'lyrics':[]})



counter=-1



def add_lyrics(album, songname, lyrics):

    global counter

    counter+=1

    name = songname

    lyrics = lyrics

    album = album

    lyrics_table.loc[counter]=[album, name, lyrics]

    

song=''''''



add_lyrics('Relentless Mutation', 'A Dark Horizontal', song)



lyrics_table.rename(columns={'name':'songname'})

lyrics_table.name[4] = 'Ancient Of Ancients'

'''
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

import nltk

from nltk.corpus import stopwords

from nltk import regexp_tokenize

from wordcloud import WordCloud

from PIL import Image

import random





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ArchspireLyrics = pd.read_csv('/kaggle/input/archspire-lyrics/ArchspireLyrics.csv')

ArchspireLyrics
ArchspireLyrics = ArchspireLyrics.drop('Unnamed: 0', axis=1)

ArchspireLyrics.lyrics[10]='''

I awake on the shore.

The entity that greets me upon this plane

Appears as many changing forms.

Flushing constant waves of shape,

None of which I have ever seen.



"To unite, we breed with all species indiscriminately,

live or dead, from any universe.

Righteous, the torment that you will endure

Just to comprehend what we are.

Over come the self, Leave your fear behind.

Open up your mind.

Let us in."



I'm asked by the being if I'm dreaming or I'm dead?



When the portal opened,Changing everything,

Catastrophic multitudes of dimensional mending

Offset the natural order.

Spreading its disease over every world

In the wake of its infection.

Causing counter planetary lifeforms to contact each other,

By engaging mental methods of unifying dream response.



Thus the alliance for timeless existence formed.

A galactic coalition of intelligence.

Unanimously ravelling their consciousness to become one.



"Ever growing, we are many, join us in our hive.

Dream with us and we will teach you

How to visualize the surroundings of your new world.

Once you break through the fabrications of fear

You can become any construct that you desire.

If your dream be that of fire

Transform the fire to earth, wind or water.

Sculpt the elements around you with intricacy.

Comprehend the nature of their pliable dimensions.



You are the conductor.



The path is clear to cut away the fabric of time, linear,That which you fear

And join us beyond.



Always question if you are dreaming.

Envision every inch of your newly born surroundings

As pieces of your self.

For you have constructed every molecule

Of this unstable plane of the subconscious,

Comprising your lucid dream.



Remove the body through the mind.

Find your hands while you are dreaming,

Carry that self to the distant place where your old dying self lay.

Sink into this decomposing human slab

And ride upon its life light as it fades in flashes."



My essence now mates with the thousands of others,

No longer apart from the creature that brought me here.

The voice of this entity has been ingrained into me,

Guiding my path into hyper reality.

Breaking down patterns of pain that emerge

In the very last seconds before my new reckoning.

Thriving in a formless state of pure expression.

Fearlessly I join the lucid collective beyond.



A web of new life awaits exempt of the body.

Leave yourself behind and become one with the hive.

Join us beyond the gateways of the dreaming dead.

'''
num_words = []

for i in ArchspireLyrics['lyrics']:

    num_words.append(len(i.split(' ')))

ArchspireLyrics['num_words']=num_words

ArchspireLyrics
sw = stopwords.words('english')

patn = '\w+'



def clean_text(text):

    text = text.lower()

    text = re.sub(r'([.!,?])', r' \1 ', text)

    text = re.sub(r'[^a-zA-Z.,!?]+', r' ', text)

    return text



Cleaned_lyrics = ArchspireLyrics['lyrics']

Cleaned_lyrics = Cleaned_lyrics.apply(lambda x: clean_text(x))



def tokenize_and_remove_sw(text):

    text = regexp_tokenize(text, patn)

    text = [i for i in text if i not in sw]

    return text

Tokenized_lyrics = Cleaned_lyrics.apply(lambda x: tokenize_and_remove_sw(x))

Tokenized_lyrics[0]
for i in range(len(Tokenized_lyrics)):

    print('Tokenized and without stopwords: {}, Original: {}'.format(len(Tokenized_lyrics[i]), ArchspireLyrics.num_words[i]))

num_tokenized = []

for i in Tokenized_lyrics:

    num_tokenized.append(len(i))

ArchspireLyrics['num_tokenized']=num_tokenized
ArchspireLyrics['cleaned_lyrics']=Cleaned_lyrics

ArchspireLyrics['tokenized_lyrics']=Tokenized_lyrics

ArchspireLyrics
values = ArchspireLyrics['num_words']

labels = ArchspireLyrics['name']



from matplotlib import cm



plt.figure(figsize=(10, 10))



viridis = cm.get_cmap('tab20', 20)

plt.pie(values, labels=labels, autopct='%1.1f%%', colors=viridis.colors)

plt.title("Size comparison of Archspire's songs", size=25)



plt.show()
values = ArchspireLyrics['num_tokenized']

labels = ArchspireLyrics['name']



from matplotlib import cm



plt.figure(figsize=(10, 10))



viridis = cm.get_cmap('tab20', 20)

plt.pie(values, labels=labels, autopct='%1.1f%%', colors=viridis.colors)

plt.title('Size comparison of songs without stopwords', size=25)



plt.show()
vocab = []

for i in Tokenized_lyrics:

    for j in i:

        vocab.append(j)



            

unique_vocab = set(vocab)
longer_words = [i for i in vocab if len(i)>2]

freq_dist = nltk.FreqDist(longer_words)
plt.figure(figsize=(16, 10))

freq_dist.plot(50)
def grey_color_func(word, font_size, position, orientation, random_state=None,

                    **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(0, 60)







mask = np.array(Image.open('/kaggle/input/archspire-lyrics/Archi.jpg'))

WordC = WordCloud(background_color='white',

                  mask=mask,

                 max_font_size=90,

                  max_words=2000,

                  random_state=42)

wcloud = WordC.generate_from_frequencies(freq_dist)

default_colors = wcloud.to_array()

plt.figure(figsize=(30, 20))

plt.imshow(wcloud.recolor(color_func=grey_color_func, random_state=42), interpolation='bilinear')

plt.axis("off")

plt.savefig('ArchspireCloud.png')

plt.show()
counter = 0

for word in unique_vocab:

    for i in Tokenized_lyrics:

        if word in i:

            counter+=1

    if counter>=10:

        print("Word '{}' was found in {} out of {} songs".format(word, counter, len(Tokenized_lyrics)-1))

        counter=0

    else:

        counter=0