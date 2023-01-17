#importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import string

from wordcloud import WordCloud, STOPWORDS

from math import pi



#importing the lexixcon



NRC = pd.read_csv('../input/bing-nrc-afinn-lexicons/NRC.csv')

print('Number of words compiled in the NRC Lexicon: ', len(NRC))

NRC.head()
#ingesting the three datasets



dunwich_raw = pd.read_csv('../input/3-horror-stories/dunwich.txt', names=['word'])

mrhyde_raw = pd.read_csv('../input/3-horror-stories/mrhyde.txt', names=['word'])

metamor_raw = pd.read_csv('../input/3-horror-stories/metamor.txt', names=['word'])
#quick check on our raw data - basically 3 one-column dataframes with one word per row.



print('How many words for "The Dunwich Horror"? :', len(dunwich_raw))

dunwich_raw.head()
print('How many words for "Strange Case of Dr Jekyll and Mr Hyde"? :', len(mrhyde_raw))

mrhyde_raw.head()
print('How many words for "The Metamorphosis"? :', len(metamor_raw))

metamor_raw.head()
#removing punctuation



dunwich_raw['word'] = dunwich_raw['word'].apply(lambda x:''.join([i for i in x 

                                                  if i not in string.punctuation]))

mrhyde_raw['word'] = mrhyde_raw['word'].apply(lambda x:''.join([i for i in x 

                                                  if i not in string.punctuation]))

metamor_raw['word'] = metamor_raw['word'].apply(lambda x:''.join([i for i in x 

                                                  if i not in string.punctuation]))



#removing common english words, irrelevant for our study (e.g. pronouns etc.)



stop = stopwords.words('english')

dunwich_raw['word'] = dunwich_raw['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

mrhyde_raw['word'] = mrhyde_raw['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

metamor_raw['word'] = metamor_raw['word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



#checking if there is any Null value



print(dunwich_raw.isnull().any())

print(mrhyde_raw.isnull().any())

print(metamor_raw.isnull().any())
#merging the DataFrames with the lexicon



dunwich = dunwich_raw.merge(NRC, on='word', how='inner')

mrhyde = mrhyde_raw.merge(NRC, on='word', how='inner')

metamor = metamor_raw.merge(NRC, on='word', how='inner')
print('The DataFrame contains', len(dunwich), 'rows.')

dunwich.head(10)
#creating a new df with no duplicate.



dunwich_word = dunwich['word'].value_counts()

dunwich_word = pd.DataFrame(dunwich_word)

print(dunwich_word.head(10))



#counting the words left and lost during the filtering



print('How many words left for "The Dunwich Horror"? : ', len(dunwich_word))

print('"The Dunwich Horror" has lost', len(dunwich_raw)-len(dunwich_word), 'words during the process.')
#creating the wordcloud

    

wordcloud = WordCloud(

    width = 800,

    height = 600,

    background_color = 'white',

    stopwords = list(STOPWORDS) + ['word']).generate(str(dunwich_word.head(50)))

plt.figure(figsize = (8, 8))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#counting and classifying the emotions



dunwich_emotions = dunwich['sentiment'].value_counts()

dunwich_emotions
#creating the bar plot



dunwich_emotions.sort_values().plot(figsize=(12,6),kind='barh', color=('mediumseagreen','mediumaquamarine'))

plt.title('"The Dunwich Horror" Word Analysis', fontsize=20);
print('Number of rows in the DataFrame:', len(mrhyde))

mrhyde.head(10)
#creating a new df with no duplicate.



mrhyde_word = mrhyde['word'].value_counts()

mrhyde_word = pd.DataFrame(mrhyde_word)

print(mrhyde_word.head(10))



#counting the words left and lost during the filtering



print('How many words left for "Strange Case of Dr Jekyll and Mr Hyde"? : ', len(mrhyde_word))

print('"Strange Case of Dr Jekyll and Mr Hyde" has lost', len(mrhyde_raw)-len(mrhyde_word), 'words during the process.')
#creating the wordcloud

    

wordcloud = WordCloud(

    width = 800,

    height = 600,

    background_color = 'white',

    stopwords = list(STOPWORDS) + ['word']).generate(str(mrhyde_word.head(50)))

plt.figure(figsize = (8, 8))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#counting and classifying the emotions of the df



mrhyde_emotions = mrhyde['sentiment'].value_counts()

mrhyde_emotions
#creating the bar chart



mrhyde_emotions.sort_values().plot(figsize=(12,6),kind='barh', color=('mediumseagreen','mediumaquamarine'))

plt.title('"Strange Case of Dr Jekyll and Mr Hyde" Word Analysis', fontsize=20);
print('NUmber of rows in the DataFrame:', len(metamor))

metamor.head()
#creating a new df with no duplicate.



metamor_word = metamor['word'].value_counts()

metamor_word = pd.DataFrame(metamor_word)

print(metamor_word.head(10))



#counting the words left and lost during the filtering



print('How many words left for "The Metamorphosis"? : ', len(metamor_word))

print('"The Metamorphosis" has lost', len(metamor_raw)-len(metamor_word), 'words during the process.')
#creating the wordcloud

    

wordcloud = WordCloud(

    width = 800,

    height = 600,

    background_color = 'white',

    stopwords = list(STOPWORDS) + ['word']).generate(str(metamor_word.head(50)))

plt.figure(figsize = (8, 8))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#counting and classying the words by emotions



metamor_emotions = metamor['sentiment'].value_counts()

metamor_emotions
#creatint the bar plot



metamor_emotions.sort_values().plot(figsize=(12,6),kind='barh', color=('mediumseagreen','mediumaquamarine'))

plt.title('"The Metamorphosis" Word Analysis', fontsize=20);
#creating a new DataFrame for the radar plot



radar_df = pd.DataFrame({

'story': ['The Dunwich Horror','Strange Case of Dr Jekyll and Mr Hyde','The Metamorphosis'],

'positive': [602, 1139, 764],

'negative': [696, 1083, 685],

'trust': [334, 704, 544],

'fear': [415, 715, 280],

'sadness': [320, 530, 417],

'anticipation': [304, 561, 480],

'joy': [195, 402, 340],

'anger': [293, 494, 213],

'surprise': [214, 306, 178],

'disgust': [234, 395, 162]

})



#creating the backgroung



categories=list(radar_df)[1:]

N = len(categories)



angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]



fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(111, projection='polar')



ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)



plt.xticks(angles[:-1], categories)



ax.set_rlabel_position(0)

plt.yticks(color="grey", size=7)



#adding data for The Dunwich Horror

values=radar_df.loc[0].drop('story').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, color='indigo', linestyle='solid', label="The Dunwich Horror")

ax.fill(angles, values, 'indigo', alpha=0.1)



#adding data for Strange Case of Dr Jekyll and Mr Hyde

values=radar_df.loc[1].drop('story').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, color='gold', linestyle='solid', label="Dr. Jekyll and Mr.Hyde")

ax.fill(angles, values, 'gold', alpha=0.1)



#adding data for the metamorphosis

values=radar_df.loc[2].drop('story').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1,color='seagreen', linestyle='solid', label="The Metamorphosis")

ax.fill(angles, values, 'seagreen', alpha=0.1)



#creating the legend



plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1));