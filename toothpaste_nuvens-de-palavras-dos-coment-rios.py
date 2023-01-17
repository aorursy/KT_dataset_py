import random as rd



import numpy as np

import pandas as pd



import nltk

from wordcloud import WordCloud



import matplotlib.pyplot as plt



play_store = pd.read_csv('../input/googleplaystore_user_reviews.csv')



print(play_store.shape)

play_store.head(5)
play_store = play_store.loc[play_store.iloc[:, 1].notna(), :]



play_store.Sentiment = play_store.Sentiment.astype('category')



print(play_store.Sentiment.cat.categories)

play_store.head(5)
apps = play_store.App.unique()

app = apps[rd.randint(0, len(apps) - 1)]



user_reviews = play_store.loc[play_store.App == app, ['Translated_Review', 'Sentiment']]

print('Total: {}'.format(user_reviews.shape[0]))



positive = user_reviews.loc[user_reviews.Sentiment == 'Positive', 'Translated_Review']

print('Positive: {}'.format(len(positive)))



negative = user_reviews.loc[user_reviews.Sentiment == 'Negative', 'Translated_Review']

print('Negative: {}'.format(len(negative)))



neutral = user_reviews.loc[user_reviews.Sentiment == 'Neutral', 'Translated_Review']

print('Neutral: {}'.format(len(neutral)))



fig, ax = plt.subplots()



sent = user_reviews.groupby('Sentiment')['Sentiment'].count()

ax.pie(sent , labels = sent.index, autopct = '%1.1f%%')

ax.set_title(app, fontsize = 22)



fig.show()



user_reviews.head(5)
words = []

stopwords = nltk.corpus.stopwords.words('english')



def comp_words(x):

    comment = nltk.word_tokenize(x)

    comment = nltk.pos_tag(comment)

    words.extend([i[0] for i in comment if i[1] == 'JJ'])



positive.apply(comp_words)

words_positive = words.copy()

words.clear()



print('{}\n{}\n\n'.format('POSITIVE', words_positive))



neutral.apply(comp_words)

words_neutral = words.copy()

words.clear()



print('{}\n{}\n\n'.format('NEUTRAL', words_neutral))



negative.apply(comp_words)

words_negative = words.copy()

words.clear()



print('{}\n{}\n\n'.format('NEGATIVE', words_negative))
def join_words(x):

    words = ''

    

    for word in x:

        words += ' ' + word.lower()

        

    return words



wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535)



if len(words_positive) >= 2:

    wordcloud.generate(join_words(words_positive))



    plt.figure(figsize=(16,9))

    plt.imshow(wordcloud)

    plt.title('POSITIVE', fontsize = 32)

    plt.axis("off")

    plt.show()



if len(words_neutral) >= 2:

    wordcloud.generate(join_words(words_neutral))



    plt.figure(figsize=(16,9))

    plt.imshow(wordcloud)

    plt.title('NEUTRAL', fontsize = 32)

    plt.axis("off")

    plt.show()



if len(words_negative) >= 2:

    wordcloud.generate(join_words(words_negative))



    plt.figure(figsize=(16,9))

    plt.imshow(wordcloud)

    plt.title('NEGATIVE', fontsize = 32)

    plt.axis("off")

    plt.show()