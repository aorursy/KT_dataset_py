import pandas as pd

#! pip install twython

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#from nltk import word_tokenize

import nltk

#nltk.download('punkt')

#nltk.download('vader_lexicon')

from nltk.util import ngrams

#from nltk import ngrams
path = '../input/tweet-sentiment-extraction/'



test = pd.read_csv(path + 'test.csv')

train = pd.read_csv(path + 'train.csv')

sample = pd.read_csv(path + 'sample_submission.csv')

#print(train.shape)

train = train.dropna()

#print(train.shape)

test.head()
#test.head()
pol = [0 for k in range(len(test))]

#train['SubSet'] = ['' for g in range(len(train))]

sentimentPolarity = {'neutral': 'neu', 'positive': 'pos', 'negative': 'neg'}

vader = SentimentIntensityAnalyzer()



for t in range(len(test)):

    curSentiment = test.iloc[t]['sentiment']

    vaderScore = vader.polarity_scores(test.iloc[t]['text'])



    pol[t] = float(vaderScore[sentimentPolarity[curSentiment]])

def ngram(text, sentiment, vader, polarity):

    sentimentPolarity = {'neutral': 'neu', 'positive': 'pos', 'negative': 'neg'}

    maxScore = 0

    sentence = ''



    for y in range(1, len(text)):

        sixgrams = ngrams(text, y)

        words = [''.join(gram) for gram in sixgrams]

        flag = 0

        if words:

            for y in words:

                subLoc = text.find(y)

                if subLoc != 0: # and text[subLoc-1] == ' ' and text[subLoc+len(y)] == ' ':

                    if text[subLoc-1] == ' ':

                        if subLoc + len(y) < len(text):

                            if text[subLoc+len(y)] == ' ':

                                scores = vader.polarity_scores(y)

                                curScore = scores[sentimentPolarity[sentiment]]



                                if curScore >= polarity: #curScore == polarity: # or (curScore <= polarity * 1.15 and curScore >= polarity * 0.85):

                                    sentence = y



                                    flag = 1

                                    return sentence

                        else:

                            scores = vader.polarity_scores(y)

                            curScore = scores[sentimentPolarity[sentiment]]



                            if curScore >= polarity: #curScore == polarity: # or (curScore <= polarity * 1.15 and curScore >= polarity * 0.85):

                                    sentence = y



                                    flag = 1

                                    return sentence

                else:

                    if subLoc + len(y) <= len(text):

                        if text[subLoc+len(y)] == ' ':

                            scores = vader.polarity_scores(y)

                            curScore = scores[sentimentPolarity[sentiment]]



                            if curScore >= polarity: #curScore == polarity: # or (curScore <= polarity * 1.15 and curScore >= polarity * 0.85):

                                    sentence = y



                                    flag = 1

                                    return sentence

                    else:

                        scores = vader.polarity_scores(y)

                        curScore = scores[sentimentPolarity[sentiment]]



                        if curScore >= polarity: #curScore == polarity: # or (curScore <= polarity * 1.15 and curScore >= polarity * 0.85):

                                    sentence = y



                                    flag = 1

                                    return sentence





                    

        if flag == 1:

            return sentence

    if len(sentence) == 0:

        return text



    return sentence







vader = SentimentIntensityAnalyzer()

l = []

for t in range(len(test)):



    

    if test.iloc[t]['sentiment'] == 'neutral':

        if test.iloc[t]['text'][0] == ' ':

            l.append(test.iloc[t]['text'][1:])

        else:

            l.append(test.iloc[t]['text'])

    

    else:

        nGramFunc = ngram(test.iloc[t]['text'], test.iloc[t]['sentiment'], vader, pol[t])

        print(nGramFunc)

        l.append(nGramFunc)







test['pred'] = l

ids = test['textID']

text_id = []

for x in range(0,len(ids)):

    text_id.append(ids.iloc[x])

final = pd.DataFrame({'textID':text_id,'selected_text':l})

final.to_csv('submission.csv', index=False)