# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob


train_df=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample_submission=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

train_df.describe()

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train_df['text'] = train_df['text'].apply(lambda x:clean_text(x))
test_df['text'] = test_df['text'].apply(lambda x:clean_text(x))

test_df.isna()


neutral_test = test_df[test_df['sentiment'] == 'neutral']
print(neutral_test.shape)

pos_test=test_df[test_df['sentiment'] == 'positive']
print(pos_test.shape)

neg_test=test_df[test_df['sentiment'] == 'negative']
print(neg_test.shape)


def neu_scores(test_string):
    a=test_string.split()
    c=[]
    d=[]
    for j in range(1,len(a)+1):
        for i in range(1,len(a)+1):
            c.append(' '.join(a[:i]))
            d.append(len(c[i-1]))
        del a[0]
    
    scores_df=pd.DataFrame(c,columns=['sentences'])
    scores_df['length']=d
    sid = SIA()
    for sentence in scores_df['sentences']:
        scores_df['neutral']= sid.polarity_scores(sentence)['neu'] 
        scores_df['compound']= sid.polarity_scores(sentence)['compound'] 
        scores_df['polarity']= TextBlob(sentence).polarity
        scores_df['subjectivity']= TextBlob(sentence).subjectivity
    sel_text=scores_df.sentences[(max(scores_df['neutral'])) and (max(scores_df['compound'])) and (max(scores_df['length']))]
    del scores_df
    return sel_text


neutral_test['selected_text'] = neutral_test['text'].apply(lambda x:neu_scores(x))
print(neutral_test) 

def neg_scores(test_string):
    a=test_string.split()
    c=[]
    d=[]
    for j in range(1,len(a)+1):
        for i in range(1,len(a)+1):
            c.append(' '.join(a[:i]))
            d.append(len(c[i-1]))
        del a[0]
    
    scores_df=pd.DataFrame(c,columns=['sentences'])
    scores_df['length']=d
    sid = SIA()
    for sentence in scores_df['sentences']:
        scores_df['negative']= sid.polarity_scores(sentence)['neg'] 
        scores_df['compound']= sid.polarity_scores(sentence)['compound'] 
        scores_df['polarity']= TextBlob(sentence).polarity
        scores_df['subjectivity']= TextBlob(sentence).subjectivity
    sel_text=scores_df.sentences[min(scores_df['polarity']) and max(scores_df['negative'])]
    del scores_df
    return sel_text


neg_test['selected_text'] = neg_test['text'].apply(lambda x:neg_scores(x))
print(neg_test)






def pos_scores(test_string):
    a=test_string.split()
    c=[]
    d=[]
    for j in range(1,len(a)+1):
        for i in range(1,len(a)+1):
            c.append(' '.join(a[:i]))
            d.append(len(c[i-1]))
        del a[0]
    
    scores_df=pd.DataFrame(c,columns=['sentences'])
    scores_df['length']=d
    sid = SIA()
    for sentence in scores_df['sentences']:
        scores_df['positive']= sid.polarity_scores(sentence)['pos'] 
        scores_df['compound']= sid.polarity_scores(sentence)['compound'] 
        scores_df['neutral']= sid.polarity_scores(sentence)['neu'] 
        scores_df['polarity']= TextBlob(sentence).polarity
        scores_df['subjectivity']= TextBlob(sentence).subjectivity
    sel_text=scores_df.sentences[max(scores_df['subjectivity']) and max(scores_df['positive']) and max(scores_df['neutral'])]
    del scores_df
    return sel_text


pos_test['selected_text'] = pos_test['text'].apply(lambda x:pos_scores(x))
print(pos_test)


submission= pos_test.append(neg_test)
submission1=submission.append(neutral_test)

sample_submission['selected_text']= submission1['selected_text']
print(sample_submission)

