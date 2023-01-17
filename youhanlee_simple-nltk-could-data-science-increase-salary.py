import numpy as np

import pandas as pd

import re

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer



%matplotlib inline
df_multipleChoice = pd.read_csv("../input/multipleChoiceResponses.csv",  encoding="ISO-8859-1", low_memory=False)

df_freeform = pd.read_csv("../input/freeformResponses.csv", low_memory=False)

df_schema = pd.read_csv("../input/schema.csv", index_col="Column")
temp = pd.DataFrame(df_freeform.loc[df_freeform['SalaryChangeFreeForm'].notnull(), 'SalaryChangeFreeForm'])

temp.reset_index(drop=True, inplace=True)



def change_number(df):

    for i in range(df.shape[0]):

        word = df.loc[i, 'SalaryChangeFreeForm']

        if bool(re.findall('\d+', word)):

            if 'incre' in word.lower():

                print('Positive: {}'.format(word))

                df.loc[i, 'SalaryChangeFreeForm'] = 'Good!'

            elif 'decre' in word:

                print('Negative: {}'.format(word))

                df.loc[i, 'SalaryChangeFreeForm'] = 'Not Good!'

            elif '%' in word:

                print('Positive: {}'.format(word))

                df.loc[i, 'SalaryChangeFreeForm'] = 'Good!'

            else:

                pass

        else:

            continue

    return df
sid = SentimentIntensityAnalyzer()
def vader_polarity(text):

    score = sid.polarity_scores(word)

    if score['pos'] > score['neg']:

        return 1

    elif score['pos'] < score['neg']:

        return 0

    else:

        return -1
temp = change_number(temp)

words = temp.SalaryChangeFreeForm.values



positive_answer = []

negagive_answer = []

neutral_answer = []

for word in words:

    judge = vader_polarity(word)

    if judge == 1:

        positive_answer.append(word)

    elif judge == 0:

        negagive_answer.append(word)

    else:

        neutral_answer.append(word)
print(len(positive_answer)-2, len(neutral_answer)+2, len(negagive_answer))
for ans in positive_answer:

    print(ans)
for ans in negagive_answer:

    print(ans)