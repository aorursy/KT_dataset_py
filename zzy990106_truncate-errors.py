import numpy as np

import pandas as pd
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
def find_failed_start(text,selected_text):

    """find the case that lable is not truncated by space"""

    text = str(text)

    selected_text = str(selected_text)

    begin = text.find(selected_text)

    end = begin + len(selected_text)

    if begin == 0:

        return False

    return text[begin -1].isalpha()



def find_failed_end(text,selected_text):

    """find the case that lable is not truncated by space"""

    text = str(text)

    selected_text = str(selected_text)

    begin = text.find(selected_text)

    end = begin + len(selected_text)

    if end == len(text):

        return False

    return text[end].isalpha()



failed_start = train[train.apply(lambda row: find_failed_start(row.text, row.selected_text),axis=1)]

failed_end = train[train.apply(lambda row: find_failed_end(row.text, row.selected_text),axis=1)]

failed_ids = set(failed_start.textID.tolist() + failed_end.textID.tolist())
failed_start
for i in failed_start.index:

    text = str(failed_start.loc[i,'text'])

    selected_text = str(failed_start.loc[i,'selected_text'])

    begin = text.find(selected_text)

    end = begin + len(selected_text)

    i = 1

    while text[begin-i]!=' ' and begin-i>=0:

        i += 1

    print(text[begin-i:end], '->', selected_text, '\n----------------')
failed_end
for i in failed_end.index:

    text = str(failed_end.loc[i,'text'])

    selected_text = str(failed_end.loc[i,'selected_text'])

    begin = text.find(selected_text)

    end = begin + len(selected_text)

    i = 0

    while end+i<len(text) and text[end+i]!=' ':

        i += 1

    print(text[begin:end+i], '->', selected_text, '\n----------------')
for i in train.index:

    text = str(train.loc[i,'text'])

    selected_text = str(train.loc[i,'selected_text'])

    begin = text.find(selected_text)

    end = begin + len(selected_text)

    i = 1

    while text[begin-i]!=' ' and begin-i>=0:

        i += 1

    if selected_text[0:2] == 'as':

        print(text[begin-i:end], '|', selected_text, '\n----------------')