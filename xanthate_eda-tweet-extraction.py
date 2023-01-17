import numpy as np
import pandas as pd


import os
import re
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
train.head()
train.sentiment.value_counts()
# remove html
# remove wierd spaces
# remove urls
# decompress the compressed words
# correct spellings
# remove html
# distribution length of words
# distriution length of tweets
# readability index
# stop words distribution
def remove_html(text):
    html = re.compile()