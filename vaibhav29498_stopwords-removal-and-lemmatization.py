import pandas as pd

import spacy

from tqdm import tqdm_notebook



tqdm_notebook().pandas()
news = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)

news.head()
news['text'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)

nlp = spacy.load('en_core_web_sm')

news['text'] = news['text'].progress_apply(nlp)
news['text'] = news['text'].progress_apply(

    lambda x: ' '.join(['<NUM>' if str(y).isdigit() else y.lemma_ for y in x if not y.is_stop]))
news = news[['text', 'category']]

news.head()
news.to_pickle('data.pkl')