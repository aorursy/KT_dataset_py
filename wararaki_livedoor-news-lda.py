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
!ls /kaggle/input/livedoor-news/text/
from dataclasses import dataclass
from glob import glob
from typing import List

from janome.tokenizer import Tokenizer
from tqdm import tqdm
@dataclass
class News:
    url: str
    date_time: str
    title: str
    content: str
    label: str
class NewsIterator:
    def __init__(self, dir_path: str):
        self._dir_path = dir_path

    def _parse(self, file_path: str) -> News:
        with open(file_path) as f:
            lines = [line.replace('　', '').strip() for line in f if line is not None]
        url = lines[0]
        date_time = lines[1]
        title = lines[2]
        content = ''.join(lines[3:])
        label = file_path.split('/')[1]
        return News(url=url, date_time=date_time, title=title, content=content, label=label)

    def __iter__(self) -> News:
        for file_path in sorted(glob(self._dir_path)):
            if 'LICENSE' in file_path:
                print(f'skip "{file_path}"')
                continue

            news = self._parse(file_path)
            yield news
class TextTokenizer:
    def __init__(self):
        self._tokenizer = Tokenizer()
    
    def __call__(self, text: str) -> List[str]:
        tokens = self._tokenizer.tokenize(text, wakati=True)
        return ' '.join(tokens)
rgx = '/kaggle/input/livedoor-news/text/*/*.txt'
news_iter = NewsIterator(rgx)

tokenizer = TextTokenizer()
n_news = len(glob(rgx))

texts = []
with tqdm(total=n_news) as pbar:
    for news in news_iter:
        texts.append(tokenizer(news.content))
        pbar.update(1)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# model definition
vectorizer = TfidfVectorizer(max_features=10000, min_df=2)
lda = LatentDirichletAllocation(n_components=20, max_iter=20, n_jobs=-1)
vectorizer.fit(texts)
lda.fit(vectorizer.transform(texts))
feature_names = vectorizer.get_feature_names()
for i, component in enumerate(lda.components_[:5]):
    print(f'component: {i}')
    indices = component.argsort()[::-1][:5]
    for index in indices:
        print(f'    {feature_names[index]}: {component[index]}')
