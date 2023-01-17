!pip install youtokentome
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
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



from transformers import BertTokenizerFast

import youtokentome as yttm



from tqdm.notebook import tqdm
tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
train = pd.read_csv('/kaggle/input/ml-guild-classification-task/train.csv')
train.head()
for i in train[train.positive == 1].review.iloc[:10]:

    print(i)

    print()
for i in train[train.negative == 1].review.iloc[:10]:

    print(i)

    print()
for i in train[train.neutral == 1].review.iloc[:10]:

    print(i)

    print()
texts = train['title'].fillna('') + train['review']

pretrained_bpe_texts = [' '.join(map(str, tokenizer.encode(text))) for text in tqdm(texts)]
model = Pipeline([

    ('tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=100000)),

    ('sgd', LogisticRegression(max_iter=1000, verbose=1))

], verbose=1)
labels = ['positive', 'negative', 'neutral']

X_train, X_val, y_train, y_val = train_test_split(texts, train[labels].to_numpy().argmax(axis=1), random_state=0, test_size=0.1)
model.fit(X_train, y_train)
%%time

predict = model.predict_proba(X_val)
roc_auc_score(y_val, predict, multi_class='ovr')
labels = ['positive', 'negative', 'neutral']

X_train, X_val, y_train, y_val = train_test_split(pretrained_bpe_texts, train[labels].to_numpy().argmax(axis=1), random_state=0, test_size=0.1)
model.fit(X_train, y_train)
%%time

predict = model.predict_proba(X_val)
roc_auc_score(y_val, predict, multi_class='ovr')
texts.to_csv('all_reviews.csv', header=None, index=None)
train_data_path = "all_reviews.csv"

model_path = "bpe.model"



# Training model

yttm.BPE.train(data=train_data_path, vocab_size=25000, model=model_path)



# Loading model

bpe = yttm.BPE(model=model_path)



# Two types of tokenization

print(bpe.encode(texts.tolist()[:1], output_type=yttm.OutputType.ID))

print(bpe.encode(texts.tolist()[:1], output_type=yttm.OutputType.SUBWORD))
custom_bpe_texts = [' '.join(map(str, text)) for text in tqdm(bpe.encode(texts.tolist(), output_type=yttm.OutputType.ID))]
labels = ['positive', 'negative', 'neutral']

X_train, X_val, y_train, y_val = train_test_split(custom_bpe_texts, train[labels].to_numpy().argmax(axis=1), random_state=0, test_size=0.1)
model.fit(X_train, y_train)
%%time

predict = model.predict_proba(X_val)
roc_auc_score(y_val, predict, multi_class='ovr')
test = pd.read_csv('/kaggle/input/ml-guild-classification-task/test.csv')
test_texts = test['title'].fillna('') + test['review']
custom_bpe_test_texts = [' '.join(map(str, text)) for text in tqdm(bpe.encode(test_texts.tolist(), output_type=yttm.OutputType.ID))]
sample_submission = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')
sample_submission[labels] = model.predict_proba(custom_bpe_test_texts)
sample_submission.to_csv('submission.csv', index=None)