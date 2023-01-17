!pip install pymystem3 razdel
import numpy as np

import pandas as pd



from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import normalize, StandardScaler

from sklearn.calibration import CalibratedClassifierCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from scipy.sparse import hstack





from sklearn.metrics import accuracy_score

from pymystem3 import Mystem

import razdel
df_train = pd.read_csv('../input/train.csv').set_index('item_id')

df_test = pd.read_csv('../input/test.csv').set_index('item_id')
def tokenize_with_razdel(text):

    return ' '.join([token.text for token in razdel.tokenize(text)])





mystem = Mystem()

def lemmatize_with_mystem(text):

    return [lemma for lemma in mystem.lemmatize(text) if not lemma.isspace()]
df_train, df_eval = train_test_split(df_train, test_size=0.1, random_state=42)
train_features = []

eval_features = []

test_features = []



def vectorize(data_train, data_eval, data_test, **kwargs):

    vectorizer = TfidfVectorizer(

            tokenizer=lambda text: lemmatize_with_mystem(tokenize_with_razdel(text.lower())), **kwargs)



    train_features.append(vectorizer.fit_transform(tqdm(data_train, desc='Vectorizing train:')))

    eval_features.append(vectorizer.transform(tqdm(data_eval, desc='Vectorizing eval:')))

    test_features.append(vectorizer.transform(tqdm(data_test, desc='Vectorizing test:')))





for column in 'title', 'description':

    for analyzer, ngram_range in (('word', (1, 1)), ('char', (1, 2))):

        vectorize(df_train[column], df_eval[column], df_test[column], analyzer=analyzer, ngram_range=ngram_range)
model = StandardScaler()

train_features.append(normalize(model.fit_transform(df_train['price'].values.reshape(-1, 1))))

eval_features.append(normalize(model.fit_transform(df_eval['price'].values.reshape(-1, 1))))

test_features.append(normalize(model.fit_transform(df_test['price'].values.reshape(-1, 1))))
train_features = normalize(hstack(train_features))

eval_features = normalize(hstack(eval_features))

test_features = normalize(hstack(test_features))
%%time

clf_svm = CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)

clf_svm.fit(train_features, df_train['category_id'])

prediction = clf_svm.predict(eval_features)

acc = accuracy_score(prediction, df_eval['category_id'])
print(f'accuracy = {acc:.3f}')
df_test = df_test.drop(['title', 'description', 'price'], axis=1)

df_test['category_id'] = clf_svm.predict(test_features)

df_test.to_csv('submission.csv')
df = pd.read_csv('../input/category.csv').set_index('category_id')



def add_cat(array):

    while len(array) != 4:

        array.append(' '.join(array))

    return array



df = pd.DataFrame(df['name'].apply(lambda s: s.split('|')).apply(add_cat).tolist())

hierarchy = {i: df[i].to_dict() for i in range(4)}
def replace(array, mask):

    return pd.DataFrame(array).replace(mask).values.ravel()



for level, adapter in hierarchy.items():

    acc = accuracy_score(replace(prediction, adapter), replace(df_eval['category_id'], adapter))

    print(f'level = {level}, accuracy = {acc:.3f}')