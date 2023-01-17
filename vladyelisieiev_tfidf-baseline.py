import numpy as np 

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.linear_model  import LogisticRegression

from sklearn.model_selection import StratifiedKFold
PATH = '/kaggle/input/company-acceptance-prediction/'
train = pd.read_pickle(PATH+'train_df.pkl')

test = pd.read_pickle(PATH+'test_df.pkl')

train
kf = StratifiedKFold(n_splits=5, random_state = 42, shuffle=True)



train_indices = []

test_indices = []



for train_index, test_index in kf.split(train, train['target']):

    train_indices.append(train_index)

    test_indices.append(test_index)



train_indices = np.array(train_indices)

test_indices = np.array(test_indices)
train['text_str'] =  train['text'].apply(lambda x: '[TEXTEND]'.join(x))

test['text_str'] =  test['text'].apply(lambda x: '[TEXTEND]'.join(x))



train['keywords_str'] = train['keywords'].apply(lambda x: ' '.join(x))

test['keywords_str'] = test['keywords'].apply(lambda x: ' '.join(x))



train.drop(columns = ['id','html','text','keywords'], inplace = True)

test.drop(columns = ['id','html','text','keywords'], inplace = True)
SEP = "[SEP]"



train_df = pd.DataFrame()

train_df["target"] = train["target"]

train_df["text"] = (

    #train["text_str"]

    #+ SEP

    + train["keywords_str"]

    + SEP

    + train["accepted_function"]

    + SEP

    + train["rejected_function"]

    + SEP

    + train["accepted_product"]

    + SEP

    + train["rejected_product"]

    + SEP

)



test_df = pd.DataFrame()

test_df["text"] = (

    #test["text_str"]

    #+ SEP

    + test["keywords_str"]

    + SEP

    + test["accepted_function"]

    + SEP

    + test["rejected_function"]

    + SEP

    + test["accepted_product"]

    + SEP

    + test["rejected_product"]

    + SEP

)
def target_metric(y_true,y_pred):

    weights = np.ones(y_true.shape[0])

    weights[y_true==2] = 2

    return weights @ (y_true == y_pred).astype("int") / weights.sum()
word_vectorizer = TfidfVectorizer(

    analyzer='word',

    stop_words='english',

    ngram_range=(1, 2),

    lowercase=True,

    min_df=5,

    max_features=100000)
print('Start validation: ')

log_reg = LogisticRegression(solver='liblinear', random_state=42)

word_vectorizer = TfidfVectorizer(

    analyzer='word',

    stop_words='english',

    lowercase=True,

    min_df=5,

    max_features=100000)



model = Pipeline([('vectorizer', word_vectorizer),  ('log_reg', log_reg)])





metrics = []

results = []



n_fold = 0

for train_idx, test_idx in zip(train_indices, test_indices):

    n_fold+=1

    print("Fold: "+str(n_fold))

    model.fit(train_df['text'].iloc[train_idx], train_df['target'].iloc[train_idx])

    y_pred = model.predict(train_df['text'].iloc[test_idx])

    

    y_true = train_df['target'].iloc[test_idx]

    metrics.append(target_metric(y_true, y_pred))

    results.append(model.predict_proba(test_df['text']))

    print("Accuracy over fold: "+str(metrics[-1]))

    

print('\nAvarage Accuracy.: '+str(np.mean(metrics)))
results = np.array(results)

results.shape
prediction = pd.Series(results.sum(axis = 0).argmax(-1))

prediction.value_counts()
submission = pd.read_csv(PATH+'sample_submission.csv')
submission['target'] = prediction

submission.to_csv('tfidf_baseline.csv', index = False)