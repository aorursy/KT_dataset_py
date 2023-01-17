import pandas as pd

import numpy as np

from joblib import Parallel, delayed



from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import f1_score

import scipy





def simple_pipeline():

    print("Load data")

    train, test = load_data()

    

    data = pd.concat([train, test], axis=0, ignore_index=True)

    print("Vectorization")

    X = vectorization(data.drop('target', axis=1))

    if type(X) == scipy.sparse.coo_matrix:

        X = X.tocsr()

        

    test_mask = data.is_test.values

    

    X_train = X[~test_mask]

    y_train = data['target'][~test_mask]

    

    X_test = X[test_mask]

    if scipy.sparse.issparse(X):

        X_train.sort_indices()

        X_test.sort_indices()



    model = build_model(X_train, y_train)

    

    print("Prediction with model")

    p = model.predict(X_test)

    

    print("Generate submission")

    make_submission(data[test_mask], p)





def load_data():

    train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

    train['is_test'] = False

    

    test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

    test['target'] = -1

    test['is_test'] = True

    

    return train, test





def calculate_validation_metric(model, X, y, metric):

    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    score = cross_val_score(model, X, y, scoring=metric, cv=folds, n_jobs=4)

    

    return np.mean(score), model





def select_model(X, y):

    models = [

        LinearSVC(C=30),

        LinearSVC(C=10),

        LinearSVC(C=3),

        LinearSVC(C=1),

        LinearSVC(C=0.3),

        LinearSVC(C=0.1),

        LinearSVC(C=0.03),

        RidgeClassifier(alpha=30),

        RidgeClassifier(alpha=10),

        RidgeClassifier(alpha=3),

        RidgeClassifier(alpha=1),

        RidgeClassifier(alpha=0.3),

        RidgeClassifier(alpha=0.1),

        RidgeClassifier(alpha=0.03),

        LogisticRegression(C=30),

        LogisticRegression(C=10),

        LogisticRegression(C=3),

        LogisticRegression(C=1),

        LogisticRegression(C=0.3),

        LogisticRegression(C=0.1),

        LogisticRegression(C=0.03),

    ]

    

    results = [calculate_validation_metric(

        model, X, y, 'f1_macro',

    ) for model in models]



    best_result, best_model = max(results, key = lambda x: x[0]) 

    print("Best model validation result: {:.4f}".format(best_result))

    print("Best model: {}".format(best_model))

    

    return best_model





def build_model(X, y):

    print("Selecting best model")

    best_model = select_model(X, y)

    

    print("Refit model to full dataset")

    best_model.fit(X, y)

    

    return best_model



    

def make_submission(data, p):

    submission = data[['id']].copy()

    submission['target'] = p

    submission.to_csv('submission.csv', index=False)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from scipy.sparse import hstack





def vectorization(data):

    """

    data is concatenated train and test datasets with target excluded

    Result value "vectors" expected to have some number of rows as data

    """

    

    

    text = data['text']  

    # char vectorizers grabbed from: https://www.kaggle.com/kcostya/micro-challenge-vectorizers-8f506f/notebook?scriptVersionId=27608159

    vectorizer_char = TfidfVectorizer(

        analyzer='char',

        ngram_range=(3, 7), 

        max_df=0.99, 

        min_df=0.001,

        use_idf=True,

        smooth_idf=True,

        sublinear_tf=False,

        norm='l2'

    )

    

    # count vectorizer with tfidf

    vectors_ngram = TfidfVectorizer(

        ngram_range=(1, 1),

        use_idf=True,

        smooth_idf=True,

        sublinear_tf=False,

        norm='l2'

    )

    

    vectors_char = vectorizer_char.fit_transform(text)

    text = data['text'].apply(lambda s: s.lower()).fillna('') + data['location'].map(str).apply(lambda s: s.lower()).fillna('')  + data['keyword'].map(str).apply(lambda s: s.lower()).fillna('')

    vectors_ngram = vectors_ngram.fit_transform(text)

    vectors = hstack((vectors_char, vectors_ngram))

    

    return vectors

%%time 



simple_pipeline()