# reference kernels

# stacked classifiers https://www.kaggle.com/thomasnelson/simple-stacking-classifier-for-beginners 

# important features from https://www.kaggle.com/cdeotte/lb-probing-strategies-0-890-2nd-place
import numpy as np

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



# load data

train = pd.read_csv('../input/train.csv')

targets = train['target']

train.drop(['id','target'], axis='columns', inplace=True)

test = pd.read_csv('../input/test.csv').drop("id", axis='columns')



RANDOM_SEED = 123



nb = GaussianNB()

svc = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)

sgd = SGDClassifier(eta0=1, max_iter=100, tol=0.0001, alpha=0.01, l1_ratio=0.0, learning_rate='adaptive', loss='log', penalty='elasticnet')

num_folds = 15

repeats = 5



from mlxtend.classifier import StackingCVClassifier

np.random.seed(RANDOM_SEED)

lr = LogisticRegression(max_iter=100, class_weight='balanced', penalty='l2', C=0.1, solver='liblinear')

sclf = StackingCVClassifier(classifiers=[lr, svc, sgd], 

                            use_probas=True,

                            use_features_in_secondary=True,

                            meta_classifier=nb,

                            cv=num_folds)



folds = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = repeats, random_state=16)

test_result = np.zeros(len(test))

auc_score = 0

v = [16,33,45,63,65,73,91,108,117,164,189,199,209,217,239]

features = list(map(str,v))



train = train[features]



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):

    print("Fold: ", fold_ + 1)

    

    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]

    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]

    

    data = RobustScaler().fit_transform(np.concatenate((X_train, X_valid), axis=0))

    X_train = data[:len(X_train)]

    X_valid = data[len(X_train):]



    sclf.fit(X_train, y_train.values)

    

    y_pred = sclf.predict_proba(X_valid)

    auc = roc_auc_score(y_valid, y_pred[:, 1])

    print(auc)

    auc_score += auc



    preds = sclf.predict_proba(test[features])

    test_result += preds[:, 1]
# print the average AUC across the folds and compute the final results on the test data

auc_score = auc_score / (num_folds*repeats)

print("AUC score: ", auc_score)

test_result = test_result /(num_folds*repeats)



# create the submission

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = test_result

submission.to_csv('stacked_classifier_repeatedfolds.csv', index=False)



# DISPLAY HISTOGRAM OF PREDICTIONS

submission["target"].plot.hist(bins=300, alpha=0.5)
