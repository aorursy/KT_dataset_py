import numpy as np

import pandas as pd

import spacy

from spacy.matcher import PhraseMatcher

from spacy.tokens import Span

from spacy import displacy
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
nlp = spacy.load('en_core_web_sm')
print(train.iloc[3,3])
doc = nlp(train.iloc[3,3])

displacy.render(doc,style = "ent")
with nlp.disable_pipes():

    train_vectors = np.array([nlp(text).vector for text in train.text])

    test_vectors = np.array([nlp(text).vector for text in test.text])



print(train_vectors.shape, test_vectors.shape)
from sklearn.model_selection import train_test_split

X_train = train_vectors

y_train = train.target.to_numpy()



train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.1)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier



svc = SVC(kernel='rbf', C=80, gamma='auto', probability=True, random_state=42)

rfc = RandomForestClassifier(n_estimators =100,random_state=42)

xgbc = XGBClassifier(n_estimators=500, learning_rate=0.2, random_state=42)

ext = ExtraTreesClassifier(n_estimators =950,random_state=42)

mlp = MLPClassifier(max_iter=100)
estimators = [svc, rfc,xgbc,ext,mlp]

for estimator in estimators:

    print("Training the", estimator)

    estimator.fit(X_train, y_train)
from sklearn.ensemble import VotingClassifier

vcf = VotingClassifier(estimators=[('svc', svc), ('rfc', rfc), ('xgbc', xgbc),('ext',ext),('mlp',mlp)], voting='soft')

vcf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

preds = vcf.predict(test_vectors)

print(accuracy_score((vcf.predict(test_x)), test_y))

print(len(test['id']), len(preds))
[estimator.score(test_x, test_y) for estimator  in estimators]
submission = pd.DataFrame(columns=['id', 'target'])

submission['id'] = test['id']

submission['target'] = preds

submission.to_csv('submission2.csv', index=False)