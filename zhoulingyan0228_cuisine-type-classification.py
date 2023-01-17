import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pprint import pprint
df = pd.read_json('../input/train.json')
df['ingredients'] = df['ingredients'].apply(lambda x: '|'.join(x))
df.head()
sns.countplot(x="cuisine", data=df)
plt.xticks(rotation=60)
features = df['ingredients'].values

labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(df['cuisine'])
labelCount = labels.shape[0]

resampling=RandomOverSampler()
resampling.fit(np.arange(labelCount).reshape(-1,1), labels.reshape(-1,1))
sampledIndex, _ = resampling.sample(np.arange(labelCount).reshape(-1,1), labels.reshape(-1,1))
sampledIndex = sampledIndex.flatten()

featuresSampled = features[sampledIndex]
labelsSampled = labels[sampledIndex]
sns.countplot(x="label", data=pd.DataFrame({'label':labelsSampled}));
cv = CountVectorizer(stop_words=None, token_pattern='.+', tokenizer=lambda x: x.split('|'), max_df=1., min_df=1)
mostLeastIngr = []
for l in labelEncoder.classes_:
    ingr = df.loc[df['cuisine']==l, 'ingredients']
    ingrVect = cv.fit_transform(ingr)
    vocabularyDict = cv.vocabulary_
    invVocabularyDict = {v: k for k, v in vocabularyDict.items()}
    s = np.array(ingrVect.sum(0))[0]
    idx = np.argsort(s)
    mostLeastIngr.append([l, invVocabularyDict[idx[-1]], invVocabularyDict[idx[-2]], invVocabularyDict[idx[-3]], invVocabularyDict[idx[2]], invVocabularyDict[idx[1]], invVocabularyDict[idx[0]]])
mostLeastIngr = pd.DataFrame(mostLeastIngr, columns=['cuisine','top1','top2','top3','bottom3','bottom2','bottom1'])
mostLeastIngr
cv = CountVectorizer(stop_words=None, token_pattern='.+', tokenizer=lambda x: x.split('|'), max_df=1., min_df=0.1)
ingrVect = cv.fit_transform(features)
vocabularyDict = cv.vocabulary_
invVocabularyDict = {v: k for k, v in vocabularyDict.items()}
s = np.array(ingrVect.sum(0))[0]
idx = np.argsort(s)
columns =[invVocabularyDict[idx[-1]], invVocabularyDict[idx[-2]], invVocabularyDict[idx[-3]], invVocabularyDict[idx[2]], invVocabularyDict[idx[1]], invVocabularyDict[idx[0]]]
mostLeastIngr = df[['cuisine']].copy()
mostLeastIngr['cuisine']
for i in [-1, -2, -3, 2, 1, 0]:
    mostLeastIngr[invVocabularyDict[idx[i]]] = ingrVect[:, idx[i]].toarray().flatten()
ingrFreqPerCuisine = mostLeastIngr.groupby('cuisine').mean()

ingrFreqPerCuisine.plot.bar(figsize=(15,5));
kfold = KFold(3)
i = 1

for train_idx, validate_idx in kfold.split(featuresSampled, labelsSampled):
    ingr_clf = Pipeline([('Vect', CountVectorizer(stop_words=None, token_pattern='.+', tokenizer=lambda x: x.split('|'), max_df=0.9, min_df=1)),
                         #('FeatureSelection', SelectPercentile(sklearn.feature_selection.chi2, percentile=50)),
                         #('ToDense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                         ('Classif', LinearSVC())])
    ingr_clf.fit(featuresSampled[train_idx], labelsSampled[train_idx])
    predicted = ingr_clf.predict(featuresSampled[validate_idx])
    print(accuracy_score(predicted, labelsSampled[validate_idx]))
    plt.subplot(3,1,i)
    plt.imshow(confusion_matrix(predicted, labelsSampled[validate_idx]))
    i+=1
ingr_clf = Pipeline([('Vect', CountVectorizer(stop_words=None, token_pattern='.+', tokenizer=lambda x: x.split('|'), max_df=1., min_df=1)),
                     #('ToDense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
                     ('Classif', LinearSVC())])
ingr_clf.fit(df['ingredients'], labels)
testDf = pd.read_json('../input/test.json')
testDf['ingredients'] = testDf['ingredients'].apply(lambda x: '|'.join(x))
predictedEncoded = ingr_clf.predict(testDf['ingredients'])
predictedLabels = labelEncoder.inverse_transform(predictedEncoded)
outDf = testDf[['id']]
outDf['cuisine'] = predictedLabels
outDf.to_csv('submission.csv', index=False)