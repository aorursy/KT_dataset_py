import nltk

import pandas as pd

import numpy as np

import re

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score
raw_data = pd.read_csv('/kaggle/input/hta-tagging/train.csv')

train_raw = raw_data.iloc[0:440,:]

y_label = train_raw.Classes.tolist()
# Custom preprocessor

# Use lemmatizer

# Filter stopword

def preprocessor(content):

    content = content.lower()

    content = re.sub(r'[^\w]', ' ', content)

    _lemmatizer = nltk.stem.WordNetLemmatizer()



    stopword = nltk.corpus.stopwords.words('english')

    token = nltk.tokenize.word_tokenize(content)

    new_content = ""

    for x in token:

        if x not in stopword:

            word1 = _lemmatizer.lemmatize(x, pos = "n")

            word2 = _lemmatizer.lemmatize(word1, pos = "v")

            word3 = _lemmatizer.lemmatize(word2, pos = "a")

            new_content += word3 + ' '

    return new_content[:-1]
def readcontent(filenames):

    content = {}

    for num, filename in enumerate(filenames):

        fd = filename.split("-")[0]

        f = open("/kaggle/input/hta-tagging/train-data/train-data/" + fd + "/" + filename, "r")

        content[num] = f.read()

    return pd.Series(content)
# remove stopword

# concat as a list of string

def transform(data,vectorizer = None):

    if not vectorizer:

        vectorizer = TfidfVectorizer(preprocessor=preprocessor, tokenizer=nltk.tokenize.word_tokenize)

        processedData = vectorizer.fit_transform(data)

    else:

        processedData = vectorizer.transform(data)

    return processedData, vectorizer



train_data, vectorizer = transform(readcontent(train_raw.Filename))
# run SVD (using truncated to reduce the computational power required as it can limit the number of topics)

lsa_model = TruncatedSVD(n_components=6)

lsa_model.fit(train_data)
# Show important terms of each latent topic

print('\nImportant terms of each latent topic\n')



terms = vectorizer.get_feature_names()

for i, comp in enumerate(lsa_model.components_):

    terms_comp = zip(terms, comp)

    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:15]

    print("Topic "+str(i)+": ",end="")

    for t in sorted_terms:

        print(t[0],end="|")

    print("\n")
# Reduce dimension

lsa_model2 = TruncatedSVD(n_components=1000)

lsa_model2.fit(train_data)
features_lsa = lsa_model2.transform(train_data)



print('Original dimension:', train_data.shape)

print('New dimension:', features_lsa.shape)
# Create model from tf-idf as a baseline

clf_tfidf = KNeighborsClassifier()

clf_tfidf.fit(X=train_data, y=y_label)



# Create model from lsa_feature

clf_lsa = KNeighborsClassifier()

clf_lsa.fit(X=features_lsa, y=y_label)
# Test predict train_data

y_pred_train = clf_tfidf.predict(train_data)

y_pred_train_lsa = clf_lsa.predict(features_lsa)



# Compare accuracy of model

print('Accuracy of KNN with train_data: ',accuracy_score(y_label, y_pred_train))

print('Accuracy of KNN with train_data_lsa: ',accuracy_score(y_label, y_pred_train_lsa))
# Test predict test_data

raw_test_data = pd.read_csv('/kaggle/input/hta-tagging/test-sample.csv')

test_raw = raw_test_data.iloc[:,:]

y_test = test_raw.Prediction.tolist()



test_data, vectorizer = transform(readcontent(test_raw.Id), vectorizer=vectorizer)

test_data_lsa = lsa_model2.transform(test_data)

y_pred_test = clf_tfidf.predict(test_data)

y_pred_test_lsa = clf_lsa.predict(test_data_lsa)



# Compare accuracy of model

print('Accuracy of KNN with test_data: ',accuracy_score(y_test, y_pred_test))

print('Accuracy of KNN with test_data_lsa: ',accuracy_score(y_test, y_pred_test_lsa))
# Test predict train_data[440:]

raw_test_data2 = pd.read_csv('/kaggle/input/hta-tagging/train.csv')

test_raw2 = raw_test_data2.iloc[440:,:]

y_test2 = test_raw2.Classes.tolist()



test_data2, vectorizer = transform(readcontent(test_raw2.Filename), vectorizer=vectorizer)

test_data2_lsa = lsa_model2.transform(test_data2)

y_pred_test2 = clf_tfidf.predict(test_data2)

y_pred_test2_lsa = clf_lsa.predict(test_data2_lsa)



# Compare accuracy of model

print('Accuracy of KNN with test_data2: ',accuracy_score(y_test2, y_pred_test2))

print('Accuracy of KNN with test_data2_lsa: ',accuracy_score(y_test2, y_pred_test2_lsa))
result_dict = { 'Id': [], 'Prediction': [] }

result = pd.DataFrame(result_dict)
result['Id'] = train_raw.Filename

result['Prediction'] = y_pred_train
out = result[['Id', 'Prediction']]

out
out.to_csv("out.csv", index=False)