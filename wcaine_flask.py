import json

import pandas as pd

train = pd.read_json('../input/whats-cooking-kernels-only/train.json')

test = pd.read_json('../input/whats-cooking-kernels-only/test.json')
from sklearn.model_selection import train_test_split

X_train = list(train['ingredients'])

y_train = list(train['cuisine'])

X_test = list(test['ingredients'])
print(len(y_train))

print(len(set(y_train)))
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english'))
from nltk.stem.porter import *

stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize

def preprocess_ingredient(ingredients):

    for ingredient in ingredients:

        for i in range(len(ingredient)):

            x = ingredient[i] #'Bertolli® Classico Olive Oil', '(10 oz.) frozen chopped spinach, thawed and squeezed dry' ,'leg of lamb', 'lamb leg'

            x = x.lower() #'bertolli® classico olive oil', '(10 oz.) frozen chopped spinach, thawed and squeezed dry' ,'leg of lamb', 'lamb leg'

            x = re.sub("[^a-z ]", "", x) #'bertolli classico olive oil', ' oz frozen chopped spinach thawed and squeezed dry' ,'leg of lamb', 'lamb leg'

            word_tokens = word_tokenize(x)

            if 'oz' in word_tokens:

                word_tokens.remove('oz')

            filtered_words = [w for w in word_tokens if not w in stop_words] 

            filtered_words.sort() #['bertolli', 'classico', 'oil', 'olive'], ['chopped', 'dry', 'frozen', 'spinach', 'squeezed', 'thawed'], ['lamb', 'leg'], ['lamb', 'leg']

            stemmed_word = [stemmer.stem(word) for word in filtered_words]

            x = ' '.join(stemmed_word) #'bertolli classico oil oliv', 'chop dri frozen spinach squeez thaw' ,'lamb leg', 'lamb leg'

            ingredient[i] = x
preprocess_ingredient(X_train)

preprocess_ingredient(X_test)
def create_vocabs(X):

    vocabs = set()

    for ingredient in X:

        vocabs.update(ingredient)

    return sorted(vocabs)
X_C_train, X_C_val, y_C_train, y_C_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
C_train_vocabs = create_vocabs(X_C_train)

train_vocabs = create_vocabs(X_train)
import pickle

with open('vocabs.pkl', 'wb') as file:

    pickle.dump(train_vocabs, file)

# with open('train_vocabs.pkl', 'rb') as file:  

#     pickled_train_vocabs = sorted(pickle.load(file))
def create_bag_of_words(ingredients, vocabs):

    data_features = list()

    for ingredient in ingredients:

        features = list()

        for item in vocabs:

            features.append(item in ingredient)

        data_features.append(features)

    return data_features
# from sklearn import preprocessing



C_train_data_features = create_bag_of_words(X_C_train, C_train_vocabs)



C_val_data_features = create_bag_of_words(X_C_val, C_train_vocabs)



# train_data_features = preprocessing.scale(create_bag_of_words(X_train))

train_data_features = create_bag_of_words(X_train, train_vocabs)



# test_data_features = preprocessing.scale(create_bag_of_words(X_test))

test_data_features = create_bag_of_words(X_test, train_vocabs)
def train_logistic_regression(c, features, label):

    from sklearn.linear_model import LogisticRegression

    ml_model = LogisticRegression(C=c, max_iter = 40000)

    ml_model.fit(features, label)

    return ml_model
c_trial = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100]

best_c = c = 0

best_score = 0

from sklearn.metrics import accuracy_score

for c in c_trial:

    ml_model = train_logistic_regression(c, C_train_data_features, y_C_train)

    predicted_y = ml_model.predict(C_val_data_features)

    score = accuracy_score(y_C_val, predicted_y)

    print(c, score)

    if score > best_score:

        best_score = score

        best_c = c



print(best_c, best_score)

ml_model = train_logistic_regression(best_c, train_data_features, y_train)

predicted_y = ml_model.predict(test_data_features)
test['cuisine'] = predicted_y

test[['id', 'cuisine']].to_csv('submission.csv', index=False)

test[['id', 'cuisine']].head()
# import pickle

# with open('cuisine_prediction_model.pkl', 'wb') as file:

#     pickle.dump(ml_model, file)

    

# with open('cuisine_prediction_model.pkl', 'rb') as file:  

#     pickled_ml_model = pickle.load(file)



import joblib

joblib.dump(ml_model, 'cuisine_prediction_model_joblib.pkl')

# pickled_ml_model = joblib.load('cuisine_prediction_model_joblib.pkl') 
# !pip freeze > '../working/dockerimage_snapshot.txt'