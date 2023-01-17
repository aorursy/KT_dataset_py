import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
data_dir = "../input/"

df = pd.read_csv(data_dir + "/spam.csv", encoding = 'latin-1')



print (df.shape)

print (df.head())
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df.v2, df.v1, test_size=0.2, random_state=0)

print (X_train.head())

print (y_train.head())
vect = CountVectorizer()

example = ['I love you, good bad bad', 'you are soo good']

example2 = ['hope not me, hey hey wu suo', 'qian mian de luoo']



result = vect.fit_transform(example)

result2 = vect.transform(example2)

print (result.shape)

print (result)

print (result2.shape)

print (result2)
vectorizer = CountVectorizer()

X_train_count = vectorizer.fit_transform(X_train)

X_test_count = vectorizer.transform(X_test)

print (X_train_count.shape)

print (X_train_count)

#print (vectorizer.vocabulary_)

print (X_test_count.shape)

print (X_test_count)
# show how word count looks like

word_freq_df = pd.DataFrame({'word': vectorizer.get_feature_names(), 'occurrences': X_train_count.toarray().sum(axis=0)})

plt.plot(word_freq_df.occurrences)

plt.show()



word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])

word_freq_df_sort = word_freq_df.sort_values(by=['occurrences'], ascending=False)

word_freq_df_sort.head()
from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB()

classifier.fit(X_train_count, y_train)

y_pred = classifier.predict(X_test_count)

print (y_pred)
# Check model accuracy

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score



print (accuracy_score(y_test, y_pred))

print (classification_report(y_test, y_pred))

print (confusion_matrix(y_test, y_pred))



# this time just calculate the train set

cross_val = cross_val_score(classifier, X_train_count, y_train, cv=20, scoring='accuracy')

print (cross_val)

print (np.mean(cross_val))
def get_vocabulary(data):

    vocabulary_set = set([])

    for document in data:

        words = document.split()

        for word in words:

            vocabulary_set.add(word)

    return list(vocabulary_set)



vocabulary_list = get_vocabulary(df.v2)

print (df.v2.shape)

def document_2_vector(vocabulary_list, data):

    word_vector = np.zeros(len(vocabulary_list))

    words = data.split()

    for word in words:

        if word in vocabulary_list:

            word_vector[vocabulary_list.index(word)] += 1

    return word_vector
train_matrix = []

for document in X_train:

    word_vector = document_2_vector(vocabulary_list, document)

    train_matrix.append(word_vector)

print (len(train_matrix))
def naive_bayes_train(train_matrix, y_train):

    docs_num = len(train_matrix)

    words_num = len(train_matrix[0])

    

    spam_vector_count = np.ones(words_num)

    ham_vector_count = np.ones(words_num)

    spam_total_count = words_num

    ham_total_count = words_num

    spam_count = 0

    ham_count = 0

    

    for i in range(docs_num):

#         if i > 100:

#             break

        if y_train[i] == 'spam':

            spam_vector_count += train_matrix[i]

            spam_total_count += sum(train_matrix[i])

            spam_count += 1

        else:

            ham_vector_count += train_matrix[i]

            ham_total_count += sum(train_matrix[i])

            ham_count += 1

    p_spam_vector = np.log(spam_vector_count/spam_total_count)

    p_ham_vector = np.log(ham_vector_count/ham_total_count)

    return p_spam_vector, np.log(spam_count / docs_num), p_ham_vector, np.log(ham_count / docs_num)

p_spam_vector, p_spam, p_ham_vector, p_ham = naive_bayes_train(train_matrix, y_train.values)



print (p_spam_vector)

    
def doc_predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham):

    spam = sum(test_word_vector * p_spam_vector) + p_spam

    ham = sum(test_word_vector * p_ham_vector) + p_ham

    if (spam > ham):

        return 'spam'

    else:

        return 'ham'

    

def predict(X_test, p_spam_vector, p_spam, p_ham_vector, p_ham):

    predictions = []

    for document in X_test.values:

        test_word_vector = document_2_vector(vocabulary_list, document)

        ans = doc_predict(test_word_vector, p_spam_vector, p_spam, p_ham_vector, p_ham)

        predictions.append(ans)

    return predictions



y_pred = predict(X_test, p_spam_vector, p_spam, p_ham_vector, p_ham)
# Check model accuracy

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import cross_val_score



print (accuracy_score(y_test, y_pred))

print (classification_report(y_test, y_pred))

print (confusion_matrix(y_test, y_pred))