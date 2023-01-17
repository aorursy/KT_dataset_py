import numpy as np
import pandas as pd
import math
import re
import random
from csv import reader
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
data = pd.read_csv('/kaggle/input/boardgamegeek-reviews/bgg-13m-reviews.csv')
data[:10]
data_drict = '/kaggle/input/boardgamegeek-reviews/bgg-13m-reviews.csv'
stopwords = (stopwords.words('english'))

with open(data_drict,'r',encoding='utf-8') as f:
    row_data = reader(f)
    review = []
    rate = []
    for row in row_data:
        if  row[0] != '' and row[3] !='':
            rate.append(round(float(row[2])))
            content = row[3].lower()
            content = content.replace("\r", "").strip()  
            content = content.replace("\n", "").strip()
            content = re.sub("[%s]+"%('.,|?|!|:|;\"\-|#|$|%|&|\|(|)|*|+|-|/|<|=|>|@|^|`|{|}|~\[\]'), "", content)
            sentence = content.split(' ')
            for i in stopwords:
                while i in sentence:
                    sentence.remove(i)
            content = ' '.join(sentence)
            review.append(content)
print('Original data size:'+str(len(data)))

print('Vaild data size:'+str(len(review)))
cnt = 11*[0]
for i in range(11):
    cnt[i] = rate.count(i)
import matplotlib.pyplot as plt

name = ['K=0','K=1','K=2','K=3','K=4','K=5','K=6','K=7','K=8','K=9','K=10']
plt.bar(name,cnt)
plt.xlabel('Rating')
plt.ylabel('Number')
for a, b in zip(name, cnt):
 plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=11)
plt.show()
x_train,x_test, y_train, y_test = train_test_split(review,rate,test_size=0.3, random_state=0)
print('Trainset_size:')
print(len(x_train))
print('Testingset_size:')
print(len(x_test))
x_0 = x_train[:100000]
y_0 = y_train[:100000]
x_1 = x_train[100000:200000]
y_1 = y_train[100000:200000]
x_2 = x_train[200000:300000]
y_2 = y_train[200000:300000]
x_test = x_test[:100000]
y_test = y_test[:100000]

test_y = np.asarray(y_test)
vectorizer0 = feature_extraction.text.CountVectorizer()
train0_x = vectorizer0.fit_transform(x_0)
train0_y = np.asarray(y_0)

test_x = vectorizer0.transform(x_test)

NB0 = MultinomialNB()
NB0.fit(train0_x,train0_y)

pred0 = NB0.predict(test_x)
acc_0 = accuracy_score(test_y,pred0)
print('The first NB classifier precision is '+ str(acc_0))
vectorizer1 = feature_extraction.text.CountVectorizer()
train1_x = vectorizer1.fit_transform(x_1)
train1_y = np.asarray(y_1)

test_x = vectorizer1.transform(x_test)

NB1 = MultinomialNB()
NB1.fit(train1_x,train1_y)

pred1 = NB1.predict(test_x)
acc_1 = accuracy_score(test_y,pred1)
print('The second NB classifier precision is '+ str(acc_1))
vectorizer2 = feature_extraction.text.CountVectorizer()
train2_x = vectorizer2.fit_transform(x_2)
train2_y = np.asarray(y_2)

test_x = vectorizer2.transform(x_test)

NB2 = MultinomialNB()
NB2.fit(train2_x,train2_y)

pred2 = NB2.predict(test_x)
acc_2 = accuracy_score(test_y,pred2)

print('The third NB classifier precision is '+ str(acc_2))
result = []
for i in range(len(test_y)):
    pred = []
    pred.append(pred0[i])
    pred.append(pred1[i])
    pred.append(pred2[i])
    tmp = dict((a, pred.count(a)) for a in pred)
    top = sorted(tmp.items() , key=lambda tmp:tmp[1] ,reverse= True)
    result.append(top[0][0])
acc = accuracy_score(test_y,result)
print('The combining NB classifier precision is '+str(acc))
test_y = list(test_y)
cnt = 11*[0]
for i in range(11):
    cnt[i] = test_y.count(i)

name = ['K=0','K=1','K=2','K=3','K=4','K=5','K=6','K=7','K=8','K=9','K=10']
plt.bar(name,cnt)
plt.xlabel('Testing set Rating')
plt.ylabel('Number')
for a, b in zip(name, cnt):
 plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=11)
plt.show()
cnt = 11*[0]
for i in range(11):
    cnt[i] = result.count(i)

name = ['K=0','K=1','K=2','K=3','K=4','K=5','K=6','K=7','K=8','K=9','K=10']
plt.bar(name,cnt)
plt.xlabel('Predicting Rating')
plt.ylabel('Number')
for a, b in zip(name, cnt):
 plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=11)
plt.show()
result1 = []
for i in range(len(test_y)):
    Min = test_y[i] - 1.1
    Max = test_y[i] + 1.1
    if result[i]>Min and result[i]<Max:
        pd = 1
    else:
        pd = 0
    result1.append(pd)
final_acc1 = result1.count(1) / len(test_y)
print('If we keep the rating error of plus or minus one:')
print('Final precision is '+str(final_acc1))
result2 = []
for i in range(len(test_y)):
    Min = test_y[i] - 2.1
    Max = test_y[i] + 2.1
    if result[i]>Min and result[i]<Max:
        pd = 1
    else:
        pd = 0
    result2.append(pd)

final_acc2 = result2.count(1) / len(test_y)
print('If we keep the rating error of plus or minus two:')
print('Final precision is '+str(final_acc2))
from sklearn.externals import joblib
joblib.dump(NB0,  "/kaggle/working/NB0.pkl")
joblib.dump(vectorizer0,  "/kaggle/working/vectorizer0.pkl")
joblib.dump(NB0,  "/kaggle/working/NB1.pkl")
joblib.dump(vectorizer0,  "/kaggle/working/vectorizer1.pkl")
joblib.dump(NB0,  "/kaggle/working/NB2.pkl")
joblib.dump(vectorizer0,  "/kaggle/working/vectorizer2.pkl")