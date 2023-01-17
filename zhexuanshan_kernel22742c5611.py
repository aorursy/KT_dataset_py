import json
import re
from random import seed, randrange
from math import log
from sklearn.utils import shuffle
import pandas as pd
from nltk.corpus import stopwords
from csv import reader
from sklearn.model_selection import train_test_split

import nltk
nltk.download("stopwords")


df = pd.read_csv('bgg-13m-reviews.csv',index_col=0)
df.head()
df = df.dropna()
df.describe()
reviews = df[['rating','comment']]
reviews.head()
reviews = shuffle(reviews)
reviews.head()

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


x = [review for review in reviews['comment']]
y = [round(r) for r in reviews['rating']]
x_train, x_test, y_train, y_test = train_test_split(review, rate, test_size=0.3, random_state=0)
print('Size of Train Set: ', len(x_train))
print('Size of Test Set: ', len(x_test))

# Get all the words in the training set non-repeatedly and record the index of each word

words_index_dict = {}
index = 0
for rating in x_train:
    for word in rating:
        if word in words_index_dict:
          continue
        else:
            words_index_dict[word]=index
            index+=1



def set_tf(idf):
    temp = []
    for cont in idf:
        temp.append(log(len(x_train)/(cont+1)))
    return temp
tf={}
idf = [0 for _ in range(len(words_index_dict))]
for review_index, review in enumerate(x_train):
    review_counts = pd.value_counts(review)
    for word_index, word in enumerate(review):
        if word not in words_index_dict:
          continue
        else:
            tf[(review_index,words_index_dict[word])] = review_counts[word]/len(review)
            idf[words_index_dict[word]]+=1
idf = set_tf(idf)





class Naive_Bayes:
    def __init__(self, data):
        self.d = data.iloc[:, 1:]
        self.headers = self.d.columns.values.tolist()
        self.prior = np.zeros(len(self.d['Class'].unique()))
        self.conditional = {}
    
    def build(self):
        y_unique = self.d['Class'].unique()
        for i in range(0,len(y_unique)):
            self.prior[i]=(sum(self.d['Class']==y_unique[i])+1)/(len(self.d['Class'])+len(y_unique))
            
        for h in self.headers[:-1]:
            x_unique = list(set(self.d[h]))
            x_conditional = np.zeros((len(self.d['Class'].unique()),len(set(self.d[h]))))
            for j in range(0,len(y_unique)):
                for k in range(0,len(x_unique)):
                    x_conditional[j,k]=(self.d.loc[(self.d[h]==x_unique[k])&(self.d['Class']==y_unique[j]),].shape[0]+1)/(sum(self.d['Class']==y_unique[j])+len(x_unique))
        
            x_conditional = pd.DataFrame(x_conditional,columns=x_unique,index=y_unique)   
            self.conditional[h] = x_conditional       
        return self.prior, self.conditional
    
    def predict(self, X):
        classes = self.d['Class'].unique()
        ans = []
        for sample in X:
            prob = []
            for i in range(len(self.prior)):
                p_i = self.prior[i]
                for j, h in enumerate(self.headers[:-1]):
                    p_i *= self.conditional[h][sample[j]][i]
                prob.append(p_i)
            ans.append(classes[np.argmax(prob)])
        return ans
label_count = [0 for _ in range(11)] + [len(x_train)]
for rating in y_train:
    label_count[rating]+=1
nb = Naive_Bayes()


accuracy = sum([nb.predict(x_test[i]) == y_test[i] for i in range((len(x_test)))])/len(x_test)
print("Accuracy of Test set is:", accuracy)