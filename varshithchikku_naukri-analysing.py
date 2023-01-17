import numpy as np

import pandas as pd

import matplotlib as plt

import sklearn

%matplotlib inline

df = pd.read_csv('../input/naukri_com-job_sample.csv')

df.head()
df.describe()
df.count()
df.describe(include = ['object'])


#Indian Institute of Technology Bombay is the higest recuter  from your site with frequency of 403 different offerings

##The most frequent expirence is 2-7 years 

###The most common jobs are in IT  fiels and in banglore

####there are 45 different types of skills 

df['numberofpositions'].isnull()
df['numberofpositions'].mean()
df['numberofpositions'].fillna(6, inplace = True)
df['numberofpositions'].head()
df['company'].value_counts().head(10)
df['company'].value_counts().head().plot(kind = 'bar')
df['payrate'].value_counts().head()
df['education'].value_counts().head()
df['education'].unique()
df['education'].nunique()
df['industry'].value_counts().head()
df['joblocation_address'].value_counts().head(10)
replacements = {

   'joblocation_address': {

      r'(Bengaluru/Bangalore)': 'Bangalore',

      r'Bengaluru': 'Bangalore',

      r'Hyderabad / Secunderabad': 'Hyderabad',

      r'Mumbai , Mumbai': 'Mumbai',

      r'Noida': 'NCR',

      r'Delhi': 'NCR',

      r'Gurgaon': 'NCR', 

      r'Delhi/NCR(National Capital Region)': 'NCR',

      r'Delhi , Delhi': 'NCR',

      r'Noida , Noida/Greater Noida': 'NCR',

      r'Ghaziabad': 'NCR',

      r'Delhi/NCR(National Capital Region) , Gurgaon': 'NCR',

      r'NCR , NCR': 'NCR',

      r'NCR/NCR(National Capital Region)': 'NCR',

      r'NCR , NCR/Greater NCR': 'NCR',

      r'NCR/NCR(National Capital Region) , NCR': 'NCR', 

      r'NCR , NCR/NCR(National Capital Region)': 'NCR', 

      r'Bangalore , Bangalore / Bangalore': 'Bangalore',

      r'Bangalore , karnataka': 'Bangalore',

      r'NCR/NCR(National Capital Region)': 'NCR',

      r'NCR/Greater NCR': 'NCR',

      r'NCR/NCR(National Capital Region) , NCR': 'NCR'

       

   }

}



df.replace(replacements, regex=True, inplace=True)

y = df['joblocation_address'].value_counts()
y.head(10)
df['joblocation_address'].value_counts().head().plot(kind = 'bar')
df['industry'].value_counts().head(10)
df['industry'].value_counts().head(10).plot(kind = 'bar')
p = df['industry'].value_counts().head(20)

q = df['joblocation_address'].value_counts().head(20)
df['jobtitle'].value_counts().head(10)
df['jobtitle'].value_counts().head(10).plot(kind = 'bar')
df.head()

df['payrate'].value_counts().head(10)
df['payrate'].value_counts().head().plot(kind = 'bar')
#df['payrate'].value_counts().head(10).plot(kind = 'bar')

#this paticular  cell is to display the elements from 1st position

#df.loc()

a = df['payrate'].value_counts().head(10)

a

df.info(memory_usage = 'deep')
#df.iloc[1:10,9].value_counts()

a.iloc[1:,].plot(kind = 'bar')

#This representation of payrates of actually given by companies
#df['payrate'].head(20)

df['industry'].head(10)
#df['payrate'].exclude('Not Disclosed by Recruiter').head(20)

X = df.payrate[(df.payrate != 'Not Disclosed by Recruiter')]

X.head(10)
X.count()
#This is in the form of cobject

#in order to train the data sets we have to convert it into feture vectors

B = df.groupby('industry').payrate
v = df['payrate'].isnull() 

v.unique().sum()
#No null values in the payment section

#so we will do classification with respect to payment and industry

df_class = df.filter(['payrate','industry'], axis=1)
df9 = df['industry'] +df['experience']

df9.head()
df_class.head(10)
#df_class.drop('Not Disclosed by Recruiter', axis = 1 )

#df_class = [(df_class.payrate != 'Not Disclosed by Recruiter')]

df_class1 = df_class[df_class.payrate != 'Not Disclosed by Recruiter']

df_class1.head()
#X.head(10)

df_class1.describe()
#p = df_class1['payrate' == '12,00,000 - 22,00,000 P.A']

#p
#df_class1.payrate

df_class1['payrate'] = df_class1['payrate'].str.replace(',', '')

'''

df_class1['payrate'] = df_class1['payrate'].astype(str)

df_class1['payrate'] = df_class1['payrate'].replace(',', '')

df_class1['payrate'] = df_class1['payrate'].astype(float)

df_class1.payrate.head()

'''
#df_class1['payrate'] = df_class1['payrate'].str.extract('(\d)', expand=False)

#df_class1['payrate'] = df_class1.payrate.str.replace(r"[a-zA-Z]",'')

#df_class1.payrate

data_set = df_class1.payrate

data_set.head()
data_set.dropna(inplace = True)

#print(data_set[3768])

#data_set.drop(df.index[3768], inplace=True)

#print(data_set[3768])
#print(data_set)

array_data = []

bottom_data = []

top_data = []

index_numbers = []

index_numbers = data_set.index

new_index_numbers = []

#print(data_set[2])

#print(data_set.index)

for element in index_numbers:

    #print(element)

    #print(data_set[element])

    #if data_set[element] == nan:

        #print("found null")

    if data_set[element].find("0") != -1:

        #print(array_data[-1].find(" P.A"))

        array_data.append(data_set[element][:data_set[element].find(" P.A")])

        #print(element)

        new_index_numbers.append(element)

        continue

    #else:

        #print("to be removed")

    #remove element

    

#print(array_data)

#print(index_numbers)

total = 0

skipped = 0

newer_index_numbers = []

for element in range(len(array_data)):

    #print(element)

    total += 1

    if array_data[element][:array_data[element].find(" ")].isdigit() == False or array_data[element][array_data[element].find(" ")+3:].isdigit() == False:

        #print("Skipped", element)

        skipped += 1

        continue

    bottom_data.append(int(array_data[element][:array_data[element].find(" ")]))

    top_data.append(int(array_data[element][array_data[element].find(" ")+3:]))

    newer_index_numbers.append(new_index_numbers[element])



print("total = ", total)

print("skipped = ",skipped)



#print(new_index_numbers)   

#print(top_data)

#print(bottom_data)

average_data = []

for element in range(len(top_data)):

    average_data.append((top_data[element]+bottom_data[element])/2)

#print(average_data)
print(average_data)
print(len(average_data))

#df_class1

#print(newer_index_numbers)
d = {'payrate' : average_data}

df2 = pd.DataFrame(data=d, index= newer_index_numbers)

df10 = pd.DataFrame(data=d, index= newer_index_numbers)

df2.head()
df2['industry'] = df_class1['industry']
df10['industry'] = df9

df10.head()
df2.head()
df_class1.industry.head(10)
#df_class1.drop()

#df_class1[df_class1.industry == 'IT-Software / Software Services']
#train_x = pandas.get_dummies(test[cols])

X = df2.industry

y = df2.payrate

#X.head()

y.head()
#X.value_counts()

#P =df.industry

#y.head(20)

#df_class1['industry'].apply(lambda row: row.astype(str).str.contains('IT-Software / Software Services').any())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



#X = pandas.get_dummies(test[cols])



'''

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures



if addpoly:

    all_data = pd.concat((X_train,

                          X_test), ignore_index=True)



    scaler = MinMaxScaler()

    scaler.fit(all_data)

    all_data=scaler.transform(all_data)

    poly = PolynomialFeatures(2)

    all_data=poly.fit_transform(all_data)



    X_train = all_data[:train_dataset.shape[0]]

    X_test = all_data[train_dataset.shape[0]:]

    ##

    print(X_train.shape)

    print(Y_train.shape)

    print(X_test.shape)

 '''



#print(X_train)

print(X_test)
#from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(X_train)

list(le.classes_)
le.transform(X_train)
le.fit(X_test)

le.transform(X_test)
from sklearn.svm import LinearSVC, SVC

from sklearn.datasets import make_classification

X_train, y_train = make_classification()

clf = LinearSVC(multi_class = 'crammer_singer')

clf.fit(X_train, y_train)



X_test, y_test = make_classification()

#clf = LinearSVC(multi_class = 'crammer_singer')

#clf.fit(X_test, y_test)

print(clf.coef_)

print(clf.intercept_)
clf.predict(X_test)

print(clf.decision_function(X_test))

clf.score(X_test, y_test, sample_weight=None)

#we got the 55 percentage with the Liner SVM
#Now we try the Naive Bayes

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(X_train, y_train)
clf.predict(X_test)

clf.score(X_test, y_test, sample_weight = None)
#now we will apply the decision tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test, sample_weight=None)
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(SVC(kernel = 'linear'))

clf.fit(X_train,y_train)

clf.predict(X_test)

clf.score(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
#as the accuracy is not satisfying we will go for the rnns

#implenantion of rnn using keras

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import LSTM



#LSTM model with rnn in keras

#model = Sequential()

#model.add(LSTM(4,input_shape=(1, p)))
df10.head()

#until now we have predicted with only two values

#but now we will add another extra column i.e expirence

#we will join this column along with the training set i.e industry
P = df10.industry

q = df10.payrate

from sklearn.model_selection import train_test_split

P_train,P_test, q_train, q_test = train_test_split(P, q, test_size=0.30, random_state=42)



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(P_train)

list(le.classes_)
le.transform(P_train)
le.fit(P_test)

le.transform(P_test)
from sklearn.svm import LinearSVC, SVC

from sklearn.datasets import make_classification

P_train, q_train = make_classification()

clf = LinearSVC(multi_class = 'crammer_singer')

clf.fit(P_train, q_train)



P_test, q_test = make_classification()

#clf = LinearSVC(multi_class = 'crammer_singer')

#clf.fit(X_test, y_test)

print(clf.coef_)

print(clf.intercept_)

print(clf.predict(P_test))

clf.score(P_test, q_test)
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(SVC(kernel = 'linear'))

clf.fit(P_train,q_train)

clf.predict(P_test)

clf.score(P_test, q_test)
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

clf = DecisionTreeClassifier(random_state=0)

clf.fit(P_train, q_train)

clf.predict(P_test)

clf.score(P_test, q_test, sample_weight = None)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(P_train, q_train)

clf.predict(P_test)

clf.score(P_test, q_test)
















'''

class MultinomialEncoder:

    def _init_(self, columns = None):

        self.columns = columns

        

    def fit(self, X_test, y_test = None):

        return self

    

    def transform(self, X_test):

        output = X_test.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output

    

    def fit_transform(self,X_test,y_test=None):

        return self.fit(X_test,y_test).transform(X_test)

  

    '''
'''

#from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.pipeline import Pipeline

class MultinomialEncoder:

    def _init_(self, columns = None):

        self.columns = columns

        

    def fit(self, X_train, y_train = None):

        return self

    

    def transform(self, X_train):

        output = X_train.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output

    

    def fit_transform(self,X_train,y_train=None):

        return self.fit(X_train,y_train).transform(X_train)

  '''  

    

    

    

#le = preprocessing.LabelEncoder()

#le.fit(X_train)

#le = LabelEncoder().fit_transform

#le.classes_

#X_train.apply(LabelEncoder().fit_transform)
'''

#from sklearn.preprocessing import OneHotEncoder

#enc = OneHotEncoder()

#enc.fit(X_train)

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(X_train, y_train)

'''
'''

#from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_train = CountVectorizer()

vectorizer_train.fit(X_train)

vector_train = vectorizer_train.transform(X_train)

print(vector_train.shape)

print(type(vector_train))

print(vector_train.toarray())

'''



#print(clf.predict(X_test))
'''

vectorizer_test = CountVectorizer()

vectorizer_test.fit(X_test)

vector_test = vectorizer_test.transform(X_test)

print(vector_test.shape)

print(type(vector_test))

print(vector_test.toarray())

'''
'''

#X_scaled = preprocessing.scale(X)

print(vectorizer_train.vocabulary_)

print(vectorizer_test.vocabulary_)

'''
#first we will convert our object to vectors

#here we will train 4238 values and we will use the rest to test our set



#from sklearn import preprocessing

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Application of the svm

#from sklearn.svm import LinearSVC, SVC

#from sklearn.datasets import make_classification

#X_train, y_train = make_classification()

#clf = LinearSVC(multi_class = 'crammer_singer')

#clf.fit(X_train, y_train)

#print(clf.coef_)

#print(clf.intercept_)





'''

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

if addpoly:

all_data = pd.concat((X_train, X_test), ignore_index = True)

scaler = MinMaxScaler()

scaler.fit(all_data)

    all_data = scaler.transform(all_data)

    ploy = PolynomialFeatures(2)

    all_data = poly.fit_transform(all_data)

    

    X_train = all_data[:train_dataset.shape[0]]

    X_test = all_data[train_dataset,shape[0]:]

    

    ##

    

    print(X_train.shape)

    print(y_train.shape)

    print(X_test.shape)

    '''
#Now  we will countvectorizer to convert out object data into vector form

#from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.feature_extraction import DictVectorizer

#print(clf.predict([X_test] ))

#clf.predict(X_test)

#vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')
#vect.fit(X_train)

#vocab = vect.vocabulary_

#vect.fit(y_train)

#vocab = vect.vocabulary_
#def convert_X_to_X_word_ids(X):

    #return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )
#X_train_word_ids = convert_X_to_X_word_ids(X_train)

#X_test_word_ids  = convert_X_to_X_word_ids(X_test)
#X_train_padded_seqs = pad_sequences(X_train_word_ids, maxlen=150, value=0)

#X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=150, value=0)
#from sklearn import svm

#clf = svm.SVC()

#clf.fit(X, y)
#Actual network
