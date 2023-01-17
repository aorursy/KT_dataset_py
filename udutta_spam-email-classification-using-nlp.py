#Spam filteting

import nltk

print(nltk.__version__)
import numpy as np

import pandas as pd

import sys

import sklearn

print("numpy version: {}".format(np.__version__))

print("pandas version: {}".format(pd.__version__))

print("sklearn version: {}".format(sklearn.__version__))

print("sys version: {}".format(sys.version))

!wget  https://archive.ics.uci.edu/ml/datasets/sms+spam+collection



        
# I initially tried to use StringIO. StringIO has been deprecated in python3. instead we are using IO 

import requests, zipfile, io

r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip')



z = zipfile.ZipFile(io.BytesIO(r.content))
# gives the filelist in the zipfile object

z.filelist
# extracts the file whose name is passed as an argument to the extract method

z.extract('SMSSpamCollection')
# now lets go to the directory where our data file is extracted

!chdir /kaggle/working
df = pd.read_table('/kaggle/working/SMSSpamCollection',header=None)


#nowe lets read the file 

# df = pd.read_table('SMSSpamCollection',header= None, encoding = 'utf-8')

# If we do not pass headee r = None, pandas will read from the second line.

print(type(df))



df.info()

print('______')

print(df.head())
A=np.array([1,1,2,3,2,3,4,])

df9=pd.DataFrame(A)

df9[0].value_counts() # .value_counts() works on series and not on the dataframe
#Lets check out the count of ham and spam



classes = df[0] # Point to be noted: we can  not use class, as class is a reserved key word in python



classes.value_counts()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()



Y = encoder.fit_transform(classes)

print(Y[:10])

# Now we have converted the classes into binary argument. Lets check some of the labels
# we can check details of any method by the command 'method?'. example below is for LabelEncoder

#LabelEncoder?
text_msg = df[1]

text_msg[0:10]




# use regular expressions to replace email addresses, URLs, phone numbers, other numbers



# Replace email addresses with 'email'

processed = text_msg.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',

                                 'emailaddress')



# Replace URLs with 'webaddress'

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',

                                  'webaddress')



# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)

processed = processed.str.replace(r'£|\$', 'moneysymb')

    

# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'

processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',

                                  'phonenumbr')

    

# Replace numbers with 'numbr'

processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')



print(processed[8])





# Remove punctuation

processed = processed.str.replace(r'[^\w\d\s]', ' ')



# Replace whitespace between terms with a single space

processed = processed.str.replace(r'\s+', ' ')



# Remove leading and trailing whitespace

processed = processed.str.replace(r'^\s+|\s+?$', '')



processed[2]
processed = processed.str.lower()

print(processed)
# Remove the stopwords

from nltk.corpus import stopwords

# print(stopwords)

stopwords = set(stopwords.words('english'))



print(stopwords, end= ' ')

print(type(stopwords))
from nltk.tokenize import sent_tokenize,word_tokenize
def filter(sent):

    filtered = dict()

    for i in processed:

        words = word_tokenize(i)



        for word in words:

            rmv=[]

            if word not in stopwords:

                rmv.append(word)

                filtered[i]= rmv

                rmv=[]

    return filtered

    
# filter(processed[1])
processed_rm_stopwords = list(map(lambda x: ' '.join(term for term in x.split() if term not in stopwords),processed))

processed_rm_s= processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stopwords))
print(processed_rm_stopwords[0:5])

print(processed_rm_s[0:5])

print(f"processed_rm_s: {type(processed_rm_s)}")

print(f"processed_rm_stopwords{type(processed_rm_stopwords)}")

df_new=pd.DataFrame(processed_rm_stopwords)

print(type(df_new))

print(df_new[0:5])

ps = nltk.PorterStemmer()

stemmed = list(map(lambda x: " ".join(ps.stem(term) for term in x.split()), processed_rm_stopwords))

print(stemmed[0:5])

df_new_stemmed = pd.DataFrame(stemmed)

print(df_new_stemmed)
# Creating a bag of words

all_words =[] # this will have all the words in our dataframe (processed)



for item in stemmed:

    

    words  = word_tokenize(item)

    for word in words:

        all_words.append(word)

    

all_words = nltk.FreqDist(all_words) # unique words

len(all_words)
# Lets find out some common words

print("most common words {}".format(all_words.most_common(10)))


all_words_common = all_words.most_common(1500)

# all_words.keys()

features = all_words_common[0:1500]

features_list=[ key for  key, val in features]

print(features_list, end = ' ')
# lets define feature finding function

def find_feature(message):

    

    words = word_tokenize(message)

    

    feature= dict()

    

    for word in features_list:

        

        feature[word]=word in words

    return feature

        

features = find_feature(stemmed[0])

# print(features)

# print(len(features))





for key, val in features.items():

    if val == True:

        print (key)

    
print(stemmed[0])
messages = list(zip(stemmed,Y))

# list(messages)
df_N = pd.DataFrame(messages)

df_N.head

print(type(df_N))
# [key, label for (key, label) in df_N[0:5]]
#define a seed 



seed = 1

np.random.seed =seed

np.random.shuffle(messages)

# print(df_N[0:5])
featureset = [(find_feature(text),label) for (text, label) in messages]
print(type(featureset))

print(len(featureset))
from sklearn.model_selection import train_test_split
training, testing = train_test_split(featureset, test_size=0.2, random_state = seed)
print(len(training))

print(type(training))
print("length of training set: {}".format(len(training)))

print("length of testing set: {}".format(len(testing)))
training[0]
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
names = ["K nearest neighbors","decision tree","random forest","SVC","Logistics regression","SGD","MultinomialNB"]

classifiers = [KNeighborsClassifier(),

               DecisionTreeClassifier(),

               RandomForestClassifier(),

               SVC(kernel = 'linear'),

               LogisticRegression(),

              SGDClassifier(max_iter=100),

               MultinomialNB()]

models = zip(names,classifiers)

list(models)





for (name, model) in models:

    nltk_model = SklearnClassifier(model)

    nltk_model.train(training)

    accuracy = nltk.classify.accuracy(nltk_model, testing)*100

    print("{} Accuracy: {}".format(name, accuracy))
# Define models to train

from nltk.classify.scikitlearn import SklearnClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",

         "Naive Bayes", "SVM Linear"]



classifiers = [

    KNeighborsClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    LogisticRegression(),

    SGDClassifier(max_iter = 100),

    MultinomialNB(),

    SVC(kernel = 'linear')

]

models = zip(names, classifiers)

from nltk.classify.scikitlearn import SklearnClassifier

for name, model in models:

    nltk_model = SklearnClassifier(model)

    nltk_model.train(training)

    accuracy = nltk.classify.accuracy(nltk_model, testing)*100

    print("{} Accuracy: {}".format(name, accuracy))
nl=SklearnClassifier(RandomForestClassifier())

nl.train(training)
accuracy_nl = nltk.classify.accuracy(nl, testing)*100

print("{} Accuracy: {}".format('Randomforest', accuracy_nl))
models = zip(names, classifiers)

# list(models)

# The zip object is very interesting. Once you execute the zip object and then you try to execute it again

#we will get an error, if you try doing that you will find out why.
classifiers
# estimators= list(models)

# # print(estimators)
from sklearn.ensemble import VotingClassifier

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = list(models), voting = 'hard', n_jobs = -1))

nltk_ensemble.train(training)

# accuracy = nltk.classifiy.accuracy(nltk_ensemble, test)*100

# print(f"the accuracy of the ensemble model is: {accuracy}")
nltk_ensemble
accuracy = nltk.classify.accuracy(nltk_ensemble, testing)*100

print(f"the accuracy of the ensemble model is: {accuracy}")
print(testing[0]) # as you can see testing is a tuple if feature_list and the label for the message
text_feature, label = zip(*testing)
predictions = nltk_ensemble.classify_many(text_feature)
print(classification_report(label, predictions))



# pd.DataFrame(confusion_matrix(label, predictions), index=[['actual','actual'],['ham','spam']],

#             columns = [['predicted','predicted'],['ham','spam']])
pd.DataFrame(confusion_matrix(label, predictions), index=[['actual','actual'],['ham','spam']],

            columns = [['predicted','predicted'],['ham','spam']])