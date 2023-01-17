import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
tweetsdf = pd.read_table('../input/tweets-of-trump-and-trudeau/tweets.csv', sep=',', names=('ID', 'Author', 'tweet'))

tweetsdf=tweetsdf.iloc[1:]

tweetsdf.head()
y=tweetsdf['Author']

x=tweetsdf['tweet']

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)

print(x_train)
tvec= TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=0.05)

t_train=tvec.fit_transform(x_train)

t_test=tvec.fit_transform(x_test)
cvec = CountVectorizer(stop_words="english",ngram_range=(1,2), max_df=0.9, min_df=0.05)

c_train=cvec.fit_transform(x_train)

c_test=cvec.fit_transform(x_test)
svclassifier = SVC(kernel='rbf')

svclassifier.fit(t_train, y_train)

t_predsvc = svclassifier.predict(t_test)
svclassifier = SVC(kernel='rbf')

svclassifier.fit(c_train, y_train)

c_predsvc = svclassifier.predict(c_test)
countsvcacc = accuracy_score(c_predsvc,y_test)

print(confusion_matrix(y_test,c_predsvc))

print(classification_report(y_test,c_predsvc))



tfidfsvmacc = accuracy_score(t_predsvc,y_test)

print(confusion_matrix(y_test,t_predsvc))

print(classification_report(y_test,t_predsvc))
logclassifier=LogisticRegression(random_state=0, solver='lbfgs') 

logclassifier.fit(t_train, y_train) 

t_predlog = logclassifier.predict(t_test)
logclassifier=LogisticRegression(random_state=0, solver='lbfgs')

logclassifier.fit(c_train, y_train)

c_predlog = logclassifier.predict(c_test)
countlogacc = accuracy_score(c_predlog,y_test)

print(confusion_matrix(y_test,c_predlog))

print(classification_report(y_test,c_predlog))



countlogacc = accuracy_score(t_predlog,y_test)

print(confusion_matrix(y_test,t_predlog))

print(classification_report(y_test,t_predlog))
tlog_confmatrix = confusion_matrix(t_predlog,y_test)

clog_confmatrix = confusion_matrix(c_predlog,y_test)



tsvc_confmatrix = confusion_matrix(t_predsvc,y_test)

csvc_confmatrix = confusion_matrix(c_predsvc,y_test)

print(tlog_confmatrix)

print(clog_confmatrix)

print(tsvc_confmatrix)

print(csvc_confmatrix)