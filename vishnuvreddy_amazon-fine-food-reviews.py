import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


#3. Read sqlite as panda dataframe with sql query
con = sqlite3.connect("../input/database.sqlite")
#filtering only positive and negative reviews i.e. 
# not taking into consideration those reviews with Score=3

filter_data = pd.read_sql_query('SELECT * FROM REVIEWS WHERE Score != 3',con)

# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.
def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filter_data['Score']
positiveNegative = actualScore.map(partition) 
filter_data['Score'] = positiveNegative

print(filter_data.shape)
filter_data.head(3)
display = pd.read_sql_query('''
SELECT *
FROM Reviews
WHERE UserId = "A395BORC6FGVXV"''',con)
display
sorted_data = filter_data.sort_values('ProductId',axis = 0,ascending = True, inplace=False, kind = 'quicksort',na_position = 'last')
#Deduplication of entries
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape

(final['Id'].size*1.0)/(filter_data['Id'].size*1.0)*100
#data with HelpfulnessNumerator > HelpfulnessDenominator

display = pd.read_sql_query('''
SELECT *
FROM Reviews
WHERE HelpfulnessNumerator > HelpfulnessDenominator and score !=3''',con)
display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

#Before starting the next phase of preprocessing lets see the number of entries left
print(final.shape)

#How many positive and negative reviews are present in our dataset?
final['Score'].value_counts()
# find sentences containing HTML tags
#import regular expressions
import re
i=0;
for sent in final['Text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1; 
import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

print('Some stop words are:')
print(stop)

#Code for implementing step-by-step the checks for data pre-processing
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    final_string.append(str1)
    i+=1
final['CleanedText'] = final_string #adding a column of CleanedText which displays the data after pre-processing of the review
final.head(3)
#define style for seaborn
def style():
    plt.subplots(figsize=(15,6))
    sns.set_style("whitegrid")
    sns.despine()
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.color_palette('dark')
sns.countplot(data=final,x='Score')
#length of text of each review
def length(x):
    return len(x)
leng=final['Text'].map(length)
leng=list(leng)
#pdf of reviews
style()
sns.distplot(leng,hist=False).set_title('pdf for length of reviews')
#convert time stamp to date and column as data
final['date'] = pd.to_datetime(final['Time'], unit='s',)
date = final['date']
style()
a = pd.to_datetime(final['Time'], unit='s')
years=a.map(lambda x:x.year)
years=years.reset_index()
years.columns=['count','year']
sns.countplot(data=years,x='year').set_title('Reviews given by users each year')

style()
a = pd.to_datetime(final['Time'], unit='s')
months=a.map(lambda x:x.month)
months=months.reset_index()
months.columns =['count','month']
months
sns.countplot(data=months,x='month').set_title('Reviews given by users each month')
#Build wordcloud
from wordcloud import WordCloud,STOPWORDS

#get all words
words = str(final['Text'].values)

# Generate a word cloud image
wordcloud = WordCloud(width=1000,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Spectral').generate(words)

# Display the generated image:
# the matplotlib way:
plt.figure(figsize=(10,15))
plt.title('Word cloud for the words in reviews')
plt.imshow(wordcloud,cmap='gist_rainbow')
plt.axis('off');

#top users
top= final.groupby(by='ProfileName')['Id'].count().reset_index()
top=top.sort_values('Id',ascending=False)[:15]
style()
sns.barplot(data=top,x='Id',y='ProfileName',palette='bright').set_title('Users given more reviews')
#top positive rated product
positive = final[final['Score']=='positive']
top= positive.groupby(by='ProductId')['Id'].count().reset_index()
top=top.sort_values('Id',ascending=False)[:10]
top.columns=['ProductId','positive_count']
top

from IPython.display import Image
from IPython.core.display import HTML
print("Top Positive product from reviews")
Image(url= "https://images-na.ssl-images-amazon.com/images/I/81c-m39Qt0L._SX522SX522_SY450_CR,0,0,522,450_PIbundle-12,TopRight,0,0_SX522_SY450_CR,0,0,522,450_SH20_.jpg")
#Top negative rated products
negative = final[final['Score']=='negative']
top= negative.groupby(by='ProductId')['Id'].count().reset_index()
top=top.sort_values('Id',ascending=False)[:10]
top.columns=['ProductId','positive_count']
top

from IPython.display import Image
from IPython.core.display import HTML
print("Top negative product from reviews")
Image(url= "https://images-na.ssl-images-amazon.com/images/I/81N3dYbSCWL._SL1500_.jpg")
#We take a sample of 100000 data points
from sklearn.utils import resample
sample = resample(final,n_samples=100000)
# sort our sampled data based on time in ascending order
sample = sample.sort_values(by='Time',kind='quicksort')

#splitting 64% as train 16% as cv and 25% as test data
a = int(sample.shape[0] * 0.80)
b = int(a * 0.8)

train = sample.iloc[:b,:]
cv = sample.iloc[b:a]
test = sample.iloc[a:,:]

# print train,cv and test size
print('train size is:',train.shape)
print('cv size is:',cv.shape)
print('test size is:',test.shape)
#convert positive label as 1 and negative as 0
def convert(x):
    if x == 'positive':
        return 1
    else:
        return 0
#train y labels    
y_train = train['Score'].map(convert)
y_cv = cv['Score'].map(convert)
y_test = test['Score'].map(convert)

#Convert text to Bag of words for each word in the corpus.
bow = CountVectorizer()
bow_counts = bow.fit_transform(train['CleanedText'].values)
bow_counts.shape
#don't forgot to normalize train data
X_bow = normalize(bow_counts,axis=0)
#convert cv text to Bow vectors
X_cv_bow = bow.transform(cv['CleanedText'].values)
#don't forgot to normalize cv data
X_cv_bow = normalize(X_cv_bow,axis=0)
#convert test text to Bow vectors
X_test_bow = bow.transform(test['CleanedText'].values)
#don't forgot to normalize cv data
X_test_bow = normalize(X_test_bow,axis=0)

#Convert text to tfidf for each word in the corpus.
#We consider only uni-grams as computing time and space is less.
tfidf_vect = TfidfVectorizer(ngram_range=(1,1))
final_tfidf = tfidf_vect.fit_transform(train['CleanedText'].values)
#print top tfidf words.
#High tfidf value implies word is more important compared to less tfidf value word.
features = tfidf_vect.get_feature_names()
len(features)

# source: https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=15):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

#don't forgot to normalize train data
X_tfidf = normalize(final_tfidf,axis=0)
#convert cv text to tfidf vectors
X_cv_tfidf = tfidf_vect.transform(cv['CleanedText'].values)
#don't forgot to normalize cv data
X_cv_tfidf = normalize(X_cv_tfidf,axis=0)
#convert test text to tfidf vectors
X_test_tfidf = tfidf_vect.transform(test['CleanedText'].values)
#don't forgot to normalize cv data
X_test_tfidf = normalize(X_test_tfidf,axis=0)

#function to give various scores on test data for a given model.
def cal_metrics(predicted,actual):
  conf = confusion_matrix(predicted,actual)
  TN,FN,FP,TP = conf[0][0],conf[0][1],conf[1][0],conf[1][1]
  P = TP+FN
  N = TN+FP
  TPR = TP/P
  FPR = FP/P
  FNR = FN/N
  TNR = TN/N
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  f1_score =  2 * (precision * recall) / (precision + recall)
  acc = accuracy_score(predicted,actual)
  print('Various metrics of model:')
  print('TPR is:',TPR)
  print('FPR is:',FPR)
  print('TNR is:',TNR)
  print('FPR is:',FPR)
  print('Precision is:',precision*100)
  print('Recall is:',recall*100)
  
  print('F1 score is:',f1_score*100)
  print('Accuracy is:',acc*100,'\n')
  
  print('Confusion Matrix is:')
  ax = sns.heatmap(confusion_matrix(predicted,actual),annot=True,fmt='g',cbar=None)
  plt.show()

#function to print top 10 positive and negative features
def imp_features(vectorizer,clf):
  feature_names = vectorizer.get_feature_names()
  n = clf.coef_[0].shape[0]
  coefs_with_features = sorted(zip(clf.coef_[0], feature_names))
  positive = coefs_with_features[:n-11:-1]
  negative = coefs_with_features[:10]
  positive = [i[1] for i in positive]
  negative = [i[1] for i in negative]

  top = {'positive':positive,'negative':negative}
  print('Top positive and negative features/words')
  top = pd.DataFrame(data=top)
  return top
cv_error = []
my_alpha = [10**x for x in range(-6,4)]

for i in my_alpha:
    clf = MultinomialNB(alpha=i,fit_prior=None)
    clf.fit(X_bow,y_train)
    score = clf.score(X_cv_bow,y_cv)
    print('For alpha = %f, cv score is : %f' %(i,score))
    cv_error.append(1-score)
    
#plot errors and cv scores
plt.figure(figsize=(10,25))
fig, ax = plt.subplots()
ax.plot(my_alpha, cv_error,c='g')
for i, txt in enumerate(np.round(cv_error,3)):
    ax.annotate((my_alpha[i],str(txt)), (my_alpha[i],cv_error[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

# changing to misclassification error
MSE = [x for x in cv_error]
# determining best alpha
optimal_alpha = my_alpha[MSE.index(min(MSE))]
print('\nThe optimal alpha is %f.' % optimal_alpha)

#train model with alpha.
clf = MultinomialNB(alpha=optimal_alpha,fit_prior=None)
clf.fit(X_bow,y_train)
#predict y labels with model on test data
predict = clf.predict(X_test_bow)
#Performance of the model on the test data
cal_metrics(predict,y_test)

#Lets get some of the important features/words for both positive and negative classes
imp_features(bow,clf)
#Visualize most positive and negative words on cloudcloud

#get all words

top = imp_features(bow,clf)
pos = top['positive']
neg = top['negative']

words = str(pos.values)
words1 = str(neg.values)

# Generate a word cloud image
wordcloud = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Greens').generate(words)
wordcloud1 = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Reds').generate(words1)

# Display the generated image:
# the matplotlib way:
print('Top positive and negative features/words of Naive Bayes classifier using Bag Of Words')

plt.figure(figsize=(15,20))
plt.subplot(121)
plt.axis('off')
plt.title("Important positve words")
plt.imshow(wordcloud);
plt.subplot(122)
plt.axis('off')
plt.title("Important negative words")
plt.imshow(wordcloud1);
cv_error = []
my_alpha = [10**x for x in range(-6,4)]

for i in my_alpha:
    clf = MultinomialNB(alpha=i,fit_prior=None)
    clf.fit(X_tfidf,y_train)
    score = clf.score(X_cv_tfidf,y_cv)
    print('For alpha = %f, cv score is : %f' %(i,score))
    cv_error.append(1-score)
    
#plot errors and cv scores
fig, ax = plt.subplots()
ax.plot(my_alpha, cv_error,c='g')
for i, txt in enumerate(np.round(cv_error,3)):
    ax.annotate((my_alpha[i],str(txt)), (my_alpha[i],cv_error[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()

# changing to misclassification error
MSE = [x for x in cv_error]
# determining best alpha
optimal_alpha = my_alpha[MSE.index(min(MSE))]
print('\nThe optimal alpha is %f.' % optimal_alpha)

#train model with alpha.
clf = MultinomialNB(alpha=optimal_alpha,fit_prior=None)
clf.fit(X_tfidf,y_train)
#predict y labels with model on test data
predict = clf.predict(X_test_tfidf)
#Performance of the model on the test data
cal_metrics(predict,y_test)

#Lets get some of the important features/words for both positive and negative classes
imp_features(tfidf_vect,clf)
#Visualize most positive and negative words on cloudcloud

#get all words

top = imp_features(tfidf_vect,clf)
pos = top['positive']
neg = top['negative']

words = str(pos.values)
words1 = str(neg.values)

# Generate a word cloud image
wordcloud = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Greens').generate(words)
wordcloud1 = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Reds').generate(words1)

# Display the generated image:
# the matplotlib way:
print('Top positive and negative features/words of Naive Bayes classifier using Tf-idf')
plt.figure(figsize=(15,20))
plt.subplot(121)
plt.axis('off')
plt.title("Important positve words")
plt.imshow(wordcloud);
plt.subplot(122)
plt.axis('off')
plt.title("Important negative words")
plt.imshow(wordcloud1);
cv_error = []
my_c = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
for i in my_c:
    clf = LogisticRegression(C=i,class_weight='balanced')
    clf.fit(X_bow,y_train)
    score = clf.score(X_cv_bow,y_cv)
    print('For C = %f, cv score is : %f' %(i,score))
    cv_error.append(1-score)
    
#plot errors and cv scores
fig, ax = plt.subplots()
ax.plot(my_c, cv_error,c='g')
for i, txt in enumerate(np.round(cv_error,3)):
    ax.annotate((my_c[i],str(txt)), (my_c[i],cv_error[i]))
plt.grid()
plt.title("Cross Validation Error for each C")
plt.xlabel("C 's")
plt.ylabel("Error measure")
plt.show()

# changing to misclassification error
MSE = [x for x in cv_error]
# determining best C
optimal_c = my_c[MSE.index(min(MSE))]
print('\nThe optimal C is %f.' % optimal_c)
    
#train model with C.
clf = LogisticRegression(C=optimal_c,class_weight='balanced')
clf.fit(X_bow,y_train)
#predict y labels with model on test data
predict = clf.predict(X_test_bow)
#Performance of the model on the test data
cal_metrics(predict,y_test)

#Lets get some of the important features/words for both positive and negative classes
imp_features(bow,clf)
#Visualize most positive and negative words on cloudcloud

#get all words

top = imp_features(bow,clf)
pos = top['positive']
neg = top['negative']

words = str(pos.values)
words1 = str(neg.values)

# Generate a word cloud image
wordcloud = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Greens').generate(words)
wordcloud1 = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Reds').generate(words1)

# Display the generated image:
# the matplotlib way:
print('Top positive and negative features/words of Logistic Regression classifier using Bag Of Words')
plt.figure(figsize=(15,20))
plt.subplot(121)
plt.axis('off')
plt.title("Important positve words")
plt.imshow(wordcloud);
plt.subplot(122)
plt.axis('off')
plt.title("Important negative words")
plt.imshow(wordcloud1);
cv_error = []
my_c = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]
for i in my_c:
    clf = LogisticRegression(C=i,class_weight='balanced')
    clf.fit(X_tfidf,y_train)
    score = clf.score(X_cv_tfidf,y_cv)
    print('For C = %f, cv score is : %f' %(i,score))
    cv_error.append(1-score)
    
#plot errors and cv scores
fig, ax = plt.subplots()
ax.plot(my_c, cv_error,c='g')
for i, txt in enumerate(np.round(cv_error,3)):
    ax.annotate((my_c[i],str(txt)), (my_c[i],cv_error[i]))
plt.grid()
plt.title("Cross Validation Error for each C")
plt.xlabel("C 's")
plt.ylabel("Error measure")
plt.show()

# changing to misclassification error
MSE = [x for x in cv_error]
# determining best C
optimal_c = my_c[MSE.index(min(MSE))]
print('\nThe optimal C is %f.' % optimal_c)
    
#train model with C.
clf = LogisticRegression(C=optimal_c,class_weight='balanced')
clf.fit(X_tfidf,y_train)
#predict y labels with model on test data
predict = clf.predict(X_test_tfidf)
#Performance of the model on the test data
cal_metrics(predict,y_test)

#Lets get some of the important features/words for both positive and negative classes
imp_features(tfidf_vect,clf)
#Visualize most positive and negative words on cloudcloud

#get all words

top = imp_features(tfidf_vect,clf)
pos = top['positive']
neg = top['negative']

words = str(pos.values)
words1 = str(neg.values)

# Generate a word cloud image
wordcloud = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Greens').generate(words)
wordcloud1 = WordCloud(width=1200,height=600,max_font_size=95,max_words=6000,background_color='black',colormap='Reds').generate(words1)

# Display the generated image:
# the matplotlib way:
print('Top positive and negative features/words of Logistic Regression classifier using Tf-idf')
plt.figure(figsize=(15,20))
plt.subplot(121)
plt.axis('off')
plt.title("Important positve words")
plt.imshow(wordcloud);
plt.subplot(122)
plt.axis('off')
plt.title("Important negative words")
plt.imshow(wordcloud1);