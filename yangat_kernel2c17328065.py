!pip install wordcloud
!pip install imblearn
import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

sns.set()

import nltk

import re

import warnings

warnings.filterwarnings('ignore')

# plotting

from wordcloud import WordCloud



# nltk

from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem import SnowballStemmer



# sklearn(classifier)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import LinearSVC

from sklearn.svm import SVC







# Metrics/Evaluation



from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE 

from sklearn.pipeline import Pipeline

from sklearn.utils import resample

from collections import Counter 

from sklearn import metrics
train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')

test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
# First 5 rows of the train dataset.



train.head()
# First 5 rows of the test data.



test.head()
train.info()
# This code looks at how many words there are in each tweet message.



train['token_length'] = [len(x.split(" ")) for x in train.message]

train 
# Finding the maximum token_length.



max(train.token_length)
# Function that finds punctuation marks.



def find_punct(string):

    

    """  

    This function takes in a string and finds punctuation marks. Thereafter, 

    it returns a list of the string of punctuatin marks  

    """

    

    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', string)

    string="".join(line)

    return list(string)
# New Features with punctuation and punctuation length.



train['message']=train['message'].str.lower()   # change message text to lowercase

train['message_punct']=train['message'].apply(lambda x:find_punct(x))

train['message_punct_len']=train['message'].apply(lambda x:len(find_punct(x)))



train
train.sentiment.value_counts()
# Message Distribution over the classes.



dist_class = train['sentiment'].value_counts()

labels = ['1', '2','0','-1']

sns.color_palette('hls')



# Bar graph plot.



sns.barplot(x=dist_class.index, y=dist_class, data = train).set_title("Tweet message distribution over the sentiments")

plt.ylabel('Count')

plt.xlabel('Sentiment')

plt.show()

plt.savefig('Tweet message distribution over the sentiments.png')
# Pie chart plot.



colors = ['green', 'red', 'orange', 'blue']

plt.pie(dist_class,

        labels=labels,

        colors = colors,

        counterclock=False,

        startangle=90,

        autopct='%1.1f%%',

        pctdistance=0.7)

plt.title("Tweet message distribution over the sentiments")
# Convert message characters to lowercase.



train['target']=train['message'].str.lower()
# Find URL's.



def find_link(string):

    

    """ 

    This function takes in a string and returns a joint string with 

    values which are part of url's.  

    """

    

    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)

    return "".join(url) 

# Creating a new df which shows url count of the respective sentiments.



train['target_url']=train['target'].apply(lambda x: find_link(x))

df=pd.DataFrame(train.loc[train['target_url']!=""]['sentiment'].value_counts()).reset_index()

df.rename(columns={"index": "sentiment", "sentiment": "url_count"})

# Plotting word clouds.



news = train[train['sentiment'] == 2]['message']

pro = train[train['sentiment'] == 1]['message']

neutral =train[train['sentiment'] == 0]['message']

Anti = train[train['sentiment'] ==-1]['message']





news = [word for line in news for word in line.split()]

pro = [word for line in pro for word in line.split()]

neutral = [word for line in neutral for word in line.split()]

Anti= [word for line in Anti for word in line.split()]



news = WordCloud(

    background_color='white',

    max_words=50,

    max_font_size=100,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(news))



pro = WordCloud(

    background_color='white',

    max_words=50,

    max_font_size=100,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(pro))







neutral = WordCloud(

    background_color='white',

    max_words=50,

    max_font_size=100,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(neutral))





Anti = WordCloud(

    background_color='white',

    max_words=50,

    max_font_size=100,

    scale=5,

    random_state=1,

    collocations=False,

    normalize_plurals=False

).generate(' '.join(Anti))





fig, axs = plt.subplots(2, 2, figsize = (20, 12))



# fig.suptitle('Clouds of polar words', fontsize = 30)



fig.tight_layout(pad = 0)



axs[0, 0].imshow(news)

axs[0, 0].set_title('Words from news tweets', fontsize = 20)

axs[0, 0].axis('off')



# axs[0, 0].tight_layout(pad = 1)



axs[0, 1].imshow(pro)

axs[0, 1].set_title('Words from pro tweets', fontsize = 20)

axs[0, 1].axis('off')



# axs[0, 1].tight_layout(pad = 1)





# axs[1, 0].tight_layout(pad = 1)



axs[1, 0].imshow(Anti)

axs[1, 0].set_title('Words from anti tweets', fontsize = 20)

axs[1, 0].axis('off')



axs[1, 1].imshow(neutral)

axs[1, 1].set_title('Words from neutral tweets', fontsize = 20)

axs[1, 1].axis('off')



# axs[1, 0].tight_layout(pad = 1)



plt.savefig('joint_cloud.png')
# Counting frequently used words. 



word_list = [word for line in train['message']  for word in line.split()]

sns.set(style="darkgrid")

counts = Counter(word_list).most_common(20)

counts_df = pd.DataFrame(counts)

counts_df

counts_df.columns = ['word', 'frequency']



# Visualising on a barplot.



fig, ax = plt.subplots(figsize = (9, 9))

ax = sns.barplot(y="word", x='frequency', ax = ax, data=counts_df, palette="hls")

plt.savefig('wordcount_bar.png')
# Checking for Nulls in the train dataframe.



train.isnull().sum()
# Checking Nulls in the test dataframe.



test.isnull().sum()
# Checking for blanks in the train dataframe.



blanks = []                          # start with an empty list

for i,mes,twe in test.itertuples():  # iterate over the DataFrame

    if type(mes)==str:               # avoid NaN values

        if mes.isspace():            # test 'review' for whitespace

            blanks.append(i)         # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
# Replacing the website url's with the word 'url'.



pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

train['message'] = train['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)

test['message'] = test['message'] .replace(to_replace = pattern_url, value = subs_url, regex = True)
nltk.download('stopwords')

nltk.download('wordnet')
stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    

    """ 

    Function that takesin text and cleans it by removing stop words

    and lemmatizing it. 

    """

    text = re.sub('<[^<]+?>','', text)

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



train['Processed_message'] = train.message.apply(lambda x: clean_text(x))



test['Processed_message'] = test.message.apply(lambda x: clean_text(x))

# Checking the preprocessed train dataframe.



train.head()

# Checking the preprocessed test dataframe.



test.head()
# Independent feature of the train dataframe.



X=train['Processed_message']



# Dependent feature of the train dataframe.



y=train['sentiment'] 



# Independent feature of test dataframe.



x_unseen=test['Processed_message']   # test independent feature
# Splitting the train dataset.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
vectoriser = TfidfVectorizer(stop_words='english', 

                             min_df=1, 

                             max_df=0.9, 

                             ngram_range=(1, 2))
vectoriser.fit(x_unseen)
# Fitting the vectoriser.



vectoriser.fit(X_train, y_train) 
# Transformation of the datasets.



X_train = vectoriser.transform(X_train)

X_test  = vectoriser.transform(X_test)

x_unseen = vectoriser.transform(x_unseen)
from sklearn.linear_model import LogisticRegression



LogisticRegression = LogisticRegression()



# Fitting the model with train dataset.



LogisticRegression = LogisticRegression.fit(X_train, y_train)
# Getting predicions from the X_test



pred = LogisticRegression.predict(X_test)



# Printing the classification report



print(metrics.classification_report(y_test,pred))



# Print the overall accuracy



print(metrics.accuracy_score(y_test,pred))
from sklearn.svm import LinearSVC

LinearSVC = LinearSVC()



# Fitting the model with train dataset.



LinearSVC = LinearSVC.fit(X_train, y_train)
# Getting predicions from the X_test.



pred1 = LinearSVC.predict(X_test)



# Printing the classification report.



print(metrics.classification_report(y_test, pred1))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,pred1))
from sklearn.svm import SVC



SVC = SVC()



# Fitting the model with train dataset.



SVC = SVC.fit(X_train, y_train)
# Getting predictions from the X_test.



pred2 = SVC.predict(X_test)



# Pritting the classification report.



print(metrics.classification_report(y_test,pred2))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,pred2))


MultinomialNB = MultinomialNB()



# Fitting the model with train dataset.



MultinomialNB  = MultinomialNB .fit(X_train, y_train)
# Getting predictions from the X_test.



pred3 = MultinomialNB.predict(X_test)



# Printing the classification report.



print(metrics.classification_report(y_test,pred3))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,pred3))
X_train.shape, y_train.shape
smote = SMOTE('minority')



X_sm, y_sm = smote.fit_sample(X_train, y_train)

print(X_sm.shape, y_sm.shape)
# Separate minority and majority classes.



df_majority= train[(train.sentiment==1) |

                  (train.sentiment ==2) |

                   (train.sentiment==0)]

df_minority = train[train.sentiment == -1]

                    



# Upscalling the minority class.



df_minority_upsampled= resample(df_minority,replace= True,

                            n_samples= 4000, random_state =42)  # sample with replacement



# Combine majority class with upscalled minority class.



df_upsampled = pd.concat ([df_majority,

                          df_minority_upsampled])

# Display new class counts.



df_upsampled.sentiment.value_counts()
# Message distribution over the classes.



dist_class = df_upsampled['sentiment'].value_counts()

labels = ['1', '2','0','-1']



fig, (ax1 )= plt.subplots(1, figsize=(12,6))



sns.barplot(x=dist_class.index, y=dist_class, ax=ax1).set_title("Tweet message distribution over the sentiments")
from sklearn.linear_model import LogisticRegression

LogisticRegression = LogisticRegression()



# Fitting the model with train dataset.



LogisticRegression_up = LogisticRegression.fit(X_sm, y_sm)
# Getting predicions from the X_test.





predict = LogisticRegression_up.predict(X_test)



# Printing the classification report.



print(metrics.classification_report(y_test, predict))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,predict))
from sklearn.svm import LinearSVC



LinearSVC = LinearSVC() 



# Fitting the model with train dataset.



LinearSVC_up = LinearSVC.fit(X_sm, y_sm)
# Getting predicions from the X_test.



predict1 = LinearSVC_up.predict(X_test)

 

# Printing the classification report.



print(metrics.classification_report(y_test, predict1))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,predict1))
from sklearn.svm import SVC



SVC = SVC()



# Fitting the model with train dataset.



SVC_up = SVC.fit(X_sm, y_sm)
# Getting predictions from the X_test.



predict2 = SVC_up.predict(X_test)



# Pritting the classification report.



print(metrics.classification_report(y_test,predict2))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,predict2))
from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

MultinomialNB = MultinomialNB()



# Fitting the model with train dataset.



MultinomialNB_up = MultinomialNB.fit(X_sm, y_sm)
# Getting predictions from the X_test.



predict3 = MultinomialNB_up.predict(X_test)



# Printing the classification report.



print(metrics.classification_report(y_test,predict3))



# Print the overall accuracy.



print(metrics.accuracy_score(y_test,predict3))
# Getting the predicted sentimet from test dataset.



y_pred = LinearSVC_up.predict(x_unseen)

y_pred1 = SVC_up.predict(x_unseen)

y_pred2 = LogisticRegression_up.predict(x_unseen)

y_pred3 = MultinomialNB_up.predict(x_unseen)

# Printing the predicted sentiment.



print(y_pred)

print(y_pred1)

print(y_pred2)

print(y_pred3)



# Making the tweetid to be the index.



test=test.set_index('tweetid')
test.head(5)
# Selecting the index of the test dataframe.



final_test= test.index
# Creating the submission Dataframe.



Final_Table = {'tweetid': final_test, 'sentiment':np.round(y_pred, 0)}

submission = pd.DataFrame(data=Final_Table)

submission = submission[['tweetid', 'sentiment']]
submission.set_index('tweetid').head(5)
# Creating the submission Dataframe.



Final_Table1 = {'tweetid': final_test, 'sentiment':np.round(y_pred1, 0)}

submission1 = pd.DataFrame(data=Final_Table1)

submission1 = submission1[['tweetid', 'sentiment']]
submission1.set_index('tweetid').head(5)
# Creating the submission Dataframe.



Final_Table2 = {'tweetid': final_test, 'sentiment':np.round(y_pred2, 0)}

submission2 = pd.DataFrame(data=Final_Table2)

submission2 = submission2[['tweetid', 'sentiment']]
submission2.set_index('tweetid').head(5)
# Creating the submission Dataframe.



Final_Table3 = {'tweetid': final_test, 'sentiment':np.round(y_pred3, 0)}

submission3 = pd.DataFrame(data=Final_Table3)

submission3 = submission3[['tweetid', 'sentiment']]
submission3.set_index('tweetid').head(5)
submission.to_csv("TestSubmission1.csv",index  = False)   # writing csv file
submission1.to_csv("TestSubmission2.csv",index  = False)  # writing csv file
submission2.to_csv("TestSubmission3.csv",index  = False)  # writing csv file
submission3.to_csv("TestSubmission4.csv",index  = False)  # writing csv file
#import pickle

#file = open('vectoriser.pkl','wb')

#pickle.dump(vectoriser, file)

#file.close()



#file = open('LogisticRegression.pkl','wb')

#pickle.dump(LogisticRegression, file)

#file.close()



#file = open('SVC.pkl','wb')

#pickle.dump(SVC, file)

#file.close()



#file = open('MultinomialNB.pkl','wb')

#pickle.dump(MultinomialNB, file)

#file.close()



#file = open('LinearSVC.pkl','wb')

#pickle.dump(LinearSVC, file)

#file.close()
