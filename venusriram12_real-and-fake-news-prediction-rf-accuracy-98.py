import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
true=pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake=pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
len(fake),len(true)
true["category"]=1
fake["category"]=0
data=pd.concat([true,fake])
data["category"].value_counts()
data.head()
data.isnull().sum()
data.info()
plt.figure(figsize =(15,10))
sb.countplot(data['subject'])

data['fulltext'] = data.title + ' ' + data.text
data.drop(['title','text'], axis=1, inplace=True)
final = data[['fulltext', 'category']]
final = data.reset_index()
final.drop(['index'], axis=1, inplace=True)

import re
i=0;
for sent in final['fulltext'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break;
    i += 1; 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop = set(stopwords.words('english')) 
sno = nltk.stem.SnowballStemmer('english')

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#.|,|)|(|\|/]',r'',sentence)
    
    return  cleaned
print(stop)

#Code for implementing step-by-step the checks mentioned in the pre-processing phase
# this code takes a while to run as it needs to run on 500k sentences.
import re
i=0
str1=' '
final_string=[]
all_true_words=[] # store words from +ve reviews here
all_fake_words=[] # store words from -ve reviews here.
s=''
for sent in final['fulltext'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['category'].values)[i] == '1': 
                        all_true_words.append(s) #list of all words used to describe positive reviews
                    if(final['category'].values)[i] == '0':
                        all_fake_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1
final['CleanedText']=final_string
final.head(3)
label=final["category"]
sample=final['CleanedText']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sample, label, test_size=0.30, random_state=0)
from sklearn.metrics import accuracy_score
##from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import f1_score
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train = tf_idf_vect.fit_transform(X_train)
X_test= tf_idf_vect.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
# Creating alpha values in the range from 10^-4 to 10^4
neighbors = []
i = 0.0001
while(i<=10000):
    neighbors.append(np.round(i,3))
    i *= 3


# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    bn = MultinomialNB(alpha = k)
    scores = cross_val_score(bn, X_train, Y_train, cv=10, scoring='f1_macro', n_jobs=-1)
    cv_scores.append(scores.mean())  
    
# determining best value of alpha
optimal_alpha = neighbors[cv_scores.index(max(cv_scores))]
print('\nThe optimal value of alpha is %.3f.' % optimal_alpha)
# plot f1_score vs alpha 
plt.plot(neighbors, cv_scores)
plt.xlabel('Value of alpha',size=10)
plt.ylabel('f1_score',size=10)
plt.title('f1_score VS Alpha_Value Plot',size=16)
plt.grid()
plt.show()

print("\n\nAlpha values :\n",neighbors)
print("\nf1_score for each alpha value is :\n ", np.round(cv_scores,5))
# ============================== Multinomial Naive Bayes with alpha = optimal_alpha ============================================
# instantiate learning model alpha = optimal_alpha
bn_optimal = MultinomialNB(alpha = optimal_alpha)

# fitting the model
bn_optimal.fit(X_train, Y_train)

# predict the response
predictions = bn_optimal.predict(X_test)

# evaluate accuracy
acc = accuracy_score(Y_test, predictions) * 100
print('\nThe Test Accuracy of the Multinomial naive Bayes classifier for alpha = %.3f is %f%%' % (optimal_alpha, acc))

# Variables that will be used for  making table in Conclusion part of this assignment
tfidf_multinomial_alpha = optimal_alpha
tfidf_multinomial_train_acc = max(cv_scores)*100
tfidf_multinomial_test_acc = acc
bn_optimal.classes_
# Now we can find log probabilities of different features for both the classes
class_features = bn_optimal.feature_log_prob_

#  row_0 is for 'Fake' class and row_1 is for 'True' class
Fake_features = class_features[0]
True_features = class_features[1]

# Getting all feature names
feature_names = tf_idf_vect.get_feature_names()

# Sorting 'Fake_features' and 'True_features' in descending order using argsort() function
sorted_Fake_features = np.argsort(Fake_features)[::-1]
sorted_True_features = np.argsort(True_features)[::-1]

print("Top 20 Important Features and their log probabilities For Fake News :\n\n")
for i in list(sorted_Fake_features[0:20]):
    print("%s\t -->\t%f  "%(feature_names[i],Fake_features[i]))
    
print("\n\nTop 20 Important Features and their log probabilities For true news :\n\n")
for i in list(sorted_True_features[0:20]):
    print("%s\t -->\t%f  "%(feature_names[i],True_features[i]))

MNB_f1 = round(f1_score(Y_test, predictions, average='weighted'), 3)
MNB_accuracy = round((accuracy_score(Y_test, predictions)*100),2)

print("Accuracy : " , MNB_accuracy , " %")
print("f1_score : " , MNB_f1)
# Code for drawing seaborn heatmaps
class_names = ['Fake','True']
df_heatmap = pd.DataFrame(confusion_matrix(Y_test, predictions), index=class_names, columns=class_names )
fig = plt.figure(figsize=(10,7))
heatmap = sb.heatmap(df_heatmap, annot=True, fmt="d")

# Setting tick labels for heatmap
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('Predicted label',size=18)
plt.xlabel('True label',size=18)
plt.title("Confusion Matrix\n",size=24)
plt.show()
%time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

depths=[1,5,50,100]
estimators=[1,5,50,100]
clf = RandomForestClassifier()

params = {'max_depth' : depths,
          'n_estimators':estimators  
          }

grid = GridSearchCV(estimator = clf,param_grid=params ,cv = 2,n_jobs = 3,scoring='roc_auc')
grid.fit(X_train, Y_train)
print("best depth = ", grid.best_params_)
print("AUC value on train data = ", grid.best_score_*100)
a1 = grid.best_params_

optimal_depth1 = a1.get('max_depth')
optimal_bases1 = a1.get('n_estimators')
clf = RandomForestClassifier(max_depth=optimal_depth1,n_estimators=optimal_bases1) 

clf.fit(X_train,Y_train)

pred = clf.predict(X_test)


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(Y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("Best AUC value")
print(roc_auc)
# Code for drawing seaborn heatmaps
class_names = ['Fake','True']
df_heatmap = pd.DataFrame(confusion_matrix(Y_test, pred), index=class_names, columns=class_names )
fig = plt.figure(figsize=(10,7))
heatmap = sb.heatmap(df_heatmap, annot=True, fmt="d")

# Setting tick labels for heatmap
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('Predicted label',size=18)
plt.xlabel('True label',size=18)
plt.title("Confusion Matrix\n",size=24)
plt.show()
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
acc1 = accuracy_score(Y_test, pred) * 100
pre1 = precision_score(Y_test, pred) * 100
rec1 = recall_score(Y_test, pred) * 100
f11 = f1_score(Y_test, pred) * 100
print('\nAccuracy=%f%%' % (acc1))
print('\nprecision=%f%%' % (pre1))
print('\nrecall=%f%%' % (rec1))
print('\nF1-Score=%f%%' % (f11))
# Calculate feature importances from decision trees
importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1][:25]

# Rearrange feature names so they match the sorted feature importances
names = tf_idf_vect.get_feature_names()

sb.set(rc={'figure.figsize':(11.7,8.27)})

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(25), importances[indices])

# Add feature names as x-axis labels
names = np.array(names)
plt.xticks(range(25), names[indices], rotation=90)

# Show plot
plt.show()
# uni_gram.get_feature_names()
df=names[indices]
print(df)
from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 600,background_color ='white').generate(str(df))
plt.imshow(wordcloud)
plt.title(" Frequent words")
plt.show()
