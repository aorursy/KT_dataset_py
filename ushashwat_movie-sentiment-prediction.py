import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
# Read and display the dataset
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head(10)
# Check number of reviews
df['sentiment'].value_counts()
# Display summary
df.describe()
# Print a sample review
print (df['review'][0])
# Label encoding
label_encoder = preprocessing.LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
df.head(10)
# Text preprocessing
def preprocessing(review_imdb):
    review = BeautifulSoup(review_imdb).get_text() # Remove HTML tags
    review = re.sub("[^a-zA-Z]", " ", review) # Remove special characters
    review = review.lower().split() # Convert to lowercase and split each word
    
    stop_w = set(stopwords.words("english")) # Use a set instead of list for faster searching
    review = [w for w in review if not w in stop_w] # Remove stop words
    review = [WordNetLemmatizer().lemmatize(w) for w in review] # Lemmatization
    
    return (" ".join(review)) # Return the words after joining each word separated by space
# Clean all movie reviews
clean_reviews = []

for i in range(0, df['review'].size):
    clean_reviews.append(preprocessing(df['review'][i]))
    if( (i+1)%5000 == 0 ):
        print ("Review %d of 50000 done\n" % ( i+1))
# Check a sample review after preprocessing
clean_reviews[0]
# Display word cloud
unique_str=(" ").join(clean_reviews)
wordcloud = WordCloud(width=1600,height=800).generate(unique_str)
plt.figure(figsize=(18,9))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(np.stack(clean_reviews), df['sentiment'], test_size=0.2)
# Use TF-IDF to vectorize
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2)) # Take 1 & 2 tokens into consideration
tfidf_train = tfidf_vec.fit_transform(x_train)
tfidf_test = tfidf_vec.transform(x_test)
# Linear support vector classification
linear_svc = LinearSVC()
linear_svc.fit(tfidf_train, y_train)
y_pred = linear_svc.predict(tfidf_test)
# Evaluation metrics
print(classification_report(y_test, y_pred,target_names=['Negative','Positive']))
# Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=[0, 1],
                      title='Confusion matrix')
# Final accuracy
print("Accuracy: ",accuracy_score(y_test, y_pred))
