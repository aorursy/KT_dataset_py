import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataword = pd.read_csv('../input/facedata/facedataen.csv')
datamain = pd.read_csv('../input/facebook-data-leak-ageemotion-data/maindata.csv')
datagoogle = pd.read_csv('../input/googletrendsfacebookdataleak/googletrends.csv')
datamain.head(10)
datagoogle.head(30)
dataword.head(10)
datamain['Emotion'].value_counts().plot(kind = 'bar')
plt.title("Emotion")
plt.grid()
datamain['Age'].value_counts().plot(kind = 'bar')
plt.title("Age")
plt.grid()
import seaborn as sns
sns.barplot(x="Facebook data leak: (Worldwide)", y="Day", data=datagoogle,  palette="Blues_d")
plt.grid()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/facedata/facedataen.csv')

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 68):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Data'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(hidden_layer_sizes=(7,), max_iter=2000, alpha=0.1,
                     solver='sgd', verbose=10, random_state=21, tol=0.000000001)

nn.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nn.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_MLPClassifier = round(accuracy_score(y_test, y_pred) * 100, 2)
print()
print("confusion_matrix:\n", cm)
print("accuracy_score: ", acc_MLPClassifier)
from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(hidden_layer_sizes=(7,), max_iter=2500, alpha=0.1,
                     solver='sgd', verbose=10,random_state=21,tol=0.000000001)

nn.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nn.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_MLPClassifier1 = round(accuracy_score(y_test, y_pred) * 100, 2)
print()
print("confusion_matrix:\n", cm)
print("accuracy_score: ", acc_MLPClassifier1)
from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(hidden_layer_sizes=(15,), max_iter=2000, alpha=0.1,
                     solver='sgd', verbose=10,random_state=21,tol=0.000000001)

nn.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nn.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_MLPClassifier2 = round(accuracy_score(y_test, y_pred) * 100, 2)
print()
print("confusion_matrix:\n", cm)
print("accuracy_score: ", acc_MLPClassifier2)
from sklearn.neural_network import MLPClassifier
nn=MLPClassifier(hidden_layer_sizes=(15,), max_iter=2500, alpha=0.1,
                     solver='sgd', verbose=10,random_state=21,tol=0.000000001)

nn.fit(X_train,y_train)
# Predicting the Test set results
y_pred = nn.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_MLPClassifier3 = round(accuracy_score(y_test, y_pred) * 100, 2)
print()
print("confusion_matrix:\n", cm)
print("accuracy_score: ", acc_MLPClassifier3)
models = pd.DataFrame({
    'Model': ['Iter=2000 HL=7', 
              'Iter=2500 HL=7', 'Iter=2000 HL=15', 
              'Iter=2500 HL=15'],
    'Score': [ acc_MLPClassifier, 
              acc_MLPClassifier1, acc_MLPClassifier2, 
             acc_MLPClassifier3]})
models.sort_values(by='Score', ascending=False)
