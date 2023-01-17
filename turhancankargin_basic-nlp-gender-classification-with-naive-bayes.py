import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding = "latin1")

data.head()
data.columns
data.info()
data = pd.concat([data.gender,data.description],axis=1) # New data contains just two columns

data.dropna(axis = 0,inplace = True) # Drop NaN values

data.gender = [1 if each == "female" else 0 for each in data.gender] # 1 for female, 0 for male

data.gender.value_counts()
data.head()
first_description = data.description[4] 

first_description
import re

description = re.sub("[^a-zA-Z]"," ",first_description)  # Except from a to z, and from A to Z will be transform to space

description = description.lower()   # Make whole words lowercase

description
import nltk # natural language tool kit

nltk.download("stopwords")      

from nltk.corpus import stopwords  

description = nltk.word_tokenize(description) # To split words

description = [ word for word in description if not word in set(stopwords.words("english"))]
description
import nltk as nlp



lemma = nlp.WordNetLemmatizer()

description = [ lemma.lemmatize(word) for word in description] 



description = " ".join(description)
description
description_list = []

for description in data.description:

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   

    description = nltk.word_tokenize(description)

    description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description)
from sklearn.feature_extraction.text import CountVectorizer # for bag of words

max_features = 5000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x

print("Most Common {} word is {}".format(max_features,count_vectorizer.get_feature_names()))
y = data.iloc[:,0].values   # male or female classes (output)

x = sparce_matrix # our input



import seaborn as sns

import matplotlib.pyplot as plt

# visualize number of digits classes

plt.figure(figsize=(15,7))

sns.countplot(y)

plt.title("Number of Gender")
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)





# naive bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)



# prediction

y_pred = nb.predict(x_test)



print("Accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))