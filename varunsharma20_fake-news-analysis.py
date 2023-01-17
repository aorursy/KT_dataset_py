# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/nlp-getting-started/train.csv")
data[:10]
train = data['text']
train.shape
Y_train = data['target']
Y_train.shape
from matplotlib import pyplot as plt
cls = np.unique(Y_train,return_counts=True)
cls
from matplotlib import pyplot as plt
real = data[data['target']==1].shape[0]
fake = data[data['target']==0].shape[0]
plt.bar(10,real,label="real")
plt.bar(12,fake,label="not_real")
plt.legend()
plt.ylabel('Number of examples')
plt.title('Propertion of examples')
plt.show()
test = pd.read_csv('../input/nlp-getting-started/test.csv')
test.head()
test.shape
x = test['text']
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
tokenizer = RegexpTokenizer(r'[a-z]\w+')
ps = PorterStemmer()
sw = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    ft = tokenizer.tokenize(text)
    clean = ' '.join(ft)
    return clean
list_X = list(train)
list_x = list(x)
clean_train = [clean_text(i) for i in list_X]
clean_train[:10]
clean_test = [clean_text(i) for i in list_x]
clean_test[:10]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2))

vocab = cv.fit_transform(clean_train)
X = vocab.toarray()
x_test = cv.transform(clean_test)
x = x_test.toarray()
print(X.shape)
print(x.shape)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
TX,tx,TY,ty = train_test_split(X,Y_train,test_size=0.3,random_state=42)
print(TX.shape)
print(TY.shape)
print(tx.shape)
print(ty.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(TX,TY)
y = lr.predict(tx)

accuracy_score(ty,y)
test_pred = lr.predict(x)
test_pred = np.array(test_pred)
df = pd.DataFrame({'id':test['id'],
                   'target':test_pred})

df.to_csv("./predictions.csv",index=None)
