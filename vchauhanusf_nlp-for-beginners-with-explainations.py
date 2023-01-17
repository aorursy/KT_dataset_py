# let's start by importing the files 

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# As we can see the input in is zip files 
# let's extract these files first
import zipfile

zip_files=['/kaggle/input/spooky-author-identification/train.zip','/kaggle/input/spooky-author-identification/test.zip']
for zip_file in zip_files:
    with zipfile.ZipFile(zip_file,'r') as z:
        z.extractall()

os.listdir()
# As we can see now the files have been extracted and now we have our train and text csv files
# Since we will we dealing with text in English Language let's import the stop words for english.


from nltk.corpus import stopwords

# these are the general words i.e ( a , the , is , an , i me , my) which are very general and does not 
#make much sense so we will fiter out these words and will consider the more robust words in our text
# example ( he is a dangerous man , here is ,a can be skipped and we get the essence of the sentence from the 
# word dangerous )
# let's look at some of the stop words
stop=stopwords.words('english')
stop
# let's read our data into our dataframes
train=pd.read_csv('train.csv',index_col=False)
test=pd.read_csv('test.csv',index_col=False)

# let's take a peek at our data
train.sample(10)
#let's see how our target variable is distributed in our overall  data
import seaborn as sns
sns.countplot(train['author'])
# since our target varibale has 3 different values we will now convert these into 0,1,2 using the label encoder

from sklearn.preprocessing import LabelEncoder
y=train.loc[:,['author']]
en=LabelEncoder()
y=en.fit_transform(y.author.values)

# now let's split our data into train and test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train.text,y,stratify=y,test_size=.25,shuffle=True)
sns.countplot(y_train)
sns.countplot(y_test)
# defining the loss function ( log_loss)
from sklearn.metrics import log_loss
# let's import the TfidfVectorizer and see how it works
from sklearn.feature_extraction.text import TfidfVectorizer

# let's create and instance of the vectorizer 

tfid=TfidfVectorizer(stop_words='english')

xx=['hello today we will try' ,
   'to understand how  tf-idf works',
   'this is sample text for this nlp program']

cc=tfid.fit_transform(xx)

dd=pd.DataFrame(cc.toarray(),columns=tfid.get_feature_names())
dd

# I have created a dataframe to demonstrate what happens under the hood

# In the text we have 4 lines, 

# let's start with first line which has 4 distinct words but notice that (will and we got dropped because those
# are stop words and the rest of the words are assigned a numeric value which depends on how many times the word
# appeared in the line and how many times the words appears in all the text)

# there is math involved in this process but that is beyond the scope of this notebook, so let's try to understand
# the working and leave the math for another notebook.
# we pass all the text into our vectorizer and fit it
tfid.fit(list(x_train)+list(x_test))
# we then transform our x_train and x_test using this vetcorizer 
x_train=tfid.transform(x_train)


# if we want to understand this process we can think of it as we tell our model these are the total words we have in
# our text ,  
#1.please remove the stop words,
#2. Please assign a numeric value to each word so that we can feed it to our machine learning model
x_test=tfid.transform(x_test)
# Now that we have our data in numeric Values we can implement any of our Machine learning models

# let's try a simple linear model , Logistic Regression

from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(C=1.0)

# let's fit our model
lgr.fit(x_train,y_train)
# now our target variable is in shape (4895,) but we want it to be in shape (4895,3) because our data has there
# different classes so let's do binarization of our target variable 
x_test
a=lgr.predict_proba(x_test)
def conver(actual,a):
    if len(actual.shape)==1:
        temp=np.zeros((actual.shape[0],a.shape[1]))
        for i,j in enumerate(actual):
            temp[i,j]=1
        actual=temp    
    return actual
aa=conver(y_test,a)

a.shape


# To put it in simple words , intially our target ( y_test ) was in form 2,1,0,0,0,1,2,1 but we want it to 
#be in th form 
[[1,0,0],
 [0,1,0],
 [0,1,0]]

# now that our predictions and target are in the same shape we can calculate the log loss
aa.shape

log_loss(aa,a)
# let's try to use a support vetcor classifier for the same data
from sklearn.svm import SVC
sv_model=SVC(probability=True)
sv_model.fit(x_train,y_train)



predicted=sv_model.predict_proba(x_test)
predicted.shape
# As we can see our SVC model performs better than the Logistic Regression model
log_loss(y_test,predicted)