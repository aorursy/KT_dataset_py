import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
data = pd.read_csv(r'../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv', encoding='latin1')

# baştaki r harfi (read) özel karakterleri (\n gibi) yok sayar. encoding='latin1' dememizin sebebi, data'nın içinde latin harflerinin olmasıdır. Öyle yapmazsak hata verir.



data.columns
data = pd.concat([data.gender, data.description], axis=1)

data.head()
print(data.isnull().sum())

print('shape: ', data.shape)
data.dropna(axis=0, inplace=True) # null olan değerleri satır olarak (axis=0) sil dedik. dropna == null değerleri siler



print(data.isnull().sum())

print('shape: ', data.shape)
data.gender = [1 if each=='female' else 0 for each in data.gender]

data.head()
import re  # Regular Expression



desc = data.description[4]

desc
new_desc = re.sub("[^a-zA-Z]", " ", desc)



new_desc = new_desc.lower() # bütün harfleri küçük yaptık

new_desc
import nltk  # natural language tool kit



# nltk.download('stopwords')  # corpus diye bir klasöre internetten indiriyor.



from nltk.corpus import stopwords  # sonra corpus klasöründen import ediyoruz
demo = "You shouldn't go there"



split_method = demo.split()

tokenize_method = nltk.word_tokenize(demo)



print(split_method)

print(tokenize_method)
new_desc = nltk.word_tokenize(new_desc)

print(new_desc)
print(stopwords.words('english'))
new_desc = [word for word in new_desc if not word in stopwords.words('english')]

# the kelimesi gitti
lemma = nltk.WordNetLemmatizer()



new_desc = [lemma.lemmatize(word) for word in new_desc]  # memories --> memory oldu



new_desc = " ".join(new_desc) # kelimeleri birleştiriyouz



new_desc