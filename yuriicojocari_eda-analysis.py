import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')



# training_data = pd.read_csv("input/train.csv.zip", encoding="ISO-8859-1")

# testing_data = pd.read_csv("input/test.csv.zip", encoding="ISO-8859-1")

# attribute_data = pd.read_csv('input/attributes.csv.zip')

# descriptions = pd.read_csv('input/product_descriptions.csv.zip')





training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")

testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")

attribute_data = pd.read_csv('../input/attributes.csv')

descriptions = pd.read_csv('../input/product_descriptions.csv')

print("training data shape is:",training_data.shape)

print("testing data shape is:",testing_data.shape)

print("attribute data shape is:",attribute_data.shape)

print("description data shape is:",descriptions.shape)
print("training data has empty values:",training_data.isnull().values.any())

print("testing data has empty values:",testing_data.isnull().values.any())

print("attribute data has empty values:",attribute_data.isnull().values.any())

print("description data has empty values:",descriptions.isnull().values.any())
training_data.head(10)
print("there are in total {} products ".format(len(training_data.product_title.unique())))

print("there are in total {} search query ".format(len(training_data.search_term.unique())))

print("there are in total {} product_uid".format(len(training_data.product_uid.unique())))





testing_data.head(10)
print("there are in total {} products ".format(len(testing_data.product_title.unique())))

print("there are in total {} search query ".format(len(testing_data.search_term.unique())))

print("there are in total {} product_uid".format(len(testing_data.product_uid.unique())))







attribute_data.head(10)
print("there are in total {} product_uid ".format(len(attribute_data.product_uid.unique())))

print("there are in total {} names ".format(len(attribute_data.name.unique())))

print("there are in total {} values".format(len(attribute_data.value.unique())))









descriptions.head(10)
print("there are in total {} product_uid ".format(len(descriptions.product_uid.unique())))

print("there are in total {} product_descriptions ".format(len(descriptions.product_description.unique())))











(descriptions.product_description.str.count('\d+') + 1).hist(bins=30)

(descriptions.product_description.str.count('\W')+1).hist(bins=30)





(training_data.product_title.str.count("\\d+") + 1).hist(bins=30)#plot number of digits in title

(training_data.product_title.str.count("\\w+") + 1).hist(bins=30)#plot number of digits in title







(training_data.search_term.str.count("\\w+") + 1).hist(bins=30) #plot number of words in search therms

(training_data.search_term.str.count("\\d+") + 1).hist(bins=30) #plot number of digits in search terms











(training_data.relevance ).hist(bins=30)
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.relevance.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.relevance)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
print('total data has html tags in',descriptions.product_description.str.count('<br$').values.sum())
descriptions[descriptions.product_description.str.contains("<br")].values.tolist()[:3]
descriptions.product_description.str.contains("Click here to review our return policy for additional information regarding returns").values.sum()
training_data[training_data.search_term.str.contains("^\\d+ . \\d+$")].head(10)
training_data[training_data.product_uid==100030]
## let's create first the cleaning functions

from bs4 import BeautifulSoup

import lxml

import re

import nltk

from nltk.corpus import stopwords # Import the stop word list

from nltk.metrics import edit_distance

from string import punctuation

from collections import Counter





def remove_html_tag(text):

    soup = BeautifulSoup(text, 'lxml')

    text = soup.get_text().replace('Click here to review our return policy for additional information regarding returns', '')

    return text



def str_stemmer(doc):

    # split into tokens by white space

    tokens = doc.split()

    # remove punctuation from each token

    table = str.maketrans('', '', punctuation)

    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens

    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)



def str_stemmer_title(s):

#     return " ".join([stemmer.stem(word) for word in s.lower().split()])

    return " ".join(map(stemmer.stem, s.lower().split()))



def str_common_word(str1, str2):

    whole_set = set(str1.split())

#     return sum(int(str2.find(word)>=0) for word in whole_set)

    return sum(int(str2.find(word)>=0) for word in whole_set)





def get_shared_words(row_data):

    return np.sum([str_common_word(*row_data[:-1]), str_common_word(*row_data[1:])])



############### cleaning html tags ##################

has_tag_in = descriptions.product_description.str.contains('<br')

descriptions.loc[has_tag_in, 'product_description'] = descriptions.loc[has_tag_in, 'product_description'].map(lambda x:remove_html_tag(x))

###############
import requests

import re

import time

from random import randint



START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"

END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (("'", '&#39;'),('"', '&quot;'),('>', '&gt;'),('<', '&lt;'),('&', '&amp;'))



def spell_check(s):

    q = '+'.join(s.split())

    time.sleep(  randint(0,1) ) #relax and don't let google be angry

    r = requests.get("https://www.google.co.uk/search?q="+q)

    content = r.text

    start=content.find(START_SPELL_CHECK) 

    if ( start > -1 ):

        start = start + len(START_SPELL_CHECK)

        end=content.find(END_SPELL_CHECK)

        search= content[start:end]

        search = re.sub(r'<[^>]+>', '', search)

        for code in HTML_Codes:

            search = search.replace(code[1], code[0])

        search = search[1:]

    else:

        search = s

    return search 
training_data = pd.merge(training_data, descriptions, 

                         on="product_uid", how="left")
print("It has blank/empty fields ",training_data.isnull().values.sum())

print("has blank/empty values",training_data.isnull().values.any())
from nltk.corpus import brown, stopwords

from nltk.cluster.util import cosine_distance

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter





def sentence_similarity(columns,stopwords=None):

    sent1, sent2 = columns[0], columns[1]

    sent1 = sent1.split(' ')

    sent2 = sent2.split(' ')

    if stopwords is None:

        stopwords = []

 

    sent1 = [w.lower() for w in sent1]

    sent2 = [w.lower() for w in sent2]

 

    all_words = list(set(sent1 + sent2))

 

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)

 

    # build the vector for the first sentence

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # build the vector for the second sentence

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2)



def get_jaccard_sim(columns): 

    str1, str2 = columns[0], columns[1]

    a = set(str1.split()) 

    b = set(str2.split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))





def calc_edit_dist(row):

    return edit_distance(*row)



    
################begin testing

## let's create first the cleaning functions

from bs4 import BeautifulSoup

import lxml

import re

import nltk

from nltk.corpus import stopwords # Import the stop word list

from nltk.metrics import edit_distance

from string import punctuation

from collections import Counter





def remove_html_tag(text):

    soup = BeautifulSoup(text, 'lxml')

    text = soup.get_text().replace('Click here to review our return policy for additional information regarding returns', '')

    return text



def str_stemmer(doc):

    # split into tokens by white space

    tokens = doc.split()

    # remove punctuation from each token

    table = str.maketrans('', '', punctuation)

    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens

    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)





def str_stemmer_tokens(tokens):

    # split into tokens by white space

#     tokens = doc.split()

    # remove punctuation from each token

    table = str.maketrans('', '', punctuation)

    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic

    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words

    stop_words = set(stopwords.words('english'))

    tokens = [w for w in tokens if not w in stop_words]

    # filter out short tokens

    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)



def str_stemmer_title(s):

    return " ".join(map(stemmer.stem, s))



def str_common_word(str1, str2):

    whole_set = set(str1.split())

#     return sum(int(str2.find(word)>=0) for word in whole_set)

    return sum(int(str2.find(word)>=0) for word in whole_set)





# def str_common_word(str1, str2):

#     return sum(int(str2.find(word)>=0) for word in str1.split())





def str_common_word2(str1, str2):

    part_of_first = set(str1)

    return sum(1 for word in str2 if word in part_of_first)

#     return sum(int(str2.find(word)>=0) for word in str1.split())



def get_shared_words_mut(row_data):

    return np.sum([str_common_word2(*row_data[:-1]), str_common_word2(*row_data[1:])])





def get_shared_words_imut(row_data):

    return np.sum([str_common_word(*row_data[:-1]), str_common_word2(*row_data[1:])])

    

from nltk.corpus import brown, stopwords

from nltk.cluster.util import cosine_distance

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter





def sentence_similarity(columns,stopwords=None):

    sent1, sent2 = columns[0], columns[1]

    if stopwords is None:

        stopwords = []

 

    sent1 = [w.lower() for w in sent1]

    sent2 = [w.lower() for w in sent2]

 

    all_words = list(set(sent1 + sent2))

 

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)

 

    # build the vector for the first sentence

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # build the vector for the second sentence

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2)



def get_jaccard_sim(columns): 

    str1, str2 = columns[0], columns[1]

    a = set(str1) 

    b = set(str2)

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))

############## apply stemming #####################

#  also .apply(, raw=True) might be a good options

# https://github.com/s-heisler/pycon2017-optimizing-pandas to see why it is done on this way

############## apply stemming #####################

# training_data['search_term'] = list(map(str_stemmer_title,training_data['search_term'].values))

# training_data['product_title'] = list(map(str_stemmer, training_data['product_title'].values))

# training_data['product_description'] = list(map(str_stemmer, training_data['product_description'].values))



# training_data['shared_words'] = list(map(get_shared_words, training_data[['search_term','product_description', 'product_title']].values))



# training_data["edistance_sprot"] = list(map(calc_edit_dist, training_data[["search_term","product_title"]].values))

# training_data["edistance_sd"] = list(map(calc_edit_dist, training_data[["search_term","product_description"]].values))





# training_data['cos_dis_sqt'] = list(map(sentence_similarity ,training_data[["search_term","product_title"]].values))

# training_data['cos_dis_sqd'] = list(map(sentence_similarity, training_data[["search_term","product_description"]].values))





# training_data['j_dis_sqt'] = list(map(get_jaccard_sim, training_data[["search_term","product_title"]].values))

# training_data['j_dis_sqd'] = list(map(get_jaccard_sim, training_data[["search_term","product_description"]].values))



# training_data['j_dis_sqt'] = list(map(get_jaccard_sim, training_data[["search_term","product_title"]].values))

# training_data['j_dis_sqd'] = list(map(get_jaccard_sim, training_data[["search_term","product_description"]].values))



# training_data['search_query_length'] = training_data.search_term.str.len()

# training_data['number_of_words_in_descr'] = training_data.product_description.str.count("\\w+")







training_data['search_term_tokens'] = training_data.search_term.str.lower().str.split()

training_data['product_title_tokens'] = training_data.product_title.str.lower().str.split()

training_data['product_description_tokens'] = training_data.product_description.str.lower().str.split()



training_data['search_term'] = [str_stemmer_title(_) for _ in training_data.search_term_tokens.values.tolist()]

training_data['product_title'] = [str_stemmer_tokens(_) for _ in training_data.product_title_tokens.values.tolist()]

training_data['product_description'] = [str_stemmer_tokens(_) for _ in training_data.product_description_tokens.values.tolist()]





training_data['shared_words_mut'] = [get_shared_words_mut(columns)

                         for columns in 

                         training_data[['search_term_tokens', 'product_title_tokens', 'product_description_tokens']].values.tolist()

                        ]



training_data['shared_words'] = list(map(get_shared_words_imut, training_data[['search_term','product_description', 'product_title']].values))







training_data["edistance_sprot"] = [edit_distance(word1, word2) for word1, word2 in

                                    training_data[["search_term","product_title"]].values.tolist()]





training_data["edistance_sd"] = [edit_distance(word1, word2) for word1, word2 in

                                    training_data[["search_term","product_description"]].values.tolist()]



training_data['j_dis_sqt'] = [get_jaccard_sim(rows) for rows in training_data[["search_term_tokens","product_title_tokens"]].values]

training_data['j_dis_sqd'] = [get_jaccard_sim(rows) for rows in training_data[["search_term_tokens","product_description_tokens"]].values]



training_data['search_query_length'] = training_data.search_term.str.len()

training_data['number_of_words_in_descr'] = training_data.product_description.str.count("\\w+")





training_data['cos_dis_sqt'] = [ sentence_similarity(rows) for rows in training_data[["search_term","product_title"]].values]

training_data['cos_dis_sqd'] = [sentence_similarity(rows) for rows in training_data[["search_term","product_description"]].values]





# training_data.corr()

training_data.head(3)
testing_data = pd.merge(testing_data, descriptions, 

                         on="product_uid", how="left")

print("has blank/empty values",testing_data.isnull().values.any())
############## apply stemming for test data #####################

# testing_data['search_term'] = list(map(str_stemmer_title, testing_data['search_term'].values))

# testing_data['product_title'] = list(map(str_stemmer, testing_data['product_title'].values))

# testing_data['product_description'] = list(map(str_stemmer, testing_data['product_description'].values))

testing_data['search_term_tokens'] = testing_data.search_term.str.lower().str.split()

testing_data['product_title_tokens'] = testing_data.product_title.str.lower().str.split()

testing_data['product_description_tokens'] = testing_data.product_description.str.lower().str.split()



testing_data['search_term'] = [str_stemmer_title(_) for _ in testing_data.search_term_tokens.values.tolist()]

testing_data['product_title'] = [str_stemmer_tokens(_) for _ in testing_data.product_title_tokens.values.tolist()]

testing_data['product_description'] = [str_stemmer_tokens(_) for _ in testing_data.product_description_tokens.values.tolist()]



############## end stemming #####################
############## building custome feature for test data, let's build a few of them before compare which one is the best ###########

# testing_data['shared_words'] = list(map(get_shared_words, testing_data[['search_term','product_description', 'product_title']].values))

# testing_data["edistance_sprot"] = list(map(calc_edit_dist, testing_data[["search_term","product_title"]].values))

# testing_data["edistance_sd"] = list(map(calc_edit_dist, testing_data[["search_term","product_description"]].values))





# testing_data['cos_dis_sqt'] = list(map(sentence_similarity ,testing_data[["search_term","product_title"]].values))

# testing_data['cos_dis_sqd'] = list(map(sentence_similarity, testing_data[["search_term","product_description"]].values))







# testing_data['j_dis_sqt'] = list(map(get_jaccard_sim, testing_data[["search_term","product_title"]].values))

# testing_data['j_dis_sqd'] = list(map(get_jaccard_sim, testing_data[["search_term","product_description"]].values))



# testing_data['j_dis_sqt'] = list(map(get_jaccard_sim, testing_data[["search_term","product_title"]].values))

# testing_data['j_dis_sqd'] = list(map(get_jaccard_sim, testing_data[["search_term","product_description"]].values))



# testing_data['search_query_length'] = testing_data.search_term.str.len()

# testing_data['number_of_words_in_descr'] = testing_data.product_description.str.count("\\w+")



testing_data['shared_words_mut'] = [get_shared_words_mut(columns)

                         for columns in 

                         testing_data[['search_term_tokens', 'product_title_tokens', 'product_description_tokens']].values.tolist()

                        ]



testing_data['shared_words'] = list(map(get_shared_words_imut, testing_data[['search_term','product_description', 'product_title']].values))







testing_data["edistance_sprot"] = [edit_distance(word1, word2) for word1, word2 in

                                    testing_data[["search_term","product_title"]].values.tolist()]





testing_data["edistance_sd"] = [edit_distance(word1, word2) for word1, word2 in

                                    testing_data[["search_term","product_description"]].values.tolist()]



testing_data['j_dis_sqt'] = [get_jaccard_sim(rows) for rows in testing_data[["search_term_tokens","product_title_tokens"]].values]

testing_data['j_dis_sqd'] = [get_jaccard_sim(rows) for rows in testing_data[["search_term_tokens","product_description_tokens"]].values]



testing_data['search_query_length'] = testing_data.search_term.str.len()

testing_data['number_of_words_in_descr'] = testing_data.product_description.str.count("\\w+")





testing_data['cos_dis_sqt'] = [ sentence_similarity(rows) for rows in testing_data[["search_term","product_title"]].values]

testing_data['cos_dis_sqd'] = [sentence_similarity(rows) for rows in testing_data[["search_term","product_description"]].values]





testing_data.corr()
training_data.describe()
testing_data.describe()
import seaborn as sns

plt.figure(figsize=(12, 12))

temp = training_data.drop(['product_uid','id'],axis=1)

sns.heatmap(temp.corr(), annot=True)

plt.show()
import seaborn as sns

plt.figure(figsize=(12, 12))

temp = testing_data.drop(['product_uid','id'],axis=1)

sns.heatmap(temp.corr(), annot=True)

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.cos_dis_sqd.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.cos_dis_sqd)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
from statsmodels.graphics.gofplots import qqplot

from scipy.stats import shapiro





from matplotlib import pyplot

qqplot(training_data.cos_dis_sqd, line='s')

pyplot.show()



stat, p = shapiro(training_data.cos_dis_sqd)

print('Statistics=%.3f, p=%.3f' % (stat, p))
from scipy.stats import normaltest



stat, p = normaltest(training_data.cos_dis_sqd)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Sample looks Gaussian (fail to reject H0)')

else:

    print('Sample does not look Gaussian (reject H0)')
from scipy.stats import anderson



result = anderson(training_data.cos_dis_sqd)

print('Statistic: %.3f' % result.statistic)

p = 0

for i in range(len(result.critical_values)):

    sl, cv = result.significance_level[i], result.critical_values[i]

    if result.statistic < result.critical_values[i]:

        print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

    else:

        print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.cos_dis_sqt.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.cos_dis_sqt)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
from matplotlib import pyplot

qqplot(training_data.cos_dis_sqt, line='s')

pyplot.show()



stat, p = shapiro(training_data.cos_dis_sqt)

print('Statistics=%.3f, p=%.3f' % (stat, p))
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.shared_words.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.shared_words)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
from statsmodels.graphics.gofplots import qqplot

from scipy.stats import shapiro





from matplotlib import pyplot

qqplot(training_data.shared_words, line='s')

pyplot.show()



stat, p = shapiro(training_data.shared_words)

print('Statistics=%.3f, p=%.3f' % (stat, p))
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.edistance_sprot.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.edistance_sprot)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



training_data.search_query_length.plot(kind='hist', normed=True)



mu, std = norm.fit(training_data.search_query_length)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



testing_data.cos_dis_sqd.plot(kind='hist', normed=True)



mu, std = norm.fit(testing_data.cos_dis_sqd)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



testing_data.cos_dis_sqt.plot(kind='hist', normed=True)



mu, std = norm.fit(testing_data.cos_dis_sqt)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



testing_data.shared_words.plot(kind='hist', normed=True)



mu, std = norm.fit(testing_data.shared_words)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



testing_data.edistance_sprot.plot(kind='hist', normed=True)



mu, std = norm.fit(testing_data.edistance_sprot)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import norm  



testing_data.search_query_length.plot(kind='hist', normed=True)



mu, std = norm.fit(testing_data.search_query_length)



xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100)

p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)



plt.show()
sns.pairplot(training_data)
sns.pairplot(testing_dataing_data)
df_training = training_data.drop(['product_title','search_term','product_description', 'product_title_tokens', 'product_description_tokens','product_title_tokens','search_term_tokens'],axis=1)



y_train = df_training['relevance'].values

X_train = df_training.drop(['id','relevance'],axis=1).values
df_training.head(3)
# X_test = testing_data.drop(['id','product_title','search_term','product_description'],axis=1).values

X_test = testing_data.drop(['id','product_title','search_term','product_description', 'product_title_tokens', 'product_description_tokens','product_title_tokens','search_term_tokens'],axis=1).values



id_test = testing_data['id']

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 3, n_jobs = -1, random_state = 17, verbose = 1)

rfr.fit(X_train, y_train)



y_pred = rfr.predict(X_test)



pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)



from sklearn.linear_model import LinearRegression

lr = LinearRegression(n_jobs = -1)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
import sklearn

from sklearn.ensemble import GradientBoostingRegressor



param_grid = {

                'loss' : ['ls'],

                'n_estimators' : [3], 

                'max_depth' : [9],

                'max_features' : ['auto'] 

             }



gbr = GradientBoostingRegressor()



model_gbr = sklearn.model_selection.GridSearchCV(estimator = gbr, n_jobs = -1, param_grid = param_grid)

model_gbr.fit(X_train, y_train)



y_pred = model_gbr.predict(X_test)



# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
from sklearn.ensemble import BaggingRegressor

rf = RandomForestRegressor(max_depth = 20, max_features =  'sqrt', n_estimators = 3)

clf = BaggingRegressor(rf, n_estimators=3, max_samples=0.1, random_state=25)



clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
# define models which will be chained togher in a bigger model, which aims to predict the relevancy score

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold



#define standard scaler

scaler = StandardScaler()

scaler.fit(X_train, y_train)

scaled_train_data = scaler.transform(X_train)

scaled_test_data = scaler.transform(X_test)





rf = RandomForestRegressor(n_estimators=4, max_depth=6, random_state=0)

clf = BaggingRegressor(rf, n_estimators=4, max_samples=0.1, random_state=25)





pipeline = Pipeline(steps = [('scaling', scaler), ('baggingregressor', clf)])

#end pipeline 

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

from sklearn.linear_model import BayesianRidge



gnb = BayesianRidge()

param_grid = {}

model_nb = sklearn.model_selection.GridSearchCV(estimator = gnb, param_grid = param_grid, n_jobs = -1)

model_nb.fit(X_train, y_train)



y_pred = model_nb.predict(X_test)

# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
from xgboost import XGBRegressor



xgb = XGBRegressor()

param_grid = {'max_depth':[5, 6], 

              'n_estimators': [130, 150, 170], 

              'learning_rate' : [0.1]}

model_xgb = sklearn.model_selection.GridSearchCV(estimator = xgb, param_grid = param_grid, n_jobs = -1)

model_xgb.fit(X_train, y_train)



y_pred = model_xgb.predict(X_test)

# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)
