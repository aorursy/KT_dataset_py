# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Required packages



import matplotlib.pyplot as plt

%matplotlib inline

from nltk.corpus import stopwords



from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator



from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
#Amazon Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt"

amazon = pd.read_csv(input_file,delimiter='\t',header=None, names=['review', 'sentiment'])

amazon['source']='amazon'



#Yelp Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt"

yelp = pd.read_csv(input_file,delimiter='\t',header=None, names=['review', 'sentiment'])

yelp['source']='yelp'



#Imdb Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt"

imdb = pd.read_csv(input_file,delimiter='\t',header=None, names=['review', 'sentiment'])

imdb['source']='imdb'



#combine all data sets

data = pd.DataFrame()

data = pd.concat([amazon, yelp, imdb])

data['sentiment'] = data['sentiment'].astype(str)

print(data.head(5))

print(data.tail(5))
print(data.info())



data['source'].value_counts().plot(kind='pie', autopct='%1.0f%%', shadow=True)
data.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
data['word_count'] = data['review'].apply(lambda x : len(x.split()))

data['char_count'] = data['review'].apply(lambda x : len(x.replace(" ","")))

data['stopwords'] = data['review'].apply(lambda x: len([x for x in x.split() if x in stopwords.words('english')]))

data['num_count'] = data['review'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

data['upper_count'] = data['review'].apply(lambda x: len([x for x in x.split() if x.isupper()]))



print(data[['word_count', 'char_count', 'stopwords', 'num_count', 'upper_count']].head(5))



data.sum(axis = 0, numeric_only = True)
all_df1 = data[['review', 'sentiment']]

all_df1['sentiment'] = all_df1['sentiment'].astype(int)

df_1 = all_df1[all_df1['sentiment']==1]

df_0 = all_df1[all_df1['sentiment']==0]

rev_All = " ".join(review for review in all_df1.review)

rev_1 = " ".join(review for review in df_1.review)

rev_0 = " ".join(review for review in df_0.review)



fig, ax = plt.subplots(3, 1, figsize  = (30,30))

# Create and generate a word cloud image:

wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(rev_All)

wordcloud_1 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(rev_1)

wordcloud_0 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(rev_0)



# Display the generated image:

ax[0].imshow(wordcloud_ALL, interpolation='bilinear')

ax[0].set_title('All Reviews', fontsize=30)

ax[0].axis('off')

ax[1].imshow(wordcloud_1, interpolation='bilinear')

ax[1].set_title('Positive Reviews',fontsize=30)

ax[1].axis('off')

ax[2].imshow(wordcloud_0, interpolation='bilinear')

ax[2].set_title('Negative Reviews',fontsize=30)

ax[2].axis('off')
data['TB.senti'] = data['review'].apply(lambda x: TextBlob(x).sentiment[0])

data['TB.senti'] = np.where(data['TB.senti']>0, '1', '0')



cm = confusion_matrix(data['sentiment'], data['TB.senti'])



results = pd.DataFrame(columns=['Clf.Model', 'Accuracy', 'Sensitivity', 'Specificity', 'ROC_AUC'])

Model1 = 'TextBlob'

accuracy1 = round(accuracy_score(data['sentiment'], data['TB.senti']), 2)

sensitivity1 = round(cm[0,0]/(cm[0,0]+cm[0,1]), 2)

specificity1 = round(cm[1,1]/(cm[1,0]+cm[1,1]), 2)

rocauc1 = round(roc_auc_score(data['sentiment'].astype(int), data['TB.senti'].astype(int)), 2)

print('\n')



results = results.append(pd.Series([Model1, accuracy1, sensitivity1, specificity1, rocauc1], index=results.columns), ignore_index=True)



fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
analyser = SentimentIntensityAnalyzer()



data['V.senti'] = data['review'].apply(lambda x: analyser.polarity_scores(x)["compound"])

data['V.senti'] = np.where(data['V.senti']>0, '1', '0')



cm = confusion_matrix(data['sentiment'], data['V.senti'])



Model1 = 'VADER'

accuracy1 = round(accuracy_score(data['sentiment'], data['V.senti']), 2)

sensitivity1 = round(cm[0,0]/(cm[0,0]+cm[0,1]), 2)

specificity1 = round(cm[1,1]/(cm[1,0]+cm[1,1]), 2)

rocauc1 = round(roc_auc_score(data['sentiment'].astype(int), data['V.senti'].astype(int)), 2)

print('\n')



results = results.append(pd.Series([Model1, accuracy1, sensitivity1, specificity1, rocauc1], index=results.columns), ignore_index=True)



fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
print(results.head())