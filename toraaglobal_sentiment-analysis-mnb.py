# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# split dataset into train test

from sklearn.model_selection import train_test_split



# NB, SVM and text vectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB



# metric

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



from sklearn.pipeline import Pipeline







#viz word cloud

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

% matplotlib inline

def read_data(path):

    '''Read data in .tsv format and return a pandas datafram

    :param : input = .tsv format

    : output : pandas dataframe

    '''

    try:

        print("Reading dataset from --> {}".format(path))

        df = pd.read_csv(path, delimiter='\t')

    except Exception as e:

        print(string(e))

    print('The dimension of the loaded dataset is: {}  \n'.format(df.shape))

    print ("Top 5 rows : {}".format(df.head))

    return df

    



    
trainpath = '../input/train.tsv'

testpath = '../input/test.tsv'



train = read_data(trainpath)

test = read_data(testpath)

train.describe().T
test.describe().T
train.info()
test.info()
train.columns
train.info()
plt.hist(train.Sentiment)
train.Sentiment.plot.kde()
# The first Phrase

text = train.Phrase[0]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
print('The sentiment is : {}'.format(train.Sentiment[0]))
text = " ".join(review for review in train.Phrase)

print ("There are {} words in the combination of all Phrase.".format(len(text)))

plt.figure(figsize=(15,10))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
text = " ".join(phrase for phrase in test.Phrase)

print ("There are {} words in the combination of all Phrase.".format(len(text)))





plt.figure(figsize=(15,10))

# Generate a word cloud image

wordcloud = WordCloud(background_color="white").generate(text)



# Display the generated image:

# the matplotlib way:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Save the image in the img folder:

wordcloud.to_file("./phrases.png")
y = train['Sentiment'].values

X = train['Phrase'].values



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4,random_state=2)



print('The X_train shape : {} '.format(X_train.shape))

print('The y_train shape : {} '.format(y_train.shape))

print('The X_test shape : {} '.format(X_test.shape))

print('The y_test shape : {} '.format(y_test.shape))

unigram_count = CountVectorizer(encoding='latin-1', min_df=5, stop_words=None)
# fit tranform X_train



X_train_vec = unigram_count.fit_transform(X_train)
# check the content

X_train_vec.shape
X_train_vec[0].toarray()
# check the size of the contructed vocabulary

len(unigram_count.vocabulary_)
# print out the entire vocabulary, each row includes the word and its index

#unigram_count.vocabulary_
# check word index in vocabulary

unigram_count.vocabulary_.get('character')
# tranform the test datasets

X_test_vec = unigram_count.transform(X_test)
X_test_vec.shape
# initialize MNB model

mnb = MultinomialNB()



# train the model

mnb.fit(X_train_vec, y_train)
# test the classifier

mnb.score(X_test_vec, y_test)
# confusion matrix

y_pred= mnb.fit(X_train_vec, y_train).predict(X_test_vec)

cm = confusion_matrix(y_test,y_pred)

print(cm)

temp = train.copy()

temp.shape
'''

The sentiment labels are:



0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive



'''



def f(col):

    if col['Sentiment'] == 0:

        val = 'negative'

    elif col['Sentiment'] == 1:

        val = 'somewhat negative'

    elif col['Sentiment'] == 2:

        val = 'neutral'

    elif col['Sentiment'] == 3:

        val = 'somewhat positive'

    else:

        val = 'positive'

        

    return val

        
temp['target_name'] = temp.apply(f, axis=1)

temp.shape
class_names = temp.target_name



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax





np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
precision_score(y_test, y_pred, average=None)

recall_score(y_test, y_pred, average=None)
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

unigram_count.vocabulary_.get('worthless')

for i in range(0,5):

    print (mnb.coef_[i][unigram_count.vocabulary_.get('worthless')])
# sort the conditional probability for category 0 "very negative"

feature_ranks = sorted(zip(mnb.coef_[0], unigram_count.get_feature_names()))
#feature_ranks
## find the calculated posterior probability

mnb.predict_proba(X_train_vec)

########## submit to Kaggle submission



kaggle_ids=test['PhraseId'].values

kaggle_X=test['Phrase'].values



# vectorize the test examples using the vocabulary fitted from the 60% training data



kaggle_X_vec=unigram_count.transform(kaggle_X)



kaggle_pred = mnb.fit(X_train_vec, y_train).predict(kaggle_X_vec)



# combine the test example ids with their predictions



#kaggle_submission=zip(kaggle_ids, kaggle_pred)





# prepare output file



#outf=open('./kaggle_submission.csv', 'w')



# write header



#outf.write('PhraseId,Sentiment\n')





# write predictions with ids to the output file



#for x, value in enumerate(kaggle_submission): outf.write(str(value[0]) + ',' + str(value[1]) + '\n')



# close the output file





os.listdir('./')
my_submission = pd.DataFrame({'PhraseId': kaggle_ids, 'Sentiment': kaggle_X_vec})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)