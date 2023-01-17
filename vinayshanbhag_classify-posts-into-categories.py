import pandas as pd

import re

import numpy as np

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline



import itertools

from sklearn.metrics import confusion_matrix



def normalize(cnf):

    """

    Returns normalized confusion matrix

    """

    return cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]



def plot_confusion_matrix(cnf, 

                          labels,

                          norm=False,

                          title='Confusion matrix',

                          colorbar=True,

                          cmap=plt.cm.OrRd,

                          xlabel='Predicted Labels',

                          ylabel='True Labels',

                          precision=1

                         ):

    """

    adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting norm=True.

    """

    #plt.figure()

    plt.figure(figsize=(8, 8))

    if norm: 

        cnf = normalize(cnf)

        

    art = plt.imshow(cnf, interpolation='nearest', cmap=cmap)

    plt.title(title,fontsize=10)

    if colorbar:

        plt.colorbar(art,shrink=0.5).ax.tick_params(labelsize=7) 

    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels, rotation=90,fontsize=8)

    plt.yticks(tick_marks, labels,fontsize=8)



    thresh = cnf.max() / 2.

    for i, j in itertools.product(range(cnf.shape[0]), repeat=2):

        if cnf.dtype=='float':

            plt.text(j, i, '{0:0.{1}f}'.format(cnf[i, j],precision),

                 horizontalalignment="center",

                 verticalalignment="center",    

                 fontsize=5,

                 color="white" if cnf[i, j] > thresh else "black")

        else:

            plt.text(j, i, cnf[i, j],

                 horizontalalignment="center",

                 verticalalignment="center",    

                 fontsize=5,

                 color="white" if cnf[i, j] > thresh else "black")



    

    plt.ylabel(ylabel,fontsize=9)

    plt.xlabel(xlabel,fontsize=9)

    plt.tight_layout()

    plt.show()
df = pd.read_csv('../input/techcrunch_posts.csv')

df.head(3)
df = df[(df.category.notnull() & df.content.notnull())]

df.category.value_counts()
min_samples=100

category_counts = df.category.value_counts()

df = df[df.category.isin(category_counts[category_counts > min_samples].index)]

df.content = df.content.apply(lambda x: ' '.join(re.sub(r'[^0-9a-zA-Z]',' ',str(x)).split()))
labels = sorted(set(df.category.values))

print('%25s: %s\n%s'%('Category','ID','-'*29))

for idx,label in enumerate(labels):

    print('%25s: %-d'%(label,idx))
df['category_id'] = df.category.apply(lambda x: int(labels.index(x)))
df = df.reindex(np.random.permutation(df.index))
t_index = int(len(df)*0.6)

X_train = df.content.values[:t_index]

y_train = df.category_id.values[:t_index]



X_test = df.content.values[t_index:]

y_test = df.category_id.values[t_index:]



df_train = df.iloc[:t_index]

df_test = df.iloc[t_index:]
classifier = Pipeline([

    ('vect', CountVectorizer(stop_words='english',max_df=0.5,ngram_range=(1,3),max_features=10000)),

    ('tfidf', TfidfTransformer(sublinear_tf=True,norm='l2',use_idf=True)),

    ('clf', LinearSVC(loss='squared_hinge',multi_class='ovr',dual=True,tol=0.001,class_weight='balanced')),

])

_ = classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

plot_confusion_matrix(confusion_matrix(y_test,predicted), 

                      labels=labels, 

                      norm=True,

                      title='Normalized confusion matrix',

                      xlabel='Predicted Category',

                      ylabel='True Category',

                      precision=1

                     )