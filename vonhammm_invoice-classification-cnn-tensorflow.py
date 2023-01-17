import os

import pandas as pd 

import numpy as np

import tensorflow.compat.v1 as tf

import seaborn as sns

import matplotlib.pyplot as plt

import re



from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk import pos_tag

from nltk.corpus import wordnet



import sys

import warnings

import random



if not sys.warnoptions:

    warnings.simplefilter("ignore")

tf.disable_v2_behavior()
df_train_csv=pd.read_csv('../input/predict-product-category-from-given-invoice/Dataset/Train.csv')

df_test_csv=pd.read_csv('../input/predict-product-category-from-given-invoice/Dataset/Test.csv')
print(df_train_csv.shape)

print(df_test_csv.shape)

df_train_csv.head(1)
df_test_csv.head(1)
def category_statistic(dataset):

    """

    Check category distribution .

    """

    category_count = dataset['Product_Category'].value_counts()

    sns.set(style="darkgrid")

    sns.set(rc={'figure.figsize':(10,5)})

    sns.barplot( category_count.index, category_count.values, alpha=0.9)

    plt.title('Frequency Distribution of category (train set)',fontsize=14)

    plt.ylabel('Number of Occurrences', fontsize=14)

    plt.xticks(size='small',rotation=90,fontsize=6)

    plt.xlabel('Category', fontsize=14)

    plt.show()



df_csv = df_train_csv

category_statistic(df_csv)
cat_sample_max = max(df_csv['Product_Category'].value_counts())

delete_cate = ['CLASS-1248','CLASS-1688','CLASS-2015','CLASS-2146','CLASS-1957','CLASS-1838','CLASS-1567','CLASS-1919','CLASS-1850',

               'CLASS-2112','CLASS-1477','CLASS-2241','CLASS-1870','CLASS-1429','CLASS-2003','CLASS-1309','CLASS-1964','CLASS-1322',

               'CLASS-1294','CLASS-1770','CLASS-1983','CLASS-1652','CLASS-1867','CLASS-2038','CLASS-1805','CLASS-2152']



for l in  delete_cate:

    df_csv.drop(df_csv.loc[df_csv['Product_Category']==l].index, inplace=True)



all_text = list(df_csv['Item_Description'][...])

all_cate = np.array(df_csv['Product_Category'].tolist())
def clean_str(string):

    """

    String cleaning .

    """

    string = re.sub(r"[^A-Za-z0-9]", " ", string) # remove unused charactor other than english letter and number, use space to replace

    return string.strip()                         # delete the first and last space



def get_wordnet_pos(treebank_tag):

    """

    Return the POS of each word for later usage .

    """

    if treebank_tag.startswith('J'):

        return wordnet.ADJ

    elif treebank_tag.startswith('V'):

        return wordnet.VERB

    elif treebank_tag.startswith('N'):

        return wordnet.NOUN

    elif treebank_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN   # no need to change 

def text_lemmatization(l,lemmatizer,t_text, word_count):

    """

    Tokenization: Split the text into words. 

    Lemmatize the text .

    """

    for i in range(len(t_text)):

        text_clean = clean_str(t_text[i])     # clean texts, remove useless symbols

        text_word = word_tokenize(text_clean) # set each individual token

        text_pos = pos_tag(text_word)         # pos tagging each token [word,POS]

        text_lemma = ""

        for item in text_pos:                 # lemmatizing each token

            # Put each word after lemmatization into the list

            text_lemma = text_lemma + " " + (lemmatizer.lemmatize(item[0],get_wordnet_pos(item[1])))     

        l.append(text_lemma.strip())          # append the preprocessed sample to x_train list, remove the space

        word_count.append(len(text_pos))

    return l, word_count   
# preprocessing

text = list()

word_count = list()                   #statistics how many words in each sample

lemmatizer = WordNetLemmatizer()      # model used to lemmatize word (defined by package nltk)

text_lemmatized, word_count = text_lemmatization(text,lemmatizer,all_text, word_count)
class Indexer:

    # Tokenizer

    def __init__(self):

        self.counter = 1

        self.d = {"<unk>": 0}

        self.rev = {}

        self._lock = False

        self.word_count = {}

        self.rev_d = {}

        self.rev[0] = "unk"

    def convert(self, w):

        if w not in self.d:

            if self._lock:

                return self.d["<unk>"]

            self.d[w] = self.counter

            self.rev[self.counter] = w

            self.counter += 1

            self.word_count[self.d[w]] = 0

        self.word_count[self.d[w]] = self.word_count[self.d[w]] + 1

        return self.d[w]

    def convertback(self, w):

        return self.rev[w]

    def lock(self):

        self._lock = True

all_data = []

split_data = []

tokenizer = Indexer()

max_len_sent = 0

num_of_padding = 0

vocabulary = 0

max_index = 0

for i, t in enumerate(text_lemmatized):

    current_convert = [tokenizer.convert(w) for w in t.split()]

    m = max(current_convert)

    max_index = max(max_index,max(current_convert))

    max_len_sent = max(max_len_sent, len(current_convert))

    split_data.append(current_convert)

vocabulary_size = max_index + 1    

    

    

for i, t in enumerate(split_data):  

    if (len(t) < max_len_sent):

        num_of_padding = max_len_sent - len(t)

        for n in range(0,num_of_padding):

            t.append(0)        

    all_data.append(t)





text_df = pd.DataFrame(all_data)

cate_df = pd.DataFrame(df_csv['Product_Category'])

df_text_cate =  text_df 

df_text_cate['category'] = 'cat'

for i in range(0,len(cate_df)):

    df_text_cate.at[i,'category'] = cate_df.iloc[i]['Product_Category']



labels =  list(df_text_cate.category.unique())

for l in labels:

    df_text_cate[l] = 0

for i in range(0,len(df_text_cate)):

    cat = df_text_cate.iloc[i]['category']

    df_text_cate.at[i,cat] = 1  


def oversampling_and_split_dataset(df,trainingratio,l):

    

    trainingratio = 0.7

    index_list = {}

    number_of_samples_cat = {}

    training_samples = {}

    train_random_indices = {}

    duplicated_train_indices = {}

    for l in labels:

        index_list[l] = np.array(df[df.category == l].index)

        number_of_samples_cat[l] = len(index_list[l]) 

        training_samples[l] = round(number_of_samples_cat[l]*trainingratio)  

        train_random_indices[l] = np.random.choice(index_list[l], training_samples[l], replace = False)

        if (l == 'CLASS-1758'): 

            duplicated_train_indices[l] = train_random_indices[l]

        else:

            baseline = int(round(cat_sample_max*trainingratio))

            duplicated_train_indices[l] = np.random.choice(train_random_indices[l], baseline , replace = True)

            duplicated_train_indices[l] = np.array(duplicated_train_indices[l])  

    

    train_indices = duplicated_train_indices['CLASS-1249']

    for l in labels:

        if (l != 'CLASS-1249'): 

            train_indices  = np.concatenate([train_indices, duplicated_train_indices[l]])

   

    training_data = df.iloc[train_indices,:]      

    testing_data = df.drop(train_indices,axis=0)  



    #shuffle the data

    training_data=training_data.sample(frac=1).reset_index(drop=True)



    split = int(len(testing_data)/2)

    train_y = training_data.iloc[:,27:57]

    train_x = training_data.iloc[:,0:26] 

    

    valid_y = testing_data.iloc[0:split,27:57]

    valid_x = testing_data.iloc[0:split,0:26]  

    

    test_y = testing_data.iloc[split:,27:57]

    test_x = testing_data.iloc[split:,0:26] 

    

    return train_x, train_y, valid_x, valid_y, test_x, test_y

    



train_ratio = 0.7

training_x, training_y, validation_x, validation_y, testing_x, testing_y = oversampling_and_split_dataset(df_text_cate,train_ratio,labels)

# Model Hyperparameters

embedding_dim = 50

filter_sizes = "3,4,5"

num_filters = 100

dropout_keep_prob = 0.5

l2_reg_lambda = 0.0



# Training parameters

batch_size = 64

num_epochs = 80

num_batches =len(training_x)/64 


class TextCNN(object):

    """

    A CNN for text classification.

    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.

    """

    def __init__(

      self, sequence_length, num_classes, vocab_size,

      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):



        # Placeholders for input, output and dropout

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")



        # Keeping track of l2 regularization loss (optional)

        l2_loss = tf.constant(0.0)



        # Embedding layer

        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")  # We initialize embedding matrix using a random uniform distribution.

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)                        

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) 



        # Create a convolution + maxpool layer for each filter size

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes): 

            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer

                filter_shape = [filter_size, embedding_size, 1, num_filters] 

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")  

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")  

                # Each filter slides over the whole embedding, but varies in how many words it covers

                conv = tf.nn.conv2d(

                    self.embedded_chars_expanded,  

                    W,                                  

                    strides=[1, 1, 1, 1],           

                    padding="VALID",

                    name="conv")

                # h is the result of applying the nonlinearity to the convolution output. 

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 

                # Maxpooling over the outputs

                pooled = tf.nn.max_pool(

                    h,

                    ksize=[1, sequence_length - filter_size + 1, 1, 1],  

                    strides=[1, 1, 1, 1],

                    padding='VALID',

                    name="pool")

                pooled_outputs.append(pooled)



        # Combine all the pooled features

        num_filters_total = num_filters * len(filter_sizes)

        #  Once we have all the pooled output tensors from each filter size we combine them into one long feature vector of shape [batch_size, num_filters_total]. 

        self.h_pool = tf.concat(pooled_outputs, 3)

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])



        # Add dropout

        with tf.name_scope("dropout"):

            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)



        # Final (unnormalized) scores and predictions

        with tf.name_scope("output"):

            W = tf.get_variable(

                "W",

                shape=[num_filters_total, num_classes])

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)

            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

            self.predictions = tf.argmax(self.scores, 1, name="predictions")



        # Calculate mean cross-entropy loss

        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss



        # Accuracy

        with tf.name_scope("accuracy"):

            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



sess = tf.Session()

with sess.as_default():

    cnn = TextCNN(

        sequence_length=training_x.shape[1],

        num_classes=training_y.shape[1],

        vocab_size=vocabulary_size,

        embedding_size=embedding_dim,

        filter_sizes=list(map(int, filter_sizes.split(","))),

        num_filters=num_filters,

        l2_reg_lambda=l2_reg_lambda)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(1e-3)

    grads_and_vars = optimizer.compute_gradients(cnn.loss)

    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



    # Summaries for loss and accuracy

    loss_summary = tf.summary.scalar("loss", cnn.loss)

    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)



    # Initialize all variables

    sess.run(tf.global_variables_initializer())



    for epoch in range(int(num_epochs)) :     

        for i in range(int(num_batches)) :

            off_1 = i * batch_size

            off_2 = i * batch_size + batch_size

            batch_x = training_x[off_1:off_2]

            batch_y = training_y[off_1:off_2]

            batch_x = np.asarray(batch_x)

            batch_y = np.asarray(batch_y)

            

            batch_val_x = np.asarray(validation_x)

            batch_val_y = np.asarray(validation_y)

            

        feed_dict_train = {

                  cnn.input_x: batch_x,

                  cnn.input_y: batch_y,

                  cnn.dropout_keep_prob: dropout_keep_prob

                }

        

        feed_dict_valid = {

                  cnn.input_x: batch_val_x,

                  cnn.input_y: batch_val_y,

                  cnn.dropout_keep_prob: 1.0

                }        

        _, step, train_loss, train_accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict_train)

        valid_loss, valid_accuracy= sess.run([cnn.loss, cnn.accuracy], feed_dict_valid)  

        if epoch % 5 == 0:

            print("Epoch{}: train_loss {:g}, train_acc {:g}, valid_loss {:g}, valid_acc {:g},".format(epoch, train_loss, train_accuracy, valid_loss, valid_accuracy))



    feed_dict_test = {

                  cnn.input_x: testing_x,

                  cnn.input_y: testing_y,

                  cnn.dropout_keep_prob: 1.0

                }

    accuracy = sess.run([cnn.accuracy],feed_dict_test)

    print("The final testing accuracy is:")

    print(accuracy)
