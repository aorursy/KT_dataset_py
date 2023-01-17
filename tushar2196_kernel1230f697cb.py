import tensorflow as tf

import numpy as np

import pandas as pd

import os



from tensorflow import keras
# loading the captions

def load_captions_file(filename):

    file = open(filename, 'r')

    text = file.read()

    return text
filename = '../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'

captions = load_captions_file(filename)
# extracting the photo descriptions

def get_descriptions(captions):

    mapping = dict()

    for line in captions.split('\n'):

        tokens = line.split()

        if len(line) < 2:

            continue

        img_id, img_desc = tokens[0], tokens[1:]

        # removing the .jpg extension

        img_id = img_id.split('.')[0]

        img_desc = ' '.join(img_desc)

        # creating list and storing the descriptions

        if img_id not in mapping:

            mapping[img_id] = list()

        mapping[img_id].append(img_desc)

        

    return mapping

    



descriptions = get_descriptions(captions)

print('No. of photo descriptions: ', len(descriptions))

# printing the description for one photo

for key in descriptions:

    print ("key: , value: ", (key, descriptions[key]))

    break
import string



def clean_text(descriptions):

    for key, all_desc in descriptions.items():

        for i in range(len(all_desc)):

            desc = all_desc[i]

            # separating by white space

            desc = desc.split()

            # converting to lower case

            desc = [word.lower() for word in desc]

            # removing punctuation

            desc = [text_original.translate(str.maketrans('','',string.punctuation))

                   for text_original in desc]

            # removing single characters

            desc = [word for word in desc if len(word) > 1]

            # removing words with numbers in them

            desc = [word for word in desc if word.isalpha()]

            # joining

            all_desc[i] = ' '.join(desc)

            

clean_text(descriptions)



# chekcing whether cleaned or not

for key in descriptions:

    print ("key: , value: ", (key, descriptions[key]))

    break
def get_vocabulary(descriptions):

    all_desc = set()

    for key in descriptions.keys():

        [all_desc.update(d.split()) for d in descriptions[key]]

    return all_desc



vocabulary = get_vocabulary(descriptions)

print('Vocabulary size: ', len(vocabulary))
def save_to_file(descriptions, filename):

    lines = list()

    

    for key, all_desc in descriptions.items():

        for desc in all_desc:

            lines.append(key + ' ' + desc)

            only_desc.append(desc)

        

        text = '\n'.join(lines)

        file = open(filename, 'w')

        file.write(text)

        file.close()



only_desc = [] # a list containing only the descriptions

save_to_file(descriptions, 'descriptions.txt')
import matplotlib.pyplot as plt



from wordcloud import WordCloud



# converting `only_desc` list to string

only_desc_str = ''

only_desc_str = only_desc_str.join(only_desc)



plt.figure(dpi=200)

wordcloud = WordCloud(width=800, height=800, margin=0, background_color="black").generate(only_desc_str)

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
from pickle import dump

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import Model



# extracting the features from each photo

def get_features(directory):

    #model = VGG16()

    model = VGG16(include_top=True,weights=None)

    ## load the locally saved weights 

    model.load_weights("../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5")

    # removing the top layer

    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    print(model.summary())

    # extracting the features

    features = dict()

    for name in os.listdir(directory):

        filename = directory + '/' + name

        image = load_img(filename, target_size=(224, 224))

        # coverting the image's pixels to array

        image = img_to_array(image)

        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image = preprocess_input(image)

        # getting the features

        feature = model.predict(image)

        # getting image id

        img_id = name.split('.')[0]

        features[img_id] = feature

    return features



# executing the function

#directory = '../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Images'

#features = get_features(directory)

#print('Extracted ', len(features), 'features')

# saving the features into a .pkl file

#dump(features, open('features.pkl', 'wb'))
# function for loading documents into memory

def load_doc(filename):

    file = open(filename, 'r')

    text = file.read()

    file.close()

    return text
# loading photo identifiers

def load_set(filename):

    doc = load_doc(filename)

    dataset = list()

    for line in doc.split('\n'):

        if len(line) < 1:

            continue

        # image id

        identifier = line.split('.')[0]

        dataset.append(identifier)

    return set(dataset)
def load_clean_desc(filename, dataset):

    doc = load_doc(filename)

    descriptions = dict()

    for line in doc.split('\n'):

        tokens = line.split()

        img_id, img_desc = tokens[0], tokens[1:]

        # skipping images that are not in the set

        if img_id in dataset:

            if img_id not in descriptions:

                descriptions[img_id] = list()

            desc = 'start' + ' '.join(img_desc) + ' end'            

            descriptions[img_id].append(desc)

    return descriptions
# loading the image features

def load_image_features(filename, dataset):

    all_features = load(open(filename, 'rb'))

    features = {k: all_features[k] for k in dataset}

    return features
# converting the dictionary of descriptions to a list

def to_list(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc



# creating unique integers for the tokens

def create_tokenizer(descriptions):

    lines = to_list(descriptions)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
# creating sequences for input images, encoded text and ouput words

def create_sequences(tokenizer, max_length, desc_list, photos):

    X1, X2, y = list(), list(), list()

    for desc in desc_list:

        # encoding

        seq = tokenizer.texts_to_sequences([desc])[0]

        # splitting the sequence

        for i in range(1, len(seq)):

            # to input and output pairs

            in_seq, out_seq = seq[:i], seq[i]

            # padding the input sequence

            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

            # one-hot encoding the output

            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                

            X1.append(photos)

            X2.append(in_seq)

            y.append(out_seq)

                

    return array(X1), array(X2), array(y)
# calculating the maximum length description

def max_length(descriptions):

    lines = to_list(descriptions)

    return max(len(d.split()) for d in lines)
# defining the captioning model

def define_model(vocab_size, max_length):

    # feature extractor model

    inputs1 = Input(shape=(4096,))

    fe1 = Dropout(0.5)(inputs1)

    fe2 = Dense(256, activation='relu')(fe1)

    # sequence model

    inputs2 = Input(shape=(max_length,))

    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)

    se2 = Dropout(0.5)(se1)

    se3 = LSTM(256)(se2)

    # decoder model

    decoder1 = add([fe2, se3])

    decoder2 = Dense(256, activation='relu',)(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    

    # merging and compiling

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    

    # summarizing

    print(model.summary())

    # creates a plot and saves it to file

    plot_model(model, to_file='model.png', show_shapes=True)

    return model
# this function is for memory managed fitting (less than 32GB of RAM)

def data_generator(descriptions, photos, tokenizer, max_length):

    while 1:

        for key, desc_list in descriptions.items():

            # getting the image features

            photo = photos[key][0]

            in_img, in_seq, out_word = create_sequences(tokenizer, 

                                                       max_length, 

                                                       desc_list, 

                                                       photo)

            yield [[in_img, in_seq], out_word]
from pickle import load

from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Input

from keras.layers import Embedding

from keras.layers import Dropout

from keras.layers.merge import add

from keras.callbacks import ModelCheckpoint



# loading the training set

filename = '../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'

train = load_set(filename)

print('Trainset: ', len(train))

# loading the clean descriptions

train_descriptions = load_clean_desc('descriptions.txt', train)

print('Descriptions for train: ', len(train_descriptions))

# loading the photo features

train_features = load_image_features('../input/features1/features.pkl', train)

print('Images in trainset: ', len(train_features))





tokenizer = create_tokenizer(train_descriptions)

vocab_size = len(tokenizer.word_index) + 1

print('Tokenized Vocab size: ', vocab_size)



# calling the `max_length()` function

max_length = max_length(train_descriptions)

print('Description Length: %d' % max_length)
# defining the model

model = define_model(vocab_size, max_length)



# training and saving models after each epoch

epochs = 12

steps = len(train_descriptions)

for i in range(epochs):

    generator = data_generator(train_descriptions, 

                               train_features, 

                               tokenizer, 

                               max_length)

    # fitting for one epoch

    model.fit_generator(generator, epochs=1, 

                       steps_per_epoch=steps, verbose=1)

    # saving the model

    model.save('model_' + str(i) + '.h5')
# map an integer to a word (to be called by `generate_desc`)

def word_for_id(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:

            return word

    return None



# generate a description for an image

def generate_desc(model, tokenizer, photo, max_length):

    # start the process

    in_text = 'start'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)

        # predicting the next word

        yhat = model.predict([photo, sequence], verbose=0)

        # covert probability to integer

        yhat = argmax(yhat)

        # map integer to word

        word = word_for_id(yhat, tokenizer)

        # if unable to map word

        if word is None:

            break

        # append as input for generating the next word

        in_text += ' ' + word

        if word == 'end':

            break

    return in_text
# evaluate the model

def evaluate_model(model, descriptions, photos, tokenizer, max_length):

    actual, predicted = list(), list()

    for key, desc_list in descriptions.items():

        # generate description

        yhat = generate_desc(model, tokenizer, photos[key], max_length)

        # storing the actual and predicted text

        references = [d.split() for d in desc_list]

        actual.append(references)

        predicted.append(yhat.split())

    

    # calculate the BLEU score

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
from numpy import argmax

from nltk.translate.bleu_score import corpus_bleu



# loading test set

filename = '../input/flickr8k-sau/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

test = load_set(filename)

print('Dataset: %d' % len(test))

# loading descriptions

test_descriptions = load_clean_desc('descriptions.txt', test)

print('Descriptions for test: %d' % len(test_descriptions))

# loading photo features

test_features = load_image_features('../input/features1/features.pkl', test)

print('Images in test: %d' % len(test_features))



# load the model

filename = 'model_11.h5'

# evaluate model

evaluate_model(model, test_descriptions, test_features,

              tokenizer, max_length)
# load the doc into memory

def load_doc(filename):

    file = open(filename, 'r')

    text = file.read()

    file.close()

    return text
# load a pre-defined list of photo identifiers

def load_set(filename):

    doc = load_doc(filename)

    dataset = list()

    for line in doc.split('\n'):

        # skip empty lines

        if len(line) < 1:

            continue

        # image identifier

        identifier = line.split('.')[0]

        dataset.append(identifier)

    return set(dataset)
# load clean descriptions into memory

def load_clean_desc(filename, dataset):

    doc = load_doc(filename)

    descriptions = dict()

    for line in doc.split('\n'):

        tokens = line.split()

        image_id, image_desc = tokens[0], tokens[1:]

        # skip images not in the set

        if image_id in dataset:

            if image_id not in descriptions:

                descriptions[image_id] = list()

            # wrap description in tokens

            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            # store

            descriptions[image_id].append(desc)

    return descriptions
def to_list(descriptions):

    all_desc = list()

    for key in descriptions.keys():

        [all_desc.append(d) for d in descriptions[key]]

    return all_desc
# fit a tokenizer given caption descriptions

def create_tokenizer(descriptions):

    lines = to_list(descriptions)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer
# load set

filename = 'Data/Flickr_8k.trainImages.txt'

train = load_set(filename)

print('Dataset: %d' % len(train))

# load descriptions

train_descriptions = load_clean_desc('descriptions.txt', train)

print('Descriptions: train=%d' % len(train_descriptions))

# prepare tokenizer

tokenizer = create_tokenizer(train_descriptions)

# save the tokenizer

dump(tokenizer, open('tokenizer.pkl', 'wb'))
from pickle import load

from numpy import argmax

from keras.preprocessing.sequence import pad_sequences

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.models import Model

from keras.models import load_model



# extract features from each photo in the directory

def extract_features(filename):

    # load the model

    model = VGG16()

    # re-structure the model

    model.layers.pop()

    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # load the photo

    image = load_img(filename, target_size=(224, 224))

    # convert the image pixels to a numpy array

    image = img_to_array(image)

    # reshape data for the model

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model

    image = preprocess_input(image)

    # get features

    feature = model.predict(image, verbose=0)

    return feature



# map an integer to a word

def word_for_id(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:

            return word

    return None



# generate a description for an image

def generate_desc(model, tokenizer, photo, max_length):

    # seed the generation process

    in_text = 'start'

    # iterate over the whole length of the sequence

    for i in range(max_length):

        # integer encode input sequence

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # pad input

        sequence = pad_sequences([sequence], maxlen=max_length)

        # predict next word

        yhat = model.predict([photo,sequence], verbose=0)

        # convert probability to integer

        yhat = argmax(yhat)

        # map integer to word

        word = word_for_id(yhat, tokenizer)

        # stop if we cannot map the word

        if word is None:

            break

        # append as input for generating the next word

        in_text += ' ' + word

        # stop if we predict the end of the sequence

        if word == 'end':

            break

    return in_text



# load the tokenizer

tokenizer = load(open('tokenizer.pkl', 'rb'))

# pre-define the max sequence length (from training)

max_length = 33

# load the model

model = load_model('model_19.h5')

# load and prepare the photograph

photo = extract_features('example.jpg')

# generate description

description = generate_desc(model, tokenizer, photo, max_length)

print(description)