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
#https://illinois.edu/fb/sec/1713398

from os import listdir

from os import path

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input



def load_photos(directory):

	images = dict()

	for name in listdir(directory):

		# load an image from file

		filename = path.join(directory, name)

		image = load_img(filename, target_size=(224, 224))

		# convert the image pixels to a numpy array

		image = img_to_array(image)

		# reshape data for the model

		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

		# prepare the image for the VGG model

		image = preprocess_input(image)

		# get image id

		image_id = name.split('.')[0]

		images[image_id] = image

	return images



# load images

directory = '/kaggle/input/flicker8k-dataset/flickr8k_dataset/Flicker8k_Dataset'

images = load_photos(directory)

print('Loaded Images: %d' % len(images))
from os import listdir

from os import path

from pickle import dump

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.layers import Input



# extract features from each photo in the directory

def extract_features(directory):

	# load the model

	in_layer = Input(shape=(224, 224, 3))

	model = VGG16(include_top=False, input_tensor=in_layer)

	model.summary()

	# extract features from each photo

	features = dict()

	for name in listdir(directory):

		# load an image from file

		filename = path.join(directory, name)

		image = load_img(filename, target_size=(224, 224))

		# convert the image pixels to a numpy array

		image = img_to_array(image)

		# reshape data for the model

		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

		# prepare the image for the VGG model

		image = preprocess_input(image)

		# get features

		feature = model.predict(image, verbose=0)

		# get image id

		image_id = name.split('.')[0]

		# store feature

		features[image_id] = feature

		print('>%s' % name)

	return features



# extract features from all images

directory = '/kaggle/input/flicker8k-dataset/flickr8k_dataset/Flicker8k_Dataset'

features = extract_features(directory)

print('Extracted Features: %d' % len(features))

# save to file

dump(features, open('features.pkl', 'wb'))
# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# extract descriptions for images

def load_descriptions(doc):

	mapping = dict()

	# process lines

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		if len(line) < 2:

			continue

		# take the first token as the image id, the rest as the description

		image_id, image_desc = tokens[0], tokens[1:]

		# remove filename from image id

		image_id = image_id.split('.')[0]

		# convert description tokens back to string

		image_desc = ' '.join(image_desc)

		# store the first description for each image

		if image_id not in mapping:

			mapping[image_id] = image_desc

	return mapping



filename = '/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr8k.token.txt'

doc = load_doc(filename)

descriptions = load_descriptions(doc)

print('Loaded: %d ' % len(descriptions))
import string

import re



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# extract descriptions for images

def load_descriptions(doc):

	mapping = dict()

	# process lines

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		if len(line) < 2:

			continue

		# take the first token as the image id, the rest as the description

		image_id, image_desc = tokens[0], tokens[1:]

		# remove filename from image id

		image_id = image_id.split('.')[0]

		# convert description tokens back to string

		image_desc = ' '.join(image_desc)

		# store the first description for each image

		if image_id not in mapping:

			mapping[image_id] = image_desc

	return mapping



# clean description text

def clean_descriptions(descriptions):

	# prepare regex for char filtering

	re_punc = re.compile('[%s]' % re.escape(string.punctuation))

	for key, desc in descriptions.items():

		# tokenize

		desc = desc.split()

		# convert to lower case

		desc = [word.lower() for word in desc]

		# remove punctuation from each word

		desc = [re_punc.sub('', w) for w in desc]

		# remove hanging 's' and 'a'

		desc = [word for word in desc if len(word)>1]

		# store as string

		descriptions[key] =  ' '.join(desc)



# save descriptions to file, one per line

def save_doc(descriptions, filename):

	lines = list()

	for key, desc in descriptions.items():

		lines.append(key + ' ' + desc)

	data = '\n'.join(lines)

	file = open(filename, 'w')

	file.write(data)

	file.close()



filename = '/kaggle/input/flicker8k-dataset/Flickr8k_text/Flickr8k.token.txt'

# load descriptions

doc = load_doc(filename)

# parse descriptions

descriptions = load_descriptions(doc)

print('Loaded: %d ' % len(descriptions))

# clean descriptions

clean_descriptions(descriptions)

# summarize vocabulary

all_tokens = ' '.join(descriptions.values()).split()

vocabulary = set(all_tokens)

print('Vocabulary Size: %d' % len(vocabulary))

# save descriptions

save_doc(descriptions, 'descriptions.txt')
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# load clean descriptions into memory

def load_clean_descriptions(filename):

	doc = load_doc(filename)

	descriptions = dict()

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		# split id from description

		image_id, image_desc = tokens[0], tokens[1:]

		# store

		descriptions[image_id] = ' '.join(image_desc)

	return descriptions



descriptions = load_clean_descriptions('descriptions.txt')

print('Loaded %d' % (len(descriptions)))

# extract all text

desc_text = list(descriptions.values())

# prepare tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(desc_text)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# integer encode descriptions

sequences = tokenizer.texts_to_sequences(desc_text)

# pad all sequences to a fixed length

max_length = max(len(s) for s in sequences)

print('Description Length: %d' % max_length)

padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# one hot encode

y = to_categorical(padded, num_classes=vocab_size)

y = y.reshape((len(descriptions), max_length, vocab_size))

print(y.shape)
from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# load clean descriptions into memory

def load_clean_descriptions(filename):

	doc = load_doc(filename)

	descriptions = dict()

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		# split id from description

		image_id, image_desc = tokens[0], tokens[1:]

		# store

		descriptions[image_id] = ' '.join(image_desc)

	return descriptions



descriptions = load_clean_descriptions('descriptions.txt')

print('Loaded %d' % (len(descriptions)))

# extract all text

desc_text = list(descriptions.values())

# prepare tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(desc_text)

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

# integer encode descriptions

sequences = tokenizer.texts_to_sequences(desc_text)

# determine the maximum sequence length

max_length = max(len(s) for s in sequences)

print('Description Length: %d' % max_length)



X, y = list(), list()

for img_no, seq in enumerate(sequences):

	# split one sequence into multiple X,y pairs

	for i in range(1, len(seq)):

		# split into input and output pair

		in_seq, out_seq = seq[:i], seq[i]

		# pad input sequence

		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

		# encode output sequence

		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

		# store

		X.append(in_seq)

		y.append(out_seq)



# convert to numpy arrays

X, y = array(X), array(y)

print(X.shape)

print(y.shape)
from os import listdir

from os import path

from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input



# load doc into memory

def load_doc(filename):

	# open the file as read only

	file = open(filename, 'r')

	# read all text

	text = file.read()

	# close the file

	file.close()

	return text



# load clean descriptions into memory

def load_clean_descriptions(filename):

	doc = load_doc(filename)

	descriptions = dict()

	for line in doc.split('\n'):

		# split line by white space

		tokens = line.split()

		# split id from description

		image_id, image_desc = tokens[0], tokens[1:]

		# store

		descriptions[image_id] = ' '.join(image_desc)

	return descriptions



# fit a tokenizer given caption descriptions

def create_tokenizer(descriptions):

	lines = list(descriptions.values())

	tokenizer = Tokenizer()

	tokenizer.fit_on_texts(lines)

	return tokenizer



# load a single photo intended as input for the VGG feature extractor model

def load_photo(filename):

	image = load_img(filename, target_size=(224, 224))

	# convert the image pixels to a numpy array

	image = img_to_array(image)

	# reshape data for the model

	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

	# prepare the image for the VGG model

	image = preprocess_input(image)[0]

	# get image id

	image_id = path.basename(filename).split('.')[0]

	return image, image_id



# create sequences of images, input sequences and output words for an image

def create_sequences(tokenizer, max_length, desc, image):

	Ximages, XSeq, y = list(), list(),list()

	vocab_size = len(tokenizer.word_index) + 1

	# integer encode the description

	seq = tokenizer.texts_to_sequences([desc])[0]

	# split one sequence into multiple X,y pairs

	for i in range(1, len(seq)):

		# select

		in_seq, out_seq = seq[:i], seq[i]

		# pad input sequence

		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

		# encode output sequence

		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

		# store

		Ximages.append(image)

		XSeq.append(in_seq)

		y.append(out_seq)

	Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)

	return [Ximages, XSeq, y]



# data generator, intended to be used in a call to model.fit_generator()

def data_generator(descriptions, tokenizer, max_length):

	# loop for ever over images

	directory = '/kaggle/input/flicker8k-dataset/flickr8k_dataset/Flicker8k_Dataset'

	while 1:

		for name in listdir(directory):

			# load an image from file

			filename = path.join(directory, name)

			image, image_id = load_photo(filename)

			# create word sequences

			desc = descriptions[image_id]

			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)

			yield [[in_img, in_seq], out_word]



# load mapping of ids to descriptions

descriptions = load_clean_descriptions('descriptions.txt')

# integer encode sequences of words

tokenizer = create_tokenizer(descriptions)

# pad to fixed length

max_length = max(len(s.split()) for s in list(descriptions.values()))

print('Description Length: %d' % max_length)

# test the data generator

generator = data_generator(descriptions, tokenizer, max_length)

inputs, outputs = next(generator)

print(inputs[0].shape)

print(inputs[1].shape)

print(outputs.shape)