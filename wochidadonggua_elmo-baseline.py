# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import zipfile

import sys

import time



# Any results you write to the current directory are saved as output.
!conda remove -y greenlet

!pip install allennlp
!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv

!wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv

!ls
from allennlp.commands.elmo import ElmoEmbedder

from allennlp.data.tokenizers import word_tokenizer

from sklearn.preprocessing import OneHotEncoder
def get_elmo_fea(data, op, wg):

	def get_nearest(slot, target):

		for i in range(target, -1, -1):

			if i in slot:

				return i

    # op = 'models/elmo_2x4096_512_2048cnn_2xhighway_options.json'

    # wg = 'models/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'



	elmo = ElmoEmbedder(options_file=op, weight_file=wg, cuda_device=0)



	tk = word_tokenizer.WordTokenizer()

	tokens = tk.batch_tokenize(data.Text)

	idx = []



	for i in range(len(tokens)):

		idx.append([x.idx for x in tokens[i]])

		tokens[i] = [x.text for x in tokens[i]]



	vectors = elmo.embed_sentences(tokens)



	ans = []

	for i, vector in enumerate([v for v in vectors]):

		P_l = data.iloc[i].Pronoun

		A_l = data.iloc[i].A.split()

		B_l = data.iloc[i].B.split()



		P_offset = data.iloc[i]['Pronoun-offset']

		A_offset = data.iloc[i]['A-offset']

		B_offset = data.iloc[i]['B-offset']



		if P_offset not in idx[i]:

			P_offset = get_nearest(idx[i], P_offset)

		if A_offset not in idx[i]:

			A_offset = get_nearest(idx[i], A_offset)

		if B_offset not in idx[i]:

			B_offset = get_nearest(idx[i], B_offset)



		emb_P = np.mean(vector[1:3, idx[i].index(P_offset), :], axis=0, keepdims=True)



		emb_A = np.mean(vector[1:3, idx[i].index(A_offset):idx[i].index(A_offset) + len(A_l), :], axis=(1, 0),

                        keepdims=True)

		emb_A = np.squeeze(emb_A, axis=0)



		emb_B = np.mean(vector[1:3, idx[i].index(B_offset):idx[i].index(B_offset) + len(B_l), :], axis=(1, 0),

                        keepdims=True)

		emb_B = np.squeeze(emb_B, axis=0)

        

		ans.append(np.concatenate([emb_A, emb_B, emb_P], axis=1))



	emb = np.concatenate(ans, axis=0)  

	return emb
def _row_to_y(row):

	if row.loc['A-coref']:

		return 0

	if row.loc['B-coref']:

		return 1

	return 2
enc = OneHotEncoder(handle_unknown='ignore')

op = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"

wg = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"



print("Started at ", time.ctime())

test_data = pd.read_csv("gap-test.tsv", sep = '\t')

X_test = get_elmo_fea(test_data, op, wg)

Y_test = test_data.apply(_row_to_y, axis=1)



validation_data = pd.read_csv("gap-validation.tsv", sep = '\t')

X_validation = get_elmo_fea(validation_data, op, wg)

Y_validation = validation_data.apply(_row_to_y, axis=1)



development_data = pd.read_csv("gap-development.tsv", sep = '\t')

X_development = get_elmo_fea(development_data, op, wg)

Y_development = development_data.apply(_row_to_y, axis=1)



print("Finished at ", time.ctime())
from keras import backend, models, layers, initializers, regularizers, constraints, optimizers

from keras import callbacks as kc

from keras import optimizers as ko



from sklearn.model_selection import cross_val_score, KFold, train_test_split

from sklearn.metrics import log_loss

import time





dense_layer_sizes = [37]

dropout_rate = 0.6

learning_rate = 0.001

n_fold = 5

batch_size = 32

epochs = 1000

patience = 100

# n_test = 100

lambd = 0.1 # L2 regularization
def build_mlp_model(input_shape):

	X_input = layers.Input(input_shape)



	# First dense layer

	X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)

	X = layers.BatchNormalization(name = 'bn0')(X)

	X = layers.Activation('relu')(X)

	X = layers.Dropout(dropout_rate, seed = 7)(X)



	# Second dense layer

# 	X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)

# 	X = layers.BatchNormalization(name = 'bn1')(X)

# 	X = layers.Activation('relu')(X)

# 	X = layers.Dropout(dropout_rate, seed = 9)(X)



	# Output layer

	X = layers.Dense(3, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)

	X = layers.Activation('softmax')(X)



	# Create model

	model = models.Model(input = X_input, output = X, name = "classif_model")

	return model
# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.

# They are very few, so I'm just dropping the rows.

remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]

X_test = np.delete(X_test, remove_test, 0)

Y_test = np.delete(Y_test, remove_test, 0)



remove_validation = [row for row in range(len(X_validation)) if np.sum(np.isnan(X_validation[row]))]

X_validation = np.delete(X_validation, remove_validation, 0)

Y_validation = np.delete(Y_validation, remove_validation, 0)



# We want predictions for all development rows. So instead of removing rows, make them 0

remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]

X_development[remove_development] = np.zeros(3*1024)
# Will train on data from the gap-test and gap-validation files, in total 2454 rows

X_train = np.concatenate((X_test, X_validation), axis = 0)

Y_train = np.concatenate((Y_test, Y_validation), axis = 0)



# Will predict probabilities for data from the gap-development file; initializing the predictions

prediction = np.zeros((len(X_development),3)) # testing predictions
# Training and cross-validation

folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)

scores = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):

	# split training and validation data

	print('Fold', fold_n, 'started at', time.ctime())

	X_tr, X_val = X_train[train_index], X_train[valid_index]

	Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]



	# Define the model, re-initializing for each fold

	classif_model = build_mlp_model([X_train.shape[1]])

	classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "sparse_categorical_crossentropy")

	callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]



	# train the model

	classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)



	# make predictions on validation and test data

	pred_valid = classif_model.predict(x = X_val, verbose = 0)

	pred = classif_model.predict(x = X_development, verbose = 0)



	# oof[valid_index] = pred_valid.reshape(-1,)

	scores.append(log_loss(Y_val, pred_valid))

	prediction += pred

prediction /= n_fold



# Print CV scores, as well as score on the test data

print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

print(scores)

print("Test score:", log_loss(Y_development,prediction))
# Write the prediction to file for submission

submission = pd.read_csv("../input/sample_submission_stage_1.csv", index_col = "ID")

submission["A"] = prediction[:,0]

submission["B"] = prediction[:,1]

submission["NEITHER"] = prediction[:,2]

submission.to_csv("submission_bert.csv")