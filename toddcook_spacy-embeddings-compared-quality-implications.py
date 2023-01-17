%matplotlib inline

import spacy

import numpy as np  

import pandas as pd  

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import learning_curve

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

# set random for reproducibility

import tensorflow as tf

import random as python_random

np.random.seed(12)

python_random.seed(12)

tf.random.set_seed(12)
# Load Spacy for embeddings, big and small

# The vectors in spacy medium use the hashing trick: 

# a number of keys map to the same vector

nlp = spacy.load('/kaggle/input/spacymd225//en_core_web_md-2.2.5/en_core_web_md/en_core_web_md-2.2.5')

# Spacy large has unique key vector mappings

nlp_large = spacy.load('/kaggle/input/spacyen-core-web-lg225/en_core_web_lg-2.2.5/en_core_web_lg/en_core_web_lg-2.2.5')



spacy_words = [tmp.text for tmp in nlp.vocab]

print('A few examples', spacy_words [:9])

spacy_words = set(spacy_words)
occupations_df = pd.read_csv('/kaggle/input/mlyoucanuse-wikidata-occupations-labeled/occupations.wikidata.all.labeled.tsv', sep='\t')

occupations_df.head()
employers_df = pd.read_csv('/kaggle/input/mlyoucanuse-wikidata-employers-labeled/employers.wikidata.all.labeled.csv', sep='\t')

employers_df.head()
all_occupations =  occupations_df['occupation'].tolist()

occs_in_spacy = [tmp for tmp in all_occupations 

                 if tmp in spacy_words]

print(f"{len(occs_in_spacy):,} occupations in Spacy")

all_employers = employers_df['employer'].tolist()

emps_in_spacy = [tmp for tmp in all_employers 

                 if tmp in spacy_words]

print(f"{len(emps_in_spacy):,} employers in Spacy")
# A good occupation and a good employer are mutually exclusive, 

# but a bad occupation might be a good employer, and vice versa, 

# so we'll carefully filter our negative examples



good_occs = set()

bad_occs = set()



good_emps = set()

bad_emps = set()



clean_negative_eg = set()



for idx, row in occupations_df.iterrows():

    if row['occupation'] in spacy_words :

        if row['label'] ==1:

            good_occs.add(row['occupation'])

        else: 

            bad_occs.add(row['occupation'])

            

for idx, row in employers_df.iterrows():

    if row['employer'] in spacy_words :

        if row['label'] ==1:

            good_emps.add(row['employer'])

        else: 

            bad_emps.add(row['employer'])

            

for occ in bad_occs:

    if occ not in good_emps:

            clean_negative_eg.add(occ)

for emp in bad_emps:

    if emp not in good_occs:

            clean_negative_eg.add(emp)

            

print(f"{len(good_occs):,} good occupations, {len(good_emps):,} good employers, {len(clean_negative_eg):,} negative examples")
# We'll create our data using full size word vectors first



Xlg = []

for word in good_occs:

    Xlg.append(nlp_large(word).vector)

for word in good_emps:

    Xlg.append(nlp_large(word).vector)

for word in clean_negative_eg:

    Xlg.append(nlp_large(word).vector)

    

Xlg = np.asarray(Xlg)

ylg = np.hstack([np.ones(len(good_occs)), 

                 np.array([2] * len(good_emps), np.int32),

                 np.zeros(len(clean_negative_eg))])



# set random for reproducibility

# np.random.seed(12) # already done above

randomize = np.arange(len(ylg))

np.random.shuffle(randomize)

Xlg = Xlg[randomize]

ylg = ylg[randomize]



print('X shape', Xlg.shape, 'y shape', ylg.shape)



X_train, X_test, y_train, y_test = train_test_split(Xlg, ylg, test_size=0.3, random_state=12)

# Partition equal sizes of test and validation sets

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=12)



# a create model method, which we'll reuse for the medium sized Spacy word embeddings



def get_model(input_dim=300):

    np.random.seed(12) # set seeds for reproducibility

    python_random.seed(12)

    tf.random.set_seed(12)

    model = Sequential(name='emp_occ_other_detector')

    model.add(Dense(256, name='Dense_layer_1',

                    input_dim=input_dim, activation='relu'))

    model.add(Dense(256, name='Dense_layer_2',

                    activation='relu'))

    model.add(Dense(3, name='Dense_layer_3', activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 

                  optimizer='adam',

                  metrics=['accuracy'])  

    return model



model = get_model()

history = model.fit(X_train, y_train,

                     epochs=20,

                     verbose=0,

                     validation_data=(X_test, y_test),

                     batch_size=32)

loss, accuracy = model.evaluate(X_train, y_train, verbose=True)

print(f"Training Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)

print(f"Testing Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = model.evaluate(X_validation, y_validation, verbose=True)

print(f"Unseen Accuracy: {accuracy:.4f} Loss: {loss:.4f}")
Xmd = []

for word in good_occs:

    Xmd.append(nlp(word).vector)

for word in good_emps:

    Xmd.append(nlp(word).vector)

for word in clean_negative_eg:

    Xmd.append(nlp(word).vector)

    

Xmd = np.asarray(Xmd)

ymd = np.hstack([np.ones(len(good_occs)),

                 np.array([2] * len(good_emps), np.int32), 

                 np.zeros(len(clean_negative_eg))])



# we'll reuse the random array from before

# randomize = np.arange(len(ymd))

# np.random.shuffle(randomize)

Xmd = Xmd[randomize]

ymd = ymd[randomize]



print('X shape', Xmd.shape, 'y shape', ymd.shape)



X_train, X_test, y_train, y_test = train_test_split(Xmd, ymd, test_size=0.3, random_state=12)

# Partition equal sizes of test and validation sets

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=12)



md_model = get_model()



history = md_model.fit(X_train, y_train,

                     epochs=20,

                     verbose=False,

                     validation_data=(X_test, y_test),

                     batch_size=32)

loss, accuracy = md_model.evaluate(X_train, y_train, verbose=True)

print(f"Training Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = md_model.evaluate(X_test, y_test, verbose=True)

print(f"Testing Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = md_model.evaluate(X_validation, y_validation, verbose=True)

print(f"Unseen Accuracy: {accuracy:.4f} Loss: {loss:.4f}")
pca = PCA(n_components=50)

Xmd50 = pca.fit_transform(Xmd)



X_train, X_test, y_train, y_test = train_test_split(Xmd50, ymd, test_size=0.3, random_state=12)

# Partition equal sizes of test and validation sets

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=12)



md_model = get_model(input_dim=50) 

print("Hashed embeddings reduced with PCA (300 -> 50 dim)")

history = md_model.fit(X_train, y_train,

                     epochs=20,

                     verbose=False,

                     validation_data=(X_test, y_test),

                     batch_size=32)

loss, accuracy = md_model.evaluate(X_train, y_train, verbose=True)

print(f"Training Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = md_model.evaluate(X_test, y_test, verbose=True)

print(f"Testing Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = md_model.evaluate(X_validation, y_validation, verbose=True)

print(f"Unseen Accuracy: {accuracy:.4f} Loss: {loss:.4f}")    



pca = PCA(n_components=50)

Xlg50 = pca.fit_transform(Xlg)



X_train, X_test, y_train, y_test = train_test_split(Xlg50, ymd, test_size=0.3, random_state=12)

# Partition equal sizes of test and validation sets

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=12)



lg50_model = get_model(input_dim=50) 

print("Full embeddings reduced with PCA (300 -> 50 dim)")

history = lg50_model.fit(X_train, y_train,

                     epochs=20,

                     verbose=False,

                     validation_data=(X_test, y_test),

                     batch_size=32)

loss, accuracy = lg50_model.evaluate(X_train, y_train, verbose=True)

print(f"Training Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = lg50_model.evaluate(X_test, y_test, verbose=True)

print(f"Testing Accuracy: {accuracy:.4f} Loss: {loss:.4f}")

loss, accuracy = lg50_model.evaluate(X_validation, y_validation, verbose=True)

print(f"Unseen Accuracy: {accuracy:.4f} Loss: {loss:.4f}")   
# Create CV training and test scores for various training set sizes, large and medium

train_sizeslg, train_scoreslg, test_scoreslg = learning_curve(

    RandomForestClassifier(), 

    Xlg, 

    ylg,

    # Number of folds in cross-validation

    cv=10,

    # Evaluation metric

    scoring='accuracy',

    # Use all computer cores

    n_jobs=-1, 

    # 20 different sizes of the training set

    train_sizes=np.linspace(0.01, 1.0, 10))



# medium

train_sizesmd, train_scoresmd, test_scoresmd = learning_curve(

    RandomForestClassifier(), 

    Xmd, 

    ymd,

    # Number of folds in cross-validation

    cv=10,

    # Evaluation metric

    scoring='accuracy',

    # Use all computer cores

    n_jobs=-1, 

    # 20 different sizes of the training set

    train_sizes=np.linspace(0.01, 1.0, 10))
# Create means and standard deviations of training set scores

train_meanlg = np.mean(train_scoreslg, axis=1)

train_stdlg = np.std(train_scoreslg, axis=1)

# Create means and standard deviations of test set scores

test_meanlg = np.mean(test_scoreslg, axis=1)

test_stdlg = np.std(test_scoreslg, axis=1) 

train_meanmd = np.mean(train_scoresmd, axis=1)

train_stdmd = np.std(train_scoresmd, axis=1)

test_meanmd = np.mean(test_scoresmd, axis=1)

test_stdmd = np.std(test_scoresmd, axis=1)



plt.figure(figsize=(9,4))

plt.plot(train_sizeslg, test_meanlg, color="#0000FF", label="Full Size Embeddings")

plt.plot(train_sizesmd, test_meanmd, color="#FF0000", label="Hashed Embeddings")

plt.title("Learning Curve: Full Size vs. Hashed Embeddings")

plt.xlabel("Training Set Size")

plt.ylabel("Accuracy")

plt.legend(loc="best")

plt.tight_layout(pad=1, w_pad=.5, h_pad=.5)

plt.savefig('/kaggle/working/learningcurve_fullvshashedembeddings.png')