import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
EXAMINE = 21
SEED = 22
np.random.seed(SEED)
def get_gender_as_num(gender):
    if gender == "male":
        return 0
    else:
        return 1
def get_age_group(age): # HIGH NOTE: changing each of the scalars to a vector. This is probably not a good idea
    if age < 18:
        # 13 - 17
        return [1, 0, 0]
    elif age < 28:
        # 23 - 27
        return [0, 1, 0]
    elif age < 49:
        # 33 - 48
        return [0, 0, 1]
    else:
        return [0, 0, 0]
blog_posts_data_dir = "../input/blog-posts-labeled-with-age-and-gender/"
train_file_name = "train.json"
test_file_name = "test.json"

# Load data
with open(blog_posts_data_dir + train_file_name) as r:
    training_set = json.load(r)
raw_posts = [instance["post"] for instance in training_set]
print(raw_posts[EXAMINE])
median_words_per_sample = np.median([len(instance["post"]) for instance in training_set])

# Map each word to a unique int value
MAX_WORD_COUNT = 20000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = MAX_WORD_COUNT)
posts = [instance["post"] for instance in training_set]
tokenizer.fit_on_texts(posts)
word_index = dict(list(tokenizer.word_index.items())[:20000])
sequences = tokenizer.texts_to_sequences(posts)
median_words_per_tokenized_sample = np.median([len(post) for post in sequences])
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = int(median_words_per_tokenized_sample),
                                                     padding = "post")
for i, instance in enumerate(training_set):
    instance["post"] = data[i]
    instance["gender"] = get_gender_as_num(instance["gender"])
    instance["age"] = get_age_group(int(instance["age"]))

print(training_set[EXAMINE]["post"])
print(training_set[EXAMINE]["age"])
print(list(word_index.items())[ : 100])
samples_count = len(training_set)

categories_count = len(training_set[0]["age"])

samples_per_class = {0 : 0, 1 : 0, 2 : 0}
for instance in training_set:
    for i, a in enumerate(instance["age"]):
        if a == 1:
            samples_per_class[i] += 1
            break
 
print("Number of Samples:", samples_count)
print("Number of Categories:", categories_count)
print("Samples per Class:", samples_per_class)
print("Median Words per Sample:", median_words_per_sample)
print("Median Words per Tokenized Sample:", median_words_per_tokenized_sample)
print("Samples to Words Per Sample Ratio:", samples_count / median_words_per_tokenized_sample)
# plt.hist(list(length_distribution.keys()))
# plt.xlabel("Length of a Sample")
# plt.ylabel("Number of samples")
# plt.show()
EMBEDDING_DIM = 50

glove_path = "../input/glove6b/"
glove_dict = {}
with open(glove_path + "glove.6B.50d.txt") as f:
    for line in f:
        line_values = line.split(" ")
        word = line_values[0]
        embedding_coefficients = np.asarray(line_values[1 : ], dtype = "float32")
        glove_dict[word] = embedding_coefficients

glove_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    glove_vector = glove_dict.get(word)
    if glove_vector is not None:
        glove_weights[i] = glove_vector
print(len(glove_weights))
# Define the model
# Embedding, Conv, Pool, Conv, Pool, Flatten, Dense, Dense
model_1 = tf.keras.Sequential()
model_1.add(tf.keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights = [glove_weights],
                                    input_length = median_words_per_tokenized_sample, trainable = True))
model_1.add(tf.keras.layers.SeparableConv1D(50, 5, activation = "relu"))
model_1.add(tf.keras.layers.MaxPooling1D())
model_1.add(tf.keras.layers.SeparableConv1D(100, 3, activation = "relu"))
model_1.add(tf.keras.layers.MaxPooling1D())
model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(24, activation = "sigmoid"))
model_1.add(tf.keras.layers.Dense(3, activation = "softmax"))

posts_train = np.array([instance["post"] for instance in training_set])
ages_train = np.array([instance["age"] for instance in training_set])
model_1.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["acc"])
model_1.summary()
history_1 = model_1.fit(posts_train, ages_train, epochs = 10, batch_size = 500, validation_split = 0.2)
# Load data
with open(blog_posts_data_dir + test_file_name) as r:
    test_set = json.load(r)
test_posts = [instance["post"] for instance in test_set]
test_sequences = tokenizer.texts_to_sequences(test_posts)
test_post_data = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen = int(median_words_per_tokenized_sample),
                                                     padding = "post")
for i, instance in enumerate(test_set):
    instance["post"] = test_post_data[i]
    instance["gender"] = get_gender_as_num(instance["gender"])
    instance["age"] = get_age_group(int(instance["age"]))
posts_test = np.array([instance["post"] for instance in test_set])
ages_test = np.array([instance["age"] for instance in test_set])
model_1.evaluate(posts_test, ages_test)
# Define the model
# Embedding, Conv, Pool, Conv, Pool, Flatten, Dense, Dense, Dense
model_2 = tf.keras.Sequential()
model_2.add(tf.keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights = [glove_weights],
                                    input_length = median_words_per_tokenized_sample, trainable = True))
model_2.add(tf.keras.layers.SeparableConv1D(100, 5, activation = "relu"))
model_2.add(tf.keras.layers.MaxPooling1D())
model_2.add(tf.keras.layers.SeparableConv1D(200, 3, activation = "relu"))
model_2.add(tf.keras.layers.MaxPooling1D())
model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(48, activation = "sigmoid"))
model_2.add(tf.keras.layers.Dense(24, activation = "sigmoid"))
model_2.add(tf.keras.layers.Dense(3, activation = "softmax"))
model_2.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["acc"])
model_2.summary()
history_2 = model_2.fit(posts_train, ages_train, epochs = 7, batch_size = 500, validation_split = 0.2)
model_2.evaluate(posts_test, ages_test)
model_2.evaluate(posts_test, ages_test)
# Define the model
# Embedding, Conv, Pool, Conv, Pool, Flatten, Dense, Dense
model_3 = tf.keras.Sequential()
model_3.add(tf.keras.layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights = [glove_weights],
                                    input_length = median_words_per_tokenized_sample, trainable = True))
model_3.add(tf.keras.layers.SeparableConv1D(100, 5, activation = "relu"))
model_3.add(tf.keras.layers.MaxPooling1D())
model_3.add(tf.keras.layers.SeparableConv1D(200, 3, activation = "relu"))
model_3.add(tf.keras.layers.MaxPooling1D())
model_3.add(tf.keras.layers.SeparableConv1D(100, 3, activation = "relu"))
model_3.add(tf.keras.layers.MaxPooling1D())
model_3.add(tf.keras.layers.Flatten())
model_3.add(tf.keras.layers.Dense(24, activation = "sigmoid"))
model_3.add(tf.keras.layers.Dense(3, activation = "softmax"))

model_3.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["acc"])
model_3.summary()
history_3 = model_3.fit(posts_train, ages_train, epochs = 10, batch_size = 500, validation_split = 0.2)
