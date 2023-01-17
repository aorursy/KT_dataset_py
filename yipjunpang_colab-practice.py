! pip install tensorflow-gpu==2.0.0-rc
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.utils import get_file

import numpy as np
# get data



(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)



print(train_data[0], train_labels[0])

print('Number of training instances: {0}, number of testing instances: {1}'.format(train_data.shape[0], test_data.shape[0]))
# get vocab

word_to_id = keras.datasets.imdb.get_word_index()

index_from=3

word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}

word_to_id["<PAD>"] = 0

word_to_id["<START>"] = 1

word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}



print(' '.join([id_to_word[i] for i in train_data[0]]))
train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                        value=word_to_id["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                       value=word_to_id["<PAD>"],

                                                       padding='post',

                                                       maxlen=256)
train_data.shape
# Problem: Use tf.data to implement input pipeline



# placeholder for implementing dataset

def create_dataset_from_tensor_slices(X, y):

    return tf.data.Dataset.from_tensor_slices((np.array(X), np.array(y)))



def create_dataset_from_generator(X, y):

    def create_gen():

        for single_x, single_y in zip(X, y):

            yield (single_x, single_y)

    output_types = (tf.int32, tf.int32)

    output_shapes = ([256], [])

    return tf.data.Dataset.from_generator(create_gen, output_types=output_types, output_shapes=output_shapes)



def create_dataset_tfrecord(X, y, mode='train'):

    file_name = '{0}.tfrecord'.format(mode)

    

    # serialize features

    # WARNING: DO NOT WRITE MULTITPLE TIMES IN PRACTICE!!! IT'S SLOW!!!

    def _int64_list_feature(value):

        """Returns an int64_list from a bool / enum / int / uint."""

        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _int64_feature(value):

        """Returns an int64_list from a bool / enum / int / uint."""

        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_fn(single_x, single_y):

        feature_tuples = {'feature': _int64_list_feature(single_x), 'label': _int64_feature(single_y)}

        example_proto = tf.train.Example(

            features=tf.train.Features(feature=feature_tuples))

        return example_proto.SerializeToString()

    # write to file

    with tf.io.TFRecordWriter(file_name) as writer:

        for single_x, single_y in zip(X, y):

            example = serialize_fn(single_x, single_y)

            writer.write(example)

            

    # read file

    dataset = tf.data.TFRecordDataset(file_name)

    def parse_fn(example_proto):

        feature_description = {'feature': tf.io.FixedLenFeature([256], tf.int64), 'label': tf.io.FixedLenFeature([], tf.int64)}

        feature_tuple = tf.io.parse_single_example(

            example_proto, feature_description)

        return feature_tuple['feature'], feature_tuple['label']

    dataset = dataset.map(parse_fn)

    return dataset



# train_dataset = create_dataset_from_generator(train_data, train_labels)

# test_dataset = create_dataset_from_generator(test_data, test_labels)



train_dataset = create_dataset_tfrecord(train_data, train_labels)

test_dataset = create_dataset_tfrecord(test_data, test_labels, mode='test')



train_dataset = train_dataset.shuffle(10000).batch(256).prefetch(100).repeat()

test_dataset = test_dataset.batch(256).prefetch(100)

    
# Problem: Implement a custom keras layer which has the identical effects of dense, but print the mean

#   of the variables if the mean value is greater than zero. Print for maximum 10 times.



# placeholder for implementing using Functional API or Model Subclassing

class WeirdDense(tf.keras.layers.Layer):



    def __init__(self, output_dim, activation):

        super(WeirdDense, self).__init__()

        self.output_dim = output_dim

        self.activation = activation

        self.print_times = tf.Variable(0, dtype=tf.int32, trainable=False)

        



    def build(self, input_shape):

        # Create a trainable weight variable for this layer.

        self.w = self.add_weight(shape=(input_shape[-1], self.output_dim),

                                 initializer='random_normal',

                                 trainable=True)

        self.b = self.add_weight(shape=(self.output_dim,),

                                 initializer='random_normal',

                                 trainable=True)

    @tf.function

    def call(self, x):

        mean_val = tf.reduce_mean(self.w)

        if tf.greater(mean_val, 0):

            if tf.less_equal(self.print_times, 10):

                tf.print(mean_val)

                self.print_times.assign_add(1)



        return_tensor = self.activation(tf.matmul(x, self.w) + self.b)

        return return_tensor

            



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_dim)
# input shape is the vocabulary count used for the movie reviews (10,000 words)

vocab_size = 10000



model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(WeirdDense(16, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

history = model.fit(train_dataset,

                    epochs=1,

                    steps_per_epoch=100,

                    validation_data=test_dataset,

                    validation_steps=100,

                    verbose=1)