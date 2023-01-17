import json
import numpy as np
import tensorflow as tf

# enable eager execution
tf.enable_eager_execution()
def load_data(path, num_words=None, skip_top=0, seed=113):
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]
    
    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]
    
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])
    
    if not num_words:
        num_words = max([max(x) for x in xs])

    xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
    
    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
    
    return (x_train, y_train), (x_test, y_test)
def get_word_index(path):
    with open(path) as f:
        return json.load(f)
(train_data, train_labels), (test_data, test_labels) = load_data('../input/imdb.npz', num_words=10000)
print('train_data shape:', train_data.shape)
print('train_labels shape:', train_labels.shape)
print('a train_data sample:', train_data[0])
print('a train_label sample:', train_labels[0])
print(max([max(review) for review in train_data]))
# dictionary that hashes words to their integer
word_to_integer = get_word_index('../input/imdb_word_index.json')

# print out the first ten keys in the dictionary
print(list(word_to_integer.keys())[0:10])

integer_to_word = dict([(value, key) for (key, value) in word_to_integer.items()])

# demonstrate how to find the word from an integer
print(integer_to_word[1])
print(integer_to_word[2])

# we need to subtract 3 from the indices because 0 is 'padding', 1 is 'start of sequence' and 2 is 'unknown'
decoded_review = ' '.join([integer_to_word.get(i - 3, 'UNK') for i in train_data[0]])
print(decoded_review)
def vectorize_sequences(sequences, dimension=10000):
    # creates an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, sequence in enumerate(sequences):
#         print(i, sequence)
        results[i, sequence] = 1. # set specific indices of results[i] to be 1s (float)
    return results

train_data = vectorize_sequences(train_data)
test_data = vectorize_sequences(test_data)

print(train_data.shape) # length is same as before
print(train_data[0]) # now, multi-hot encode

# vectorize the labels as well and reshape from (N, ) to (N, 1)
train_labels = np.reshape(np.asarray(train_labels, dtype=np.float32), (len(train_data), 1))
test_labels = np.reshape(np.asarray(test_labels, dtype=np.float32), (len(test_data), 1))
# create model
model = tf.keras.Sequential()

# input shape here is the length of our movie review vector
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(10000, )))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

model.summary()
VAL_SIZE = 10000

val_data = train_data[:VAL_SIZE]
partial_train_data = train_data[VAL_SIZE:]

val_labels = train_labels[:VAL_SIZE]
partial_train_labels = train_labels[VAL_SIZE:]
BATCH_SIZE = 512
SHUFFLE_SIZE = 1000

training_set = tf.data.Dataset.from_tensor_slices((partial_train_data, partial_train_labels))
training_set = training_set.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
EPOCHS = 10

# store list of metric values for plotting later
training_loss_list = []
training_accuracy_list = []
validation_loss_list = []
validation_accuracy_list = []

for epoch in range(EPOCHS):
    for reviews, labels in training_set:
        # calculate training loss and accuracy
        training_loss, training_accuracy = model.train_on_batch(reviews, labels)
        
    # calculate validation loss and accuracy
    validation_loss, validation_accuracy = model.evaluate(val_data, val_labels)
    
    # add to the lists
    training_loss_list.append(training_loss)
    training_accuracy_list.append(training_accuracy)
    validation_loss_list.append(validation_loss)
    validation_accuracy_list.append(validation_accuracy)
    
    print(('EPOCH %d\t Training Loss: %.2f\t Training Accuracy: %.2f\t Validation Loss: %.2f\t Validation Accuracy: %.2f') % (epoch + 1, training_loss, training_accuracy, validation_loss, validation_accuracy))
    
import matplotlib.pyplot as plt

epochs = range(1, EPOCHS + 1)

# "bo" specifies "blue dot"
plt.plot(epochs, training_loss_list, 'bo', label='Training loss')
# b spcifies "solid blue line"
plt.plot(epochs, validation_loss_list, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf() # clear plot

plt.plot(epochs, training_accuracy_list, 'bo', label='Training accuracy')
plt.plot(epochs, validation_accuracy_list, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
loss, accuracy = model.evaluate(test_data, test_labels)
print('Test accuracy: %.2f' % (accuracy))