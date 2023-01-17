import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# enable eager execution
tf.enable_eager_execution()
def load_data(path, seed=113, val_split=0.2):
    """This is a helper function to load training and test data.
    
    Args:
        path: path of the files. For kaggle it's ../input/
        seed: seed value used to randomize data
        val_split: fraction of training data to be split for validation
    Return:
        tuples of training, validation and test data with labels in that order
    """
    # load test data into features and labels
    test_data = pd.read_csv(path + 'fashion-mnist_test.csv')
    x_test, y_test = np.array(test_data.iloc[:, 1:]), np.array(test_data.iloc[:, 0])
    
    # load training data into features and labels with a pre-defined split
    training_data = pd.read_csv(path + 'fashion-mnist_train.csv')
    xs = np.array(training_data.iloc[:, 1:])
    labels = np.array(training_data.iloc[:, 0])
    
    # it's always a good idea to shuffle the data initially
    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]

    idx = int(len(xs) * (1 - val_split))
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_val, y_val = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
# load the data
(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_data('../input/')
# print shapes to confirm that data was loaded properly
print("Training data", train_data.shape, "Training labels", train_labels.shape)
print("Validation data", val_data.shape, "Validation labels", val_labels.shape)
print("Test data", test_data.shape, "Test labels", test_labels.shape)
# pre-process data to change all the feature's shape into an array of one-hot encoded values
TRAINING_SIZE = len(train_data)
VAL_SIZE = len(val_data)
TEST_SIZE = len(test_data)

train_data = np.asarray(train_data, dtype=np.float32) / 255
train_data = train_data.reshape((TRAINING_SIZE, 28, 28, 1))

val_data = np.asarray(val_data, dtype=np.float32) / 255
val_data = val_data.reshape((VAL_SIZE, 28, 28, 1))

test_data = np.asarray(test_data, dtype=np.float32) / 255
test_data = test_data.reshape((TEST_SIZE, 28, 28, 1))
# pre-process label data as categorical columns
LABEL_DIMENSIONS = 10

train_labels = tf.keras.utils.to_categorical(train_labels, LABEL_DIMENSIONS)
val_labels = tf.keras.utils.to_categorical(val_labels, LABEL_DIMENSIONS)
test_labels = tf.keras.utils.to_categorical(test_labels, LABEL_DIMENSIONS)

# cast the labels to floats
train_labels= train_labels.astype(np.float32)
val_labels = val_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu))

model.summary()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
BATCH_SIZE = 128

SHUFFLE_SIZE = 10000

# create the dataset
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)
EPOCHS = 10

# store list of metric values for plotting later
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []

for epoch in range(EPOCHS):
    for (batch, (images, labels)) in enumerate(dataset):
        train_loss, train_accuracy = model.train_on_batch(images, labels)
        
        # print out after 10 loops so that we don't get bored
        if batch % 10 == 0:
            print(batch, train_accuracy)
    val_loss, val_accuracy = model.evaluate(val_data, val_labels)
    
    ## add all the data lists to visualize it later
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    
    print('Epoch #%d\t Training Loss: %.6f\t Training Accuracy: %.6f' % (epoch + 1, train_loss, train_accuracy))
    print('Epoch #%d\t Validation Loss: %.6f\t Validation Accuracy: %.6f' % (epoch + 1, val_loss, val_accuracy))
# plot loss
epochs = range(1, EPOCHS + 1)

plt.plot(epochs, train_loss_list, 'bo', label='Training loss')
plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf() # clear plot

# plot accuracy
plt.plot(epochs, train_accuracy_list, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy_list, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
loss, accuracy = model.evaluate(test_data, test_labels)
print('Test accuracy: %.4f' % (accuracy))
# grab an image from test set
img = test_data[10]
print(img.shape)

# add image to a batch where it's the only member
img = (np.expand_dims(img,0))
print(img.shape)

# predict the image
predictions = model.predict(img)
print(predictions)

# model.predict returns a list of lists, one for each image in the batch of data. Grab the predictions for our (only) image in the batch
prediction = predictions[0]
np.argmax(prediction)
