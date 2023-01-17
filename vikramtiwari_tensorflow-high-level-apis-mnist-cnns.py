import tensorflow as tf
import numpy as np

# enable eager execution
tf.enable_eager_execution()
def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
(train_images, train_labels), (test_images, test_labels) = load_data('../input/mnist.npz')
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

train_images = np.asarray(train_images, dtype=np.float32) / 255

# convert the train images and add channels
train_images = train_images.reshape((TRAINING_SIZE, 28, 28, 1))

test_images = np.asarray(test_images, dtype=np.float32) / 255
# convert the train images and add channels
test_images = test_images.reshape((TEST_SIZE, 28, 28, 1))
# how many digits we are predicting from (0-9)
LABEL_DIMENSIONS = 10

train_labels = tf.keras.utils.to_categorical(train_labels, LABEL_DIMENSIONS)
test_labels = tf.keras.utils.to_categorical(test_labels, LABEL_DIMENSIONS)

# cast the labels to floats, needed later
train_labels = train_labels.astype(np.float32)
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
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)
EPOCHS = 5

for epoch in range(EPOCHS):
    for (batch, (images, labels)) in enumerate(dataset):
        train_loss, train_accuracy = model.train_on_batch(images, labels)
        
        if batch % 10 == 0:
            print(batch, train_accuracy)
            
    print('Epoch #%d\t Loss: %.6f\t Accuracy: %.6f' %(epoch + 1, train_loss, train_accuracy))
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('\nTest Model \t\t Loss: %.6f\t Accuracy: %.6f' % (test_loss, test_accuracy))