!pip install tensorflow==2.0.0-alpha0



from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from sklearn.model_selection import train_test_split



from os import listdir
train_dir = '../input/train/train/'

train_imgs = listdir(train_dir)

nr_train_images = len(train_imgs)

nr_train_images
train_lables_df = pd.read_csv('../input/train.csv', index_col='id')

print('Total entries: ' + str(train_lables_df.size))

print(train_lables_df.head(10))
train_lables_df['has_cactus'].value_counts()
def get_test_image_path(id):

    return train_dir + id



def draw_cactus_image(id, ax):

    path = get_test_image_path(id)

    img = mpimg.imread(path)

    plt.imshow(img)

    ax.set_title('Label: ' + str(train_lables_df.loc[id]['has_cactus']))



fig = plt.figure(figsize=(20,20))

for i in range(12):

    ax = fig.add_subplot(3, 4, i + 1)

    draw_cactus_image(train_imgs[i], ax)
train_image_paths = [train_dir + ti for ti in train_imgs]

train_image_labels = [train_lables_df.loc[ti]['has_cactus'] for ti in train_imgs]



for i in range(10):

    print(train_image_paths[i], train_image_labels[i])
def img_to_tensor(img_path):

    img_tensor = tf.cast(tf.image.decode_image(tf.io.read_file(img_path)), tf.float32)

    img_tensor /= 255.0 # normalized to [0,1]

    return img_tensor



img_to_tensor(train_image_paths[0])
X_train, X_valid, y_train, y_valid = train_test_split(train_image_paths, train_image_labels, test_size=0.2)



def process_image_in_record(path, label):

    return img_to_tensor(path), label



def build_training_dataset(paths, labels, batch_size = 32):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(process_image_in_record)

    ds = ds.shuffle(buffer_size = len(paths))

    ds = ds.repeat()

    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds



def build_validation_dataset(paths, labels, batch_size = 32):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(process_image_in_record)

    ds = ds.batch(batch_size)

    return ds



train_ds = build_training_dataset(X_train, y_train)

validation_ds = build_validation_dataset(X_valid, y_valid)
mini_train_ds = build_training_dataset(X_train[:5], y_train[:5], batch_size=2)

# Fetch and print the first batch of 2 images

for images, labels in mini_train_ds.take(1):

    print(images)

    print(labels)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)
def plot_accuracies_and_losses(history):

    plt.title('Accuracy')

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.legend(['training', 'validation'], loc='upper left')

    plt.show()

    

    plt.title('Cross-entropy loss')

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.legend(['training', 'validation'], loc='upper left')

    plt.show()



plot_accuracies_and_losses(history)
cnn_model = tf.keras.Sequential()



cnn_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))

cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))

cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))

cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))



cnn_model.add(tf.keras.layers.Flatten())

cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))

cnn_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()
history = cnn_model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)
plot_accuracies_and_losses(history)
# Following the example in https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras

layer_outputs = [layer.output for layer in cnn_model.layers]

activation_model = tf.keras.Model(inputs=cnn_model.input, outputs=layer_outputs)



def print_example_and_activations(example, col_size, row_size, act_index): 

    example_array = img_to_tensor(example).numpy()

    plt.imshow(example_array)

    activations = activation_model.predict(example_array.reshape(1,32,32,3)) # batch of 1 - just the example array

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray') # image for the first (and only) element in the batch at activation_index

            activation_index += 1
# Visualizing layer 1

print_example_and_activations(X_train[0], 8, 4, 1)
# Visualizing layer 3

print_example_and_activations(X_train[0], 8, 4, 3)
test_dir = '../input/test/test/'

test_imgs = listdir(test_dir)

print(len(test_imgs))

test_imgs[:5]
def path_to_numpy_array(path):

    tensor = img_to_tensor(path)

    array = tensor.numpy()

    return array



test_image_paths = [test_dir + ti for ti in test_imgs]

test_instances = np.asarray([path_to_numpy_array(tip) for tip in test_image_paths])



test_instances[:2]
predictions = cnn_model.predict(test_instances)

print(len(predictions))
submission_data = pd.DataFrame({'id': test_imgs, 'has_cactus': predictions.flatten()})

submission_data.head(20)
submission_data.to_csv('submission.csv', index=False)
!head submission.csv