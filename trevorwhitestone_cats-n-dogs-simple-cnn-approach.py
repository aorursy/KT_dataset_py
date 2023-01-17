import os



import numpy as np



import matplotlib.pyplot as plt

from scipy import signal

from scipy.io import wavfile
# Read in train directories for cats and dogs before combining them



dirname = '../input/audio-cats-and-dogs/cats_dogs/train/dog/'

train_dog = []

for filename in os.listdir(dirname):

    sample_rate, samples = wavfile.read(dirname+filename)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    train_dog.append(spectrogram)

    

dirname = '../input/audio-cats-and-dogs/cats_dogs/train/cat/'

train_cat = []

for filename in os.listdir(dirname):

    sample_rate, samples = wavfile.read(dirname+filename)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    train_cat.append(spectrogram)
# Read in train directories for cats and dogs before combining them



dirname = '../input/audio-cats-and-dogs/cats_dogs/test/test/'

test_dog = []

for filename in os.listdir(dirname):

    sample_rate, samples = wavfile.read(dirname+filename)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    test_dog.append(spectrogram)

    

dirname = '../input/audio-cats-and-dogs/cats_dogs/test/cats/'

test_cat = []

for filename in os.listdir(dirname):

    sample_rate, samples = wavfile.read(dirname+filename)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    test_cat.append(spectrogram)
# pad clips so they're all of equal length

def pad_spec(spectrogram, max_length):

    return np.pad(spectrogram, ((0,0),(0,max_length-spectrogram.shape[1])), mode='mean')



max_length = max([s.shape[1] for s in train_dog]+[s.shape[1] for s in train_cat]+[s.shape[1] for s in test_dog]+[s.shape[1] for s in test_cat])
# Aggregate into train and test sets



Y_train = np.array([1]*len(train_dog)+[0]*len(train_cat))

X_train = np.array([pad_spec(s, max_length) for s in train_dog]+[pad_spec(s, max_length) for s in train_cat])



Y_test = np.array([1]*len(test_dog)+[0]*len(test_cat))

X_test = np.array([pad_spec(s, max_length) for s in test_dog]+[pad_spec(s, max_length) for s in test_cat])
with np.errstate(divide='ignore'):

    plt.pcolormesh(times, frequencies, np.log(X_train[5][:,:857]))

#plt.imshow(spectrogram)

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

plt.show()



# see what pad does here -- after the true clip has elapsed, the rest of it is represented as the average frequencies of the clip to pad it out
# Shuffle train and test sets



p = np.random.permutation(X_train.shape[0])

X_train = X_train[p]

Y_train = Y_train[p]



p = np.random.permutation(X_test.shape[0])

X_test = X_test[p]

Y_test = Y_test[p]
import tensorflow as tf

from tensorflow.keras import layers, models, backend, optimizers
backend.clear_session()



# Turns out RNN layers aren't needed here

model = models.Sequential()

model.add(layers.Conv1D(8, (3), activation='relu', padding='same', input_shape=(129,1283)))

model.add(layers.MaxPooling1D((2)))

model.add(layers.BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dropout(rate=0.7))

model.add(layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=50, batch_size=16,

                    validation_data=(X_test, Y_test))
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1.05])

plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)



# A little too good to be true -- how would it do on real world or user-submitted data?
sample_rate, samples = wavfile.read('../input/wwwkagglecomtrevorwhitestone/small-dog-barking_daniel-simion.wav')

frequencies, times, spectrogram = signal.spectrogram((samples.sum(axis=1) / 2), sample_rate) # need to convert this from stereo to mono
with np.errstate(divide='ignore'):

    plt.pcolormesh(times, frequencies, np.log(spectrogram))

#plt.imshow(spectrogram)

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

plt.show()
model.predict(np.array([spectrogram[:,:1283]]))

# model thinks this small dog is a cat -- not perfect after all, but encouraging results on a small dataset