import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing import sequence

import numpy as np
sentences = [

    'I love my cat',

    'You love my dog',

    'i dont like coffee',

    'my mother is very beautiful',

    'i hate it when people touch my shoulder',

    'i adore my brother and sister',

    'I love my dog',

    'i dont like milk cream',

    'folwers are very nice',

    'i was hurt when i did not qualify for the exams',

    'i dont like exams',

    'i adore small puppies',

    'i dont like mosquitoes',

    'i love small talks',

    'i get offended cery easily',

    'i like readiing books',

    'i get angry when someone disrespects me',

    'i appreciate your tough muscles',

    'i hate that you love her',

    'you cant see me because i am bad',

    'i have a crush on you'

    

]





output = [1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,1]

output = np.array(output).reshape(-1,1)

tokenizer = Tokenizer(num_words =100,oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences,15)

print(output.shape,padded.shape)

print(output[:4])

print(padded[:4])
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(100,5, input_length = 15),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(9)),

    tf.keras.layers.Dense(8, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(padded, output,epochs=55, batch_size=8, verbose=2)

# Final evaluation of the model

#scores = model.evaluate(X_test, y_test, verbose=0)

#print("Accuracy: %.2f%%" % (scores[1]*100))
sentences1 = [

    'fast and furious is a very good movies',

    'I love my cat movie',

    'i dont like bhoot',

    'i appreciate your hard work',

    'you dont like me'

]

output1 = [1,1,0,1,0]

output1 = np.array(output).reshape(-1,1)

sequences1 = tokenizer.texts_to_sequences(sentences1)

padded1 = pad_sequences(sequences1,15)

model.predict(padded1)

model.evaluate(padded1,output1)
sentences1 = [

    'fast and furious is a very good movies',

    'I love my cat movie',

    'i dont like bhoot',

    'i appreciate your hard work',

    'you dont like me',

    'i hate that you love me',

    'you see me because i am good '

]

sequences1 = tokenizer.texts_to_sequences(sentences1)

padded1 = pad_sequences(sequences1,15)

model.predict(padded1)
