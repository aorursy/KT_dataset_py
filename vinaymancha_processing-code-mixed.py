import os

import time

import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

from transformers import *

import tokenizers

print('TF version',tf.__version__)
class Vocabulary:

    def __init__(self, name):

        PAD_token = 0   # Used for padding short sentences

        unk_token = 1

        SOS_token = 2   # Start-of-sentence token

        EOS_token = 3   # End-of-sentence token

        

        self.name = name

        self.word2index = {"[PAD]" : PAD_token,"[UNK]" : unk_token, "[SOS]" : SOS_token, "[EOS]" : EOS_token}

        self.word2count = {}

        self.index2word = {PAD_token: "[PAD]", unk_token: "[UNK]", SOS_token: "[SOS]", EOS_token: "[EOS]" }

        self.num_words = 4

        self.num_sentences = 0

        self.longest_sentence = 0



    def add_word(self, word):

        if word not in self.word2index:

            # First entry of word into vocabulary

            self.word2index[word] = self.num_words

            self.word2count[word] = 1

            self.index2word[self.num_words] = word

            self.num_words += 1

        else:

            # Word exists; increase word count

            self.word2count[word] += 1

            

    def add_sentence(self, sentence):

        sentence = str(sentence)

        sentence_len = 0

        for word in sentence.split(' '):

            sentence_len += 1

            self.add_word(word)

        if sentence_len > self.longest_sentence:

            # This is the longest sentence

            self.longest_sentence = sentence_len

        # Count the number of sentences

        self.num_sentences += 1



    def to_word(self, index):

        return self.index2word[index]



    def to_index(self, word):

        return self.word2index[word]

    

    def encode(self, st):

        tokens = []

        for i in st.split():

            try:

                tokens.append(self.to_index(i))

            except:

                tokens.append(self.to_index("[UNK]"))

        return tokens

    def decode(self, tokens):

        st = []

        for i in tokens:

            st.append(self.to_word(i))

        return " ".join(st)
df=pd.read_csv("../input/code-mixed-telugu/cleaned data.csv", dtype=str)
inp_lang = Vocabulary("input")

targ_lang = Vocabulary("target")
for k in range(df.shape[0]):

    inp_lang.add_sentence(df.loc[k,'text'])

for k in range(df.shape[0]):

    targ_lang.add_sentence(df.loc[k,'target'])
MAX_LEN = 74

ct = df.shape[0]

input_ids = []

attention_mask = []

token_type_ids = []

target_ids = []



for k in range(df.shape[0]):

    tokens = inp_lang.encode(str(df.loc[k,'text']))

    target_toks = targ_lang.encode(str(df.loc[k,'target']))

    

    if len(tokens)>70:

        tokens = tokens[:70]

    if len(target_toks)>70:

        target_toks = target_toks[:70]

    

    

    tokens = [2] + tokens + [3]

    target_toks = [2] + target_toks + [3]

    input_ids.append(tokens)

    target_ids.append(target_toks)

    
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_ids,

                                                         padding='post')

target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_ids,

                                                         padding='post')
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.15)



# Show length

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
BUFFER_SIZE = len(input_tensor_train)

BATCH_SIZE = 64

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

print(steps_per_epoch)

embedding_dim = 256

units = 1024



vocab_inp_size = len(inp_lang.word2index)+1

vocab_tar_size = len(targ_lang.word2index)+1



dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
vocab_inp_size
example_input_batch, example_target_batch = next(iter(dataset))

example_input_batch.shape, example_target_batch.shape
class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):

        super(Encoder, self).__init__()

        self.batch_sz = batch_sz

        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,

                                       return_sequences=True,

                                       return_state=True,

                                       recurrent_initializer='glorot_uniform')



    def call(self, x, hidden):

        x = self.embedding(x)

        output, state = self.gru(x, initial_state = hidden)

        return output, state



    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, self.enc_units))
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)



# sample input

sample_hidden = encoder.initialize_hidden_state()

sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))

print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):

        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)

        self.W2 = tf.keras.layers.Dense(units)

        self.V = tf.keras.layers.Dense(1)



    def call(self, query, values):

        # query hidden state shape == (batch_size, hidden size)

        # query_with_time_axis shape == (batch_size, 1, hidden size)

        # values shape == (batch_size, max_len, hidden size)

        # we are doing this to broadcast addition along the time axis to calculate the score

        query_with_time_axis = tf.expand_dims(query, 1)



        # score shape == (batch_size, max_length, 1)

        # we get 1 at the last axis because we are applying score to self.V

        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(tf.nn.tanh(

            self.W1(query_with_time_axis) + self.W2(values)))



        # attention_weights shape == (batch_size, max_length, 1)

        attention_weights = tf.nn.softmax(score, axis=1)



        # context_vector shape after sum == (batch_size, hidden_size)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)



        return context_vector, attention_weights
attention_layer = BahdanauAttention(10)

attention_result, attention_weights = attention_layer(sample_hidden, sample_output)



print("Attention result shape: (batch size, units) {}".format(attention_result.shape))

print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):

        super(Decoder, self).__init__()

        self.batch_sz = batch_sz

        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.dec_units,

                                   return_sequences=True,

                                   return_state=True,

                                   recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)



        # used for attention

        self.attention = BahdanauAttention(self.dec_units)



    def call(self, x, hidden, enc_output):

        # enc_output shape == (batch_size, max_length, hidden_size)

        context_vector, attention_weights = self.attention(hidden, enc_output)



        # x shape after passing through embedding == (batch_size, 1, embedding_dim)

        x = self.embedding(x)



        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



        # passing the concatenated vector to the GRU

        output, state = self.gru(x)



        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))



        # output shape == (batch_size, vocab)

        x = self.fc(output)



        return x, state, attention_weights
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)



sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),

                                      sample_hidden, sample_output)



print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(

        from_logits=True, reduction='none')



def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)



    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask



    return tf.reduce_mean(loss_)
checkpoint_dir = '/kaggle/working/'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer,

                                 encoder=encoder,

                                 decoder=decoder)
def train_step(inp, targ, enc_hidden):

    loss = 0



    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inp, enc_hidden)



        dec_hidden = enc_hidden



        dec_input = tf.expand_dims([targ_lang.word2index["[SOS]"]] * BATCH_SIZE, 1)



        # Teacher forcing - feeding the target as the next input

        for t in range(1, targ.shape[1]):

            # passing enc_output to the decoder

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)



            loss += loss_function(targ[:, t], predictions)



            # using teacher forcing

            dec_input = tf.expand_dims(targ[:, t], 1)



    batch_loss = (loss / int(targ.shape[1]))



    variables = encoder.trainable_variables + decoder.trainable_variables



    gradients = tape.gradient(loss, variables)



    optimizer.apply_gradients(zip(gradients, variables))



    return batch_loss
EPOCHS = 25



for epoch in range(EPOCHS):

    start = time.time()



    enc_hidden = encoder.initialize_hidden_state()

    total_loss = 0



    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

        batch_loss = train_step(inp, targ, enc_hidden)

        total_loss += batch_loss



        if batch % 40 == 0:

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,

                                                       batch,

                                                       batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs

    if (epoch + 1) % 5 == 0:

        checkpoint.save(file_prefix = checkpoint_prefix)



    print('Epoch {} Loss {:.4f}'.format(epoch + 1,

                                      total_loss / steps_per_epoch))

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
max_length_targ, max_length_inp = 72, 72
def evaluate(sentence):

    attention_plot = np.zeros((max_length_targ, max_length_inp))



    #sentence = preprocess_sentence(sentence)

    sentence = "[SOS] "+sentence+" [EOS]"



    inputs = inp_lang.encode(sentence)

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],

                                                         maxlen=max_length_inp,

                                                         padding='post')

    inputs = tf.convert_to_tensor(inputs)



    result = ''



    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)



    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word2index['[SOS]']], 0)



    for t in range(max_length_targ):

        predictions, dec_hidden, attention_weights = decoder(dec_input,

                                                             dec_hidden,

                                                             enc_out)



        # storing the attention weights to plot later on

        attention_weights = tf.reshape(attention_weights, (-1, ))

        attention_plot[t] = attention_weights.numpy()



        predicted_id = tf.argmax(predictions[0]).numpy()



        result += targ_lang.index2word[predicted_id] + ' '



        if targ_lang.index2word[predicted_id] == '[EOS]':

            return result, sentence, attention_plot



        # the predicted ID is fed back into the model

        dec_input = tf.expand_dims([predicted_id], 0)



    return result, sentence, attention_plot
def translate(sentence):

    result, sentence, attention_plot = evaluate(sentence)



    print('Input: %s' % (sentence))

    print('Predicted translation: {}'.format(result))



    #attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

    #plot_attention(attention_plot, sentence.split(' '), result.split(' '))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for k in range(df.sample(n=10,random_state=42).shape[0]):

    translate(df.loc[k,'text'])