import re

import  numpy as np

import pandas as pd

import tensorflow as tf



import transformers



import matplotlib.pyplot as plt



from tqdm import tqdm

tqdm.pandas()
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU: ', tpu.master())

    

except Exception as e:

    tpu = None

    



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    

else:

    strategy = tf.distribute.get_strategy()

    

print('Number of clusters: ', strategy.num_replicas_in_sync)
data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

# data = data.sample(80000)
test_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test_data.head()
valid_data = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

valid_data.head()
contractions_dict = {     

"ain't": "am not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"I'd": "I had",

"I'd've": "I would have",

"I'll": "I will",

"I'll've": "I will have",

"I'm": "I am",

"I've": "I have",

"isn't": "is not",

"it'd": "it had",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "iit will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so is",

"that'd": "that had",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there had",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they had",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}
def clean(text, contractions=contractions_dict, remove_stop=False):

    text = text.lower()

    text = re.sub(r'\([^)]*\)', '', text)

    text = ' '.join([contractions[t] if t in contractions else t for t in text.split(' ')])

    text = re.sub(r"'s\b", "", text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = re.sub('[m]{2, }', 'mm', text)

    

    return ' '.join(text.strip().split())
print('* BEFORE CLEANING: ', data.comment_text.iloc[0], '\n')

print('* AFTER CLEANING: ', clean(data.comment_text.iloc[0]))
data['comment_text_clean'] = data['comment_text'].apply(clean)
# max_len = int(np.percentile(data['comment_text_clean'].str.split().apply(len), 90))

max_len = 256

# max_len = int(max(data['comment_text_clean'].str.split().apply(len))) - 250

print('Max comment length: ', max_len)
def encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
AUTO_TUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 15

BATCH_SIZE = 10 * strategy.num_replicas_in_sync

MAX_LEN = max_len

NUM_CLASSES = 1
tokenizer = transformers.AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')
data['comment_text_clean'] = data['comment_text'].apply(clean)

valid_data['comment_text_clean'] = valid_data['comment_text'].apply(clean)
X_train = encode(data.comment_text_clean.astype(str), tokenizer, maxlen=MAX_LEN)

X_test = encode(test_data.content.astype(str), tokenizer, maxlen=MAX_LEN)

X_valid = encode(valid_data.comment_text_clean.astype(str), tokenizer, maxlen=MAX_LEN)



y_train = data.toxic.values

y_valid = valid_data.toxic.values
train_dataset = (tf.data.Dataset

                .from_tensor_slices((X_train, y_train))

                .repeat()

                .shuffle(2048)

                .batch(BATCH_SIZE)

                .prefetch(AUTO_TUNE))



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(X_test)

    .batch(BATCH_SIZE)

)





valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO_TUNE)

)
def build_model(transformer, max_len=MAX_LEN):

    input_word_ids = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32, name='input_word_ids')

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(cls_token)

    

    model = tf.keras.models.Model(inputs=input_word_ids, outputs=out)

    

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
with strategy.scope():

    transformer_layer = transformers.TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-large')

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
n_steps = X_train.shape[0] // BATCH_SIZE

history = model.fit(train_dataset,

                    steps_per_epoch=n_steps,

                    epochs=EPOCHS,

                    validation_data=valid_dataset,

                    callbacks=callbacks)
n_steps = X_valid.shape[0] // BATCH_SIZE

history_test = model.fit(valid_dataset.repeat(),

                    steps_per_epoch=n_steps,

                    epochs=EPOCHS,

                    callbacks=callbacks)
plt.figure(figsize=(12, 8))

    

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], c='b', label='Train Acc')

plt.plot(history.history['val_accuracy'], c='g', label='Test Acc')

plt.title('XLM Roberta Large - Training/Testig Accuracy')



plt.legend()



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], c='r', label='Train Loss')

plt.plot(history.history['val_loss'], c='g', label='Test Loss')

plt.title('XLM Roberta Large - Testing Loss')



plt.legend()



plt.show()
submission_file = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

submission_file.head()
submission_file['toxic'] = model.predict(test_dataset, verbose=1)

submission_file.to_csv('submission.csv', index=False)
def toxicity(x):

    x = encode(np.array([x]), tokenizer, maxlen=MAX_LEN)

    return model.predict(x, verbose=0)
toxicity('That was not so bad, I like it')
toxicity('Go burn in hell you idiot')