import pandas as pd, numpy as np, tensorflow as tf

from keras.preprocessing import text, sequence
TEXT_COL = 'comment_text'

BATCH_SIZE = 512

MAX_LEN = 220
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
df = None
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
x_train = preprocess(train[TEXT_COL])

y_train = np.where(train['target'] >= 0.5, 1, 0)

x_test = preprocess(test[TEXT_COL])
tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train_seq = tokenizer.texts_to_sequences(x_train)

x_test_seq = tokenizer.texts_to_sequences(x_test)
x_train = pd.DataFrame.from_dict({

    'text': x_train,

    'as_numbers': x_train_seq

})
x_train['length'] = x_train.as_numbers.str.len()

x_train['target'] = y_train
x_train.head()
class BucketedDataIterator():

    def __init__(self, df, num_buckets = 5):

        df = df.sort_values('length').reset_index(drop=True)

        self.size = len(df) / num_buckets

        self.dfs = []

        for bucket in range(num_buckets):

            self.dfs.append(df.loc[bucket*self.size: (bucket+1)*self.size - 1])

        self.num_buckets = num_buckets



        # cursor[i] will be the cursor for the ith bucket

        self.cursor = np.array([0] * num_buckets)

        self.shuffle()



        self.epochs = 0



    def shuffle(self):

        #sorts dataframe by sequence length, but keeps it random within the same length

        for i in range(self.num_buckets):

            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)

            self.cursor[i] = 0



    def next_batch(self, n):

        if np.any(self.cursor+n+1 > self.size):

            self.epochs += 1

            self.shuffle()



        i = np.random.randint(0,self.num_buckets)



        res = self.dfs[i].loc[self.cursor[i]:self.cursor[i]+n-1]

        self.cursor[i] += n



        # Pad sequences with 0s so they are all the same length

        maxlen = max(res['length'])

        x = np.zeros([n, maxlen], dtype=np.int32)

        for i, x_i in enumerate(x):

            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]



        return x, res['target'], res['length']
tr = BucketedDataIterator(x_train, 5)

padding = 0

for i in range(100):

    lengths = tr.next_batch(BATCH_SIZE)[2].values

    max_len = max(lengths)

    padding += np.sum(max_len - lengths)

print("Average padding with bucketing:", padding/(BATCH_SIZE*100))
#If x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN) is used then the padding results will be

print("Average padding without bucketing :", np.sum(MAX_LEN - x_train['length'])/len(x_train))
#Sample usage to extract batch for training

batch = tr.next_batch(BATCH_SIZE)

x = batch[0]

y = batch[1]