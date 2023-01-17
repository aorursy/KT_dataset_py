import tflearn
import numpy as np
import re
def fade(x):
    '''
    Fading the input values exponentially.
    You could use any other function or values in order to fade out the previous ones,
    but I've just found this one works best.
    Evolving the constant "2" can be done using a genetic algorithm.
    '''
    return x / 2
class SimpleSequenceGenerator(tflearn.DNN):
    '''
    We'll create a wrapper class for TFLearn's DNN.
    '''
    def __init__(self, vocab):
        '''
        A vocabulary = a list of words.
        Creating basic variables and the model.
        
        As explained, model's shape is N, K, K, N where here K = N * 2.
        '''
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_freq = np.zeros(self.vocab_size)
        self.char_idx = {j:i for i, j in enumerate(vocab)}
        self.idx_char = {i:j for i, j in enumerate(vocab)}
        # Creating the model
        model = tflearn.input_data(shape = [None, len(self.vocab)], name = 'inputs')
        model = tflearn.fully_connected(model, 2 * len(vocab), activation = 'relu')
        model = tflearn.dropout(model, 0.5)
        model = tflearn.fully_connected(model, 2 * len(vocab), activation = 'relu')
        model = tflearn.dropout(model, 0.5)
        model = tflearn.fully_connected(model, len(vocab), activation = 'softmax', name = 'targets')
        model = tflearn.regression(model, optimizer = 'adam', loss = 'categorical_crossentropy', learning_rate = 0.001)
        tflearn.DNN.__init__(self, model)

    def one_hot(self, n):
        '''
        Useful function for converting an index of range 1...vocab_size to one hot (categorical format)
        '''
        a = np.zeros(self.vocab_size)
        a[n] = 1
        return a

    def seperate_sentences(self, text):
        '''
        Split sentences by a line braek. Completely depends on how you create the class.
        '''
        sentences = text.split('\n')
        return [i.split() for i in sentences]

    def fit(self, text, n_epoch = 3):
        '''
        Overidding the fit method.
        Create the training data from the text.
        I'm using the fade function in order to fade out the previous values.
        '''
        x = []
        y = []
        sentences = self.seperate_sentences(text)
        for sentence in sentences:
            current_x = np.zeros(self.vocab_size)
            for i in range(len(sentence) - 1):
                idx = self.char_idx[sentence[i]]
                self.idx_freq[idx] += 1
                current_x = fade(current_x)
                current_x[idx] += 1
                x.append(np.array(current_x))
                y.append(self.one_hot(self.char_idx[sentence[i + 1]]))
    
    
        '''
        Doing the actual training!
        '''
        print('Training data: {} rows'.format(len(x)))
        x = np.array(x)
        y = np.array(y)
        tflearn.DNN.fit(self, x, y, n_epoch = n_epoch)

    def generate(self, seed, gen_len = 20):
        '''
        Covnert the seed into the input format,
        and the same way concat new generated values.
        '''
        seed_value = np.zeros(self.vocab_size)
        seed_words = seed.split(' ')
        for word in seed_words:
            seed_value = fade(seed_value)
            seed_value[self.char_idx[word]] += 1

        fade_val = 0
        for i in range(gen_len):
            result = self.predict([seed_value])
            result_idx = np.random.choice(self.vocab_size, p = result[0])
            if result[0][result_idx] < 2 / self.vocab_size:
                '''
                This condition means: If the model has no idea what to pick -> stop picking.
                Also hardcoded values and can be modified.
                A good stop would be adding a lexicon item which stands for stopping the sentence.
                '''
                return seed
            seed_value = fade(seed_value)
            seed += ' ' + self.idx_char[result_idx]
            seed_value[result_idx] += 1

        return seed
def strip_unknown_chars(text):
    '''
    Unifying the text into one format, no unknown chars.
    '''
    text = text.lower()
    text = re.sub('[^a-z \n]', '', text)
    return text

text = ''
with open('../input/headlines.txt', 'r') as f:
    limit = 1500
    i = 0
    for line in f:
        if 'syria' in line:
            text += line + '\n'
            i += 1
            if i == limit:
                break
text = text[:-1]
text = strip_unknown_chars(text)
words = list(set(text.split()))

model = SimpleSequenceGenerator(words)
model.fit(text, 1)

new_lines = []
for i in range(50):
    np.random.shuffle(words)
    seed = words[0] + ' ' + words[1]
    sentence = model.generate(seed, 20)
    with open('generated_text.txt', 'a+') as f:
        f.write(sentence + '\n')

with open('generated_text.txt', 'r') as f:
    limit = 10
    c = 0
    for i in f:
        if c <= limit:
            print(i)
            c += 1

