'''

** Try sentences between 50-150, short ones have higher percentage of punctuations/contractions and so probably lesser semantic value

** No sche, 100e, 500:500, len(sent)>50 (compare with V31)



V35 Sche(d-lr), 100e, 500:500, 50<=len(quote)<=200 (try if the d-lr helps with overfitting)

V34 Sche(d-lr), 100e, 100:100, 50<=len(quote)<=200 (compare with V30, V25)

V31 No sche, 100e, 500:500, 50<=len(quote)<=200 (compare with V25; V30 forgot to change weights size; term early cus overfit at<40e)

V30 No sche, 100e, 100:100, 50<=len(quote)<=200 (compare with V25)

V29 Sche (use loss from criterion instead of evaluate), 100e, 500:500 (compared with V25 and V22... previous scheduler configured wrongly, it converges too early)

V28 Sched, 300e, 100:100 (loss curve looked identical to V26)

V26 No sched, 300e, 100:100 val_l near 5

V25 No sched, 100e, 500:500, val_l <5, overfit (lr schema no eff on converge)

V24 No sched, 8e, 500:500

V23 Sche, 100e, 200:200, val_l near 5 and decresing (smaller size = longer to converge)

V22 Sche, 100e, 500:500, val_l <5, overfit

'''
from IPython.display import Image

from IPython.core.display import HTML



from IPython.display import clear_output
from collections import namedtuple

import numpy as np

from tqdm import tqdm

import pandas as pd

from gensim.corpora import Dictionary



import torch

from torch import nn, optim, tensor, autograd

from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler



from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(12, 8)})

torch.manual_seed(42)
try: # Use the default NLTK tokenizer.

    from nltk import word_tokenize, sent_tokenize 

    # Testing whether it works. 

    # Sometimes it doesn't work on some machines because of setup issues.

    print(word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0]))

except: # Use a naive sentence tokenizer and toktok.

    import re

    from nltk.tokenize import ToktokTokenizer

    # See https://stackoverflow.com/a/25736515/610569

    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)

    # Use the toktok tokenizer that requires no dependencies.

    toktok = ToktokTokenizer()

    word_tokenize = word_tokenize = toktok.tokenize
#Calling data source which had been preloaded into Kaggle workspace

file ="../input/author-quote.txt.csv"

text_df = pd.read_csv(file,delimiter='\t',header=None)



#Omitting the author names present in first column

text_df=text_df[1]



#text_df.head()

len(text_df) #36165
#see distribution of quote length

distribution = [len(quote) for quote in text_df]

plt.plot(sorted(distribution))

plt.show()

#see distribution of sentence length

distribution = [len(sent) for quote in text_df for sent in sent_tokenize(quote) ]

plt.plot(sorted(distribution))

plt.show()
tokens = [word_tokenize(quote.lower()) for quote in text_df for sent in sent_tokenize(quote) if len(sent) >=50 and len(sent) <= 200]



#Checking how the indexing works in Dictionary

#tok_dict = Dictionary(tokens)

#[(tok_dict[i],tok_dict.dfs[i]) for i in tok_dict.dfs]
class KilgariffDataset(nn.Module):

    def __init__(self, texts):

        self.texts = texts

        

        # Initialize the vocab 

        special_tokens = {'<pad>': 0, '<unk>':1, '<s>':2, '</s>':3}

        self.vocab = Dictionary(texts)

        self.vocab.patch_with_special_tokens(special_tokens)

        

        # Keep track of the vocab size.

        self.vocab_size = len(self.vocab)

        

        # Keep track of how many data points.

        self._len = len(texts)

        

        # Find the longest text in the data.

        self.max_len = max(len(txt) for txt in texts) 

        

    def __getitem__(self, index):

        vectorized_sent = self.vectorize(self.texts[index])

        x_len = len(vectorized_sent)

        # To pad the sentence:

        # Pad left = 0; Pad right = max_len - len of sent.

        pad_dim = (0, self.max_len - len(vectorized_sent))

        vectorized_sent = F.pad(vectorized_sent, pad_dim, 'constant')

        return {'x':vectorized_sent[:-1], 

                'y':vectorized_sent[1:], 

                'x_len':x_len}

    

    def __len__(self):

        return self._len

    

    def vectorize(self, tokens, start_idx=2, end_idx=3):

        """

        :param tokens: Tokens that should be vectorized. 

        :type tokens: list(str)

        """

        # See https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary.doc2idx 

        # Lets just cast list of indices into torch tensors directly =)

        

        vectorized_sent = [start_idx] + self.vocab.doc2idx(tokens) + [end_idx]

        return torch.tensor(vectorized_sent)

    

    def unvectorize(self, indices):

        """

        :param indices: Converts the indices back to tokens.

        :type tokens: list(int)

        """

        return [self.vocab[i] for i in indices]
kilgariff_data = KilgariffDataset(tokens)

print ('Count of quotes: ' + str(kilgariff_data.vocab.num_docs))

print ('Count of UNIQUE words: ' + str(len(kilgariff_data.vocab)))

print ('Count of TOTAL words: ' + str(kilgariff_data.vocab.num_pos))
'''

dataloader = DataLoader(dataset=kilgariff_data, batch_size=10)

for data_dict in dataloader:

    # Sort indices of data in batch by lengths.

    

    sorted_indices = np.array(data_dict['x_len']).argsort()[::-1].tolist()

    data_batch = {name:_tensor[sorted_indices]

                  for name, _tensor in data_dict.items()}

    

    print(data_dict)

    break

'''
class Generator(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):

        super(Generator, self).__init__()



        # Initialize the embedding layer with the 

        # - size of input (i.e. no. of words in input vocab)

        # - no. of hidden nodes in the embedding layer

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        

        # Initialize the RNN with the 

        # - size of the input (i.e. embedding layer)

        # - size of the hidden layer 

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)

        

        # Initialize the "classifier" layer to map the RNN outputs

        # to the vocabulary. Remember we need to -1 because the 

        # vectorized sentence we left out one token for both x and y:

        # - size of hidden_size of the RNN output.

        # - size of vocabulary

        self.classifier = nn.Linear(hidden_size, vocab_size)

        

    def forward(self, inputs, use_softmax=False, hidden=None):

        # Look up for the embeddings for the input word indices.

        embedded = self.embedding(inputs)

        # Put the embedded inputs into the RNN.

        output, hidden = self.lstm(embedded, hidden)

        

        # Matrix manipulation magic.

        batch_size, sequence_len, hidden_size = output.shape

        # Technically, linear layer takes a 2-D matrix as input, so more manipulation...

        output = output.contiguous().view(batch_size * sequence_len, hidden_size)

        # Apply dropout.

        output = F.dropout(output, 0.2)

        # Put it through the classifier

        # And reshape it to [batch_size x sequence_len x vocab_size]

        output = self.classifier(output).view(batch_size, sequence_len, -1)

        

        return (F.softmax(output,dim=2), hidden) if use_softmax else (output, hidden)
# Wraps hidden states in new Tensors, to detach them from their history.

def repackage_hidden(h):    

    if isinstance(h, torch.Tensor):

        return h.detach()

    

# Training routine.

#def train(num_epochs, dataloader,val_loader, model, criterion, optimizer, scheduler):

def train(num_epochs, dataloader,val_loader, model, criterion, optimizer):

    losses = []

    val_losses = []

    plt.ion()

    hidden = None

    model.train()

    for e in range(num_epochs):

        total_loss = 0.0

        for batch in tqdm(dataloader):   

            hidden = repackage_hidden(hidden)

            # Zero gradient.

            optimizer.zero_grad()

            x = batch['x'].to(device)

            x_len = batch['x_len'].to(device)

            y = batch['y'].to(device) 

            # Feed forward. 

            output, hidden = model(x, use_softmax=False, hidden=hidden)

            # Compute loss:

            loss = criterion(output.permute(0, 2, 1), y)

            total_loss += loss.float().data

            loss.backward()

            optimizer.step()

                

        losses.append(total_loss/float(len(dataloader)))  

        if e % 10 == 0:

            model_filename = 'model_{0:05d}.pth'.format(e)

            torch.save(model.state_dict(), model_filename)

        torch.save(model.state_dict(), 'model.pth')

        clear_output(wait=True)

        plt.plot(losses, label='training_losses')

        

        _v=evaluate(val_loader, model, criterion).item()

        val_losses.append(_v)

        

        if _v < val_losses[-1]:

            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 1.5

            

        plt.plot(val_losses, label='val_losses')

        

        plt.legend()

        plt.ylabel('Entropy Loss')

        plt.xlabel('Epoch')

        plt.savefig('plot.png')

        plt.show()

        

        #scheduler.step(loss)

        model.train()       

        

    print (val_losses)



def initialize_data_model_optim_loss(hyperparams,kilgariff_data):

    # Creating data indices for training and test splits:

    dataset_size = len(kilgariff_data)

    indices = list(range(dataset_size))

    # test and validation set each occupies 10% of entire dataset

    # train:test:val = 8:1:1

    n_batch = dataset_size//hyperparams.batch_size

    split1 = int(n_batch*0.8*hyperparams.batch_size)

    split2 = split1+int(n_batch*0.1*hyperparams.batch_size)

    endsplit = n_batch*hyperparams.batch_size

    

    np.random.seed(1234)

    np.random.shuffle(indices)

    train_indices, test_indices, val_indices = indices[:split1], indices[split1:split2],indices[split2:endsplit]

    

    #print train loader size and test loader size

    print('total datasize :',(dataset_size), 

          ', trainsize :', len(train_indices),

          ', testsize ::', len(test_indices),

          ', valsize ::', len(val_indices))



    # Creating random data samplers and loaders:

    train_sampler = SubsetRandomSampler(train_indices)

    test_sampler = SubsetRandomSampler(test_indices)

    val_sampler = SubsetRandomSampler(val_indices)

    

    # Create loader for training set with one certian batch size and random sampler

    train_loader = torch.utils.data.DataLoader(kilgariff_data, batch_size=hyperparams.batch_size, 

                                               sampler=train_sampler)

    # Create test loader for test set without batch size

    test_loader = torch.utils.data.DataLoader(kilgariff_data, sampler=test_sampler) 

    val_loader = torch.utils.data.DataLoader(kilgariff_data, sampler=val_sampler) 



    # Loss function.

    criterion = hyperparams.loss_func(ignore_index=kilgariff_data.vocab.token2id['<pad>'], 

                                      reduction='mean')



    # Model.

    model = Generator(len(kilgariff_data.vocab), hyperparams.embed_size, 

                      hyperparams.hidden_size, hyperparams.num_layers).to(device)



    # Optimizer.

    optimizer = hyperparams.optimizer(model.parameters(), lr=hyperparams.learning_rate)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience =2, verbose =True, min_lr=1e-9)

    

    #return train_loader, test_loader, val_loader, model, optimizer, scheduler, criterion

    return train_loader, test_loader, val_loader, model, optimizer, criterion
import math



# evaluate the model.

def evaluate(dataloader, model, criterion):

    model.eval()

    total_loss = 0

    hidden = None

    with torch.no_grad():

        for items in tqdm(dataloader):

            x = items['x'].to(device)

            x_len = items['x_len'].to(device)

            y = items['y'].to(device)

            # Feed forward. 

            output, hidden = model(x, use_softmax=False, hidden=hidden)

            # Compute loss:

            total_loss += criterion(output.permute(0, 2, 1), y).float().data

    return total_loss/float(len(dataloader))
def generate_example(model, temperature=1, max_len=50, hidden_state=None):

    start_token, start_idx = '<s>', 2

    # Start state.

    inputs = torch.tensor(kilgariff_data.vocab.token2id[start_token]).unsqueeze(0).unsqueeze(0).to(device)



    sentence = [start_token]

    i = 0

    while i < max_len and sentence[-1] not in ['</s>', '<pad>']:

        i += 1

        

        embedded = model.embedding(inputs)

        output, hidden_state = model.lstm(embedded, hidden_state)



        batch_size, sequence_len, hidden_size = output.shape

        output = output.contiguous().view(batch_size * sequence_len, hidden_size)    

        output = model.classifier(output).view(batch_size, sequence_len, -1).squeeze(0)

        

        word_weights = output.div(temperature).exp().cpu()

        if len(word_weights.shape) > 1:

            word_weights = word_weights[-1] # Pick the last word.    

        word_idx = torch.multinomial(word_weights, 1).view(-1)

        

        sentence.append(kilgariff_data.vocab[int(word_idx)])

        

        inputs = tensor([kilgariff_data.vocab.token2id[word] for word in sentence]).unsqueeze(0).to(device)

    print(' '.join(sentence))
_hyper = ['embed_size', 'hidden_size', 'num_layers',

          'loss_func', 'learning_rate', 'optimizer', 'batch_size']

Hyperparams = namedtuple('Hyperparams', _hyper)



hyperparams = Hyperparams(embed_size=500, hidden_size=500, num_layers=1,

                          loss_func=nn.CrossEntropyLoss,

                          learning_rate=0.00001, optimizer=optim.Adam, batch_size=64)



#train_loader, test_loader,val_loader, model, optimizer, scheduler, criterion = initialize_data_model_optim_loss(hyperparams,kilgariff_data)

train_loader, test_loader,val_loader, model, optimizer, criterion = initialize_data_model_optim_loss(hyperparams,kilgariff_data)#
[(param,tensors.shape) for param,tensors in model.named_parameters()]
#train(100, train_loader, val_loader, model, criterion, optimizer, scheduler)

train(100, train_loader, val_loader, model, criterion, optimizer)
# load best model pre-trained

# model_filename = 'model_{0:05d}.pth'.format(1)

model.load_state_dict(torch.load('model.pth'))

for _ in range(50):

    generate_example(model)
for _ in range(50):

    generate_example(model, temperature=0.7)
for _ in range(50):

    generate_example(model, temperature=0.3)
test_loss = evaluate(test_loader, model, criterion)

print('| End of training | test loss {:5.2f} | test perplexity {:8.2f}'.format(

    test_loss, math.exp(test_loss)))
