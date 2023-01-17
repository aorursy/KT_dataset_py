from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-rnns").exists():

    DATA_DIR /= "ucfai-core-sp20-rnns"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-rnns/data

    DATA_DIR = Path("data")
# This is a bit of code to make things work on Kaggle

import os

from pathlib import Path



if os.path.exists("/kaggle/input/ucfai-core-sp20-rnns"):

    DATA_DIR = Path("/kaggle/input/ucfai-core-sp20-rnns")

else:

    DATA_DIR = Path("data/")
# general imports

import numpy as np

import time

import os

import glob



# torch imports

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.backends.cudnn as cudnn

from torch.utils.data import TensorDataset, DataLoader



# tensorboardX

from tensorboardX import SummaryWriter
"""

Loads text from specified path, exits program if the file is not found.

"""

def load_script(path):



    if not os.path.isfile(path):

        print("Error! {} was not found.".format(path))

        return None

    

    with open(path, 'r') as file:

        text = file.read()

    return text



# dictionaries for tokenizing puncuation and converting it back

punctuation_to_tokens = {

                             '!':' ||exclaimation_mark|| ', 

                             ',':' ||comma|| ',

                             '"':' ||quotation_mark|| ',

                             ';':' ||semicolon|| ',

                             '.':' ||period|| ',

                             '?':' ||question_mark|| ',

                             '(':' ||left_parentheses|| ',

                             ')':' ||right_parentheses|| ',

                             '--':' ||dash|| ',

                             '\n':' ||return|| ',

                             ':': ' ||colon|| '

                        }



# reverses aboves dictionary

tokens_to_punctuation = {token.strip(): punc for punc, token in punctuation_to_tokens.items()}



# for all of the puncuation in replace_list, convert it to tokens

def tokenize_punctuation(text):

    replace_list = list(punctuation_to_tokens.keys())

    for char in replace_list:

        text = text.replace(char, punctuation_to_tokens[char])

    return text



# convert tokens back to puncuation

def untokenize_punctuation(text):

    replace_list = list(tokens_to_punctuation.keys())

    for char in replace_list:

        if char == '||left_parentheses||':# added this since left parentheses had an extra space

            text = text.replace(' ' + char + ' ', tokens_to_punctuation[char])

        text = text.replace(' ' + char, tokens_to_punctuation[char])

    return text



"""

Takes text already converted to ints and a sequence length and returns the text split into 

seq_length sequences and generates targets for those sequences

"""

def gen_sequences(int_text, seq_length):

    seq_text = []

    targets = []

    for i in range(0, len(int_text) - seq_length, 1):

        # input sequence

        seq_in = int_text[i:i + seq_length]

        # target for sequence

        seq_out = int_text[i + seq_length]

        seq_text.append([word for word in seq_in])

        targets.append(seq_out)# target is next word after the sequence

        

    return np.array(seq_text, dtype=np.int_), np.array(targets, dtype=np.int_)
from tabulate import tabulate



BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"

EPOCH_TEMPLATE = "Epoch [{} / {}]:"

TEST_TEMPLATE = "Epoch [{}] Test:"



def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):

    """

    Formats an iteration. kwargs should be a variable amount of metrics=vals

    Optional Arguments:

        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])

        epochs(int): total number of epochs

        batch_i(int): current batch iteration

        num_batches(int): total number of batches

        writer(SummaryWriter): tensorboardX summary writer object

        msg(bool): if true, doesn't print but returns the message string



    if curr_epoch and epochs is defined, will format end of epoch iteration

    if batch_i and num_batches is also defined, will define a batch iteration

    if curr_epoch is only defined, defines a validation (testing) iteration

    if none of these are defined, defines a single testing iteration

    if writer is not defined, metrics are not saved to tensorboard

    """

    if curr_epoch is not None:

        if batch_i is not None and num_batches is not None and epochs is not None:

            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)

        elif epochs is not None:

            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)

        else:

            out = TEST_TEMPLATE.format(curr_epoch + 1)

    else:

        out = "Testing Results:"



    floatfmt = []

    for metric, val in kwargs.items():

        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:

            floatfmt.append(".4f")

        elif "accuracy" in metric or "acc" in metric:

            floatfmt.append(".2f")

        else:

            floatfmt.append(".6f")



        if writer and curr_epoch:

            writer.add_scalar(metric, val, curr_epoch)

        elif writer and batch_i:

            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))



    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)



    if msg:

        return out

    print(out)
script_text = load_script(str(DATA_DIR / 'moes_tavern_lines.txt'))

#script_text = load_script(str(DATA_DIR / 'harry-potter.txt'))

# if you want to load in your own data, add it a directory called data (as many text files as you want)

# and uncomment this here: (remember that these stats wont be accurate unless you use the simpsons dataset)

# spript_text = ""

#for script in sort(glob.glob(str(DATA_DIR))):

#    script_text += load_script(script)



print('----------Dataset Stats-----------')

print('Approximate number of unique words: {}'.format(len({word: None for word in script_text.split()})))

scenes = script_text.split('\n\n')

print('Number of scenes: {}'.format(len(scenes)))

sentence_count_scene = [scene.count('\n') for scene in scenes]

print('Average number of sentences in each scene: {:.0f}'.format(np.average(sentence_count_scene)))



sentences = [sentence for scene in scenes for sentence in scene.split('\n')]

print('Number of lines: {}'.format(len(sentences)))

word_count_sentence = [len(sentence.split()) for sentence in sentences]

print('Average number of words in each line: {:.0f}'.format(np.average(word_count_sentence)))
script_text = tokenize_punctuation(script_text) # helper function to convert non-word characters

script_text = script_text.lower()



script_text = script_text.split() # splits the text based on spaces into a list

script_text[:20]
sequence_length = 12

batch_size = 64



int_to_word = {i+1: word for i, word in enumerate(set(script_text))}

# flip word_to_int dict to get int to word

word_to_int = {word: i for i, word in int_to_word.items()}



# convert text to integers using word_to_int

int_script_text = np.array([word_to_int[word] for word in script_text], dtype=np.int_)



# transform int_script_text to sequences of sequence_length and generate targets

int_script_text, targets = gen_sequences(int_script_text, sequence_length)

print("Int Script Text Shape: {}, Targets: {}".format(int_script_text.shape, targets.shape))



# add one since 0 is reserved as whitespace, which counts as another "word" in our vocabulary list

vocab_size = len(word_to_int) + 1



# convert to tensors and define dataset. 

# TensorDataset is a dataset provided by torch that just returns pairs of tensors

dataset = TensorDataset(torch.from_numpy(int_script_text), torch.from_numpy(targets))



# define dataloader for the dataset

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)



print("Number of vocabulary: {}, Dataloader size: {}".format(vocab_size, len(dataloader)))
class LSTM_Model(nn.Module):

    def __init__(self, vocab_size, embed_size, lstm_size=400, num_layers=1, dropout=0.3):

        super(LSTM_Model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # batch_size is first

        self.hidden_dim = lstm_size

        self.vocab_size = vocab_size

        self.num_layers = num_layers

        

        # Define the LSTM of the model as self.LSTM:

        # YOUR CODE HERE

        raise NotImplementedError()

        

        # Define the classifier layer of vocab_size, name it self.classifier

        # YOUR CODE HERE

        raise NotImplementedError()

        

    def forward(self, x, prev_hidden):

        # get batch size for this batch

        batch_size = x.size(0)

        

        out = self.embedding(x)

        out, hidden = self.LSTM(out, prev_hidden)

        # the output from the LSTM needs to be flattened for the classifier, 

        # so reshape output to: (batch_size * seq_len, hidden_dim)

        out = out.contiguous().view(-1, self.hidden_dim)

        

        # get raw predictions (before softmax)

        out = self.classifier(out)

        

        # reshape to split apart the batch_size * seq_len dimension

        out = out.view(batch_size, -1, self.vocab_size)

        

        # only need the output of the layer, so remove the middle seq_len dimension

        out = out[:, -1]

        

        return out, hidden

    

    def init_hidden(self, batch_size):

        # get shape of our weights

        weight = next(self.parameters()).data

    

        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),

                      weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

        

        return hidden
# YOUR CODE HERE

raise NotImplementedError()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device: {}".format(device))

model.to(device)



learn_rate = 0.001



# write out the optimizer and criterion here, using CrossEntropyLoss and the Adam optimizer

# YOUR CODE HERE

raise NotImplementedError()



# torch summary has a bug where it won't work with embedding layers

print(model)



if device == 'cuda':

    # helps with runtime of model, use if you have a constant batch size

    # basically some optimization on NVIDIA's end so it processes batches faster

    cudnn.benchmark = True
# load weights if continuing training

load = torch.load("best.weights.pt")

model.load_state_dict(load["net"])

print("Loaded model from epoch {}.".format(load["epoch"]))
epochs = 7



# view tensorboard with command: tensorboard --logdir=tensorboard_logs

# os.makedirs("tensorboard_logs", exist_ok=True)

# os.makedirs("checkpoints", exist_ok=True)



# ten_board = SummaryWriter('tensorboard_logs/run_{}'.format(start_time))

weight_save_path = 'best.weights.pt'

print_step = len(dataloader) // 20



# put model into training mode so dropout is used.

model.train()



best_loss = 0

start_time = time.time()



for e in range(epochs):

    train_loss = 0

    

    # get inital hidden state

    hidden = model.init_hidden(batch_size)

    # put hidden state to device

    hidden = (hidden[0].to(device), hidden[1].to(device))

    

    for i, data in enumerate(dataloader):

        # make sure you iterate over completely full batches, only

        if len(data[0]) < batch_size:

            break

        

        # grab our data

        inputs, targets = data

        inputs, targets = inputs.to(device), targets.to(device)

        

        # this gets the data tensors for the hidden states

        hidden = tuple([each.data for each in hidden])

        

        optimizer.zero_grad()

        outputs, hidden = model(inputs, hidden)

        

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        

        # print every print_step

        if i % print_step == 0:

            print_iter(curr_epoch=e, epochs=epochs, batch_i=i, num_batches=len(dataloader), loss=train_loss/(i+1))

    

    # print iteration takes the tensorboardX writer and adds the metrics we have to it

    # print_iter(curr_epoch=e, epochs=epochs, writer=writer, loss=train_loss/len(train_dataloader))

    print_iter(curr_epoch=e, epochs=epochs, loss=train_loss/len(dataloader))

    

    if e == 0:

        best_loss = train_loss

    elif train_loss < best_loss:

        print('\nSaving Checkpoint..\n')

        state = {'net': model.state_dict(), 'loss': train_loss, 'epoch': e, 'sequence_length': sequence_length, 'batch_size': batch_size, 'int_to_word': int_to_word, 'word_to_int': word_to_int}

        torch.save(state, weight_save_path)

        best_loss = train_loss



print("Model took {:.2f} minutes to train.".format((time.time() - start_time) / 60))
load = torch.load(weight_save_path)

model.load_state_dict(load["net"])
model.eval()



def sample(prediction, temp=0):

    # temp of 0 is argmax

    if temp <= 0:

        return np.argmax(prediction)

    

    prediction = np.asarray(prediction).astype('float64')

    # perform a log on each prediction and divide it by the temperature

    prediction = np.log(prediction) / temp

    

    # take each prediction and replace it with e^prediction

    expo_prediction = np.exp(prediction)

    

    # divide each prediction by the total sum

    prediction = expo_prediction / np.sum(expo_prediction)

    

    # get new probability distrubtion

    probabilities = np.random.multinomial(1, prediction, 1)

    

    # get argmax of this new distrubtion

    return np.argmax(probabilities)



def pad_sequences(sequence, maxlen, value=0):

    # pad with zeros until sequence length is reached

    while len(sequence) < maxlen:

        sequence = np.insert(sequence, 0, value)

        

    # if sequence is too big, remove first elements

    if len(sequence) > maxlen:

        sequence = sequence[len(sequence) - maxlen:]

    return sequence



#generate new script

def generate_text(seed_text, num_words, temp=0):

    input_text = seed_text

    for _  in range(num_words):

        #tokenize text to ints

        int_text = tokenize_punctuation(input_text)

        int_text = int_text.lower()

        int_text = int_text.split()

        int_text = np.array([word_to_int[word] for word in int_text], dtype=np.int_)

        

        #pad text if it is too short, pads with zeros at beginning of text, so shouldnt have too much noise added

        int_text = pad_sequences(int_text, maxlen=sequence_length)

        int_text = np.expand_dims(int_text, axis=0)

        

        # init hiddens state

        hidden = model.init_hidden(int_text.shape[0])

        hidden = (hidden[0].to(device), hidden[1].to(device))

        

        #predict next word:

        prediction, _ = model(torch.from_numpy(int_text).to(device), hidden)

        prediction = prediction.to("cpu").detach()

        

        # get our probability distruibtion

        prediction = F.softmax(prediction, dim=1).data

        

        prediction = prediction.numpy().squeeze()

        

        output_word = int_to_word[sample(prediction, temp=temp)]

        

        # append to the result

        input_text += ' ' + output_word

        

    #convert tokenized punctuation and other characters back

    result = untokenize_punctuation(input_text)

    return result
#input amount of words to generate, and the seed text, good options are 'Homer_Simpson:', 'Bart_Simpson:', 'Moe_Szyslak:', or other character's names.:

seed = 'Homer_Simpson:'

num_words = 200

temp = 0.5



# print amount of characters specified.

print("Starting seed is: {}\n\n".format(seed))

print(generate_text(seed, num_words, temp=temp))