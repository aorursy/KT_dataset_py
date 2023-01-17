import time

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

import itertools

import random



couplet_list = []



all_chars = set()

char2id = {}

id2char = []





def add_char(char):

    char2id[char] = len(all_chars)

    all_chars.add(char)

    id2char.append(char)





def add_char_in_sentence(sentence):

    for char in sentence:

        if char in all_chars:

            continue

        add_char(char)





add_char('pad')

add_char('sos')

add_char('eos')



with open("../input/couplet/in.txt") as fin, open("../input/couplet/out.txt") as fout:

    # with open("couplet/train/in.txt") as fin, open("couplet/train/out.txt") as fout:

    count = 0

    for line1, line2 in zip(fin, fout):

        line1 = line1.strip().replace(' ', '')

        line2 = line2.strip().replace(' ', '')



        add_char_in_sentence(line1)

        add_char_in_sentence(line2)



        couplet_list.append((line1, line2))

        count += 1





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def indexesFromSentence(sentence):

    return [char2id[char] for char in sentence] + [char2id['eos']]





def zeroPadding(l, fillvalue=char2id['pad']):

    return list(itertools.zip_longest(*l, fillvalue=fillvalue))





def binaryMatrix(l, value=char2id['pad']):

    m = []

    for i, seq in enumerate(l):

        m.append([])

        for token in seq:

            if token == value:

                m[i].append(0)

            else:

                m[i].append(1)

    return m





# Returns padded input sequence tensor and lengths

def inputVar(l):

    indexes_batch = [indexesFromSentence(sentence) for sentence in l]

    lengths = torch.tensor([len(indexes) for indexes in indexes_batch], device=device)

    padList = zeroPadding(indexes_batch)

    padVar = torch.tensor(padList, device=device)

    return padVar, lengths





# Returns padded target sequence tensor, padding mask, and max target length

def outputVar(l):

    indexes_batch = [indexesFromSentence(sentence) for sentence in l]

    max_target_len = max([len(indexes) for indexes in indexes_batch])

    padList = zeroPadding(indexes_batch)

    mask = binaryMatrix(padList)

    mask = torch.ByteTensor(mask).to(device)

    padVar = torch.tensor(padList, device=device)

    return padVar, mask, max_target_len





# Returns all items for a given batch of pairs

def batch2TrainData(pair_batch):

    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)

    input_batch, output_batch = [], []

    for pair in pair_batch:

        input_batch.append(pair[0])

        output_batch.append(pair[1])

    inp, lengths = inputVar(input_batch)

    output, mask, max_target_len = outputVar(output_batch)

    return inp, lengths, output, mask, max_target_len





idx = 0

batch_size = 4000





def get_batch_data():

    global idx

    pair_batch = []



    for i in range(batch_size):

        pair_batch.append(couplet_list[idx])

        idx = (idx + 1) % len(couplet_list)

    return pair_batch





class EncoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = embedding

        self.gru = nn.GRU(embedding.embedding_dim, hidden_size)



    def forward(self, input_seq, input_lengths):

        embedded = self.embedding(input_seq)



        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)



        outputs, hidden = self.gru(packed)  ####



        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)



        return outputs, hidden





class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, embedding):

        super(DecoderRNN, self).__init__()



        self.hidden_size = hidden_size

        self.output_size = output_size



        self.embedding = embedding



        self.gru = nn.GRU(embedding.embedding_dim, hidden_size)



        self.out = nn.Linear(hidden_size, output_size)



    def forward(self, input_step, last_hidden):

        embedded = self.embedding(input_step)



        rnn_output, hidden = self.gru(embedded, last_hidden)



        rnn_output = rnn_output.squeeze(0)



        output = self.out(rnn_output)

        output = F.softmax(output, dim=1)

        # Return output and final hidden state

        return output, hidden





def maskNLLLoss(input, target, mask):

    nTotal = mask.sum()

    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))

    loss = crossEntropy.masked_select(mask).mean()

    loss = loss.to(device)

    return loss, nTotal.item()





teacher_forcing_ratio = 1

clip = 50.0





def train(batch_data):

    input_variable, lengths, target, mask, max_target_len = batch2TrainData(batch_data)



    encoderrnn.zero_grad()

    decoderrnn.zero_grad()



    loss = 0

    n_totals = 0

    all_chars_loss = 0



    encoder_outputs, encoder_hidden = encoderrnn(input_variable, lengths)



    decoder_input = torch.tensor([[char2id['sos'] for _ in range(batch_size)]], device=device)



    decoder_hidden = encoder_hidden



    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False



    if use_teacher_forcing:



        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoderrnn(decoder_input, decoder_hidden)



            decoder_input = target[t].view(1, -1)



            mask_loss, nTotal = maskNLLLoss(decoder_output, target[t], mask[t])



            loss += mask_loss

            all_chars_loss += mask_loss.item() * nTotal

            n_totals += nTotal

    else:

        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoderrnn(decoder_input, decoder_hidden)



            _, topi = decoder_output.topk(1)

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

            decoder_input = decoder_input.to(device)



            mask_loss, nTotal = maskNLLLoss(decoder_output, target[t], mask[t])



            loss += mask_loss

            all_chars_loss += mask_loss.item() * nTotal

            n_totals += nTotal



    loss.backward()



    _ = nn.utils.clip_grad_norm_(encoderrnn.parameters(), clip)

    _ = nn.utils.clip_grad_norm_(decoderrnn.parameters(), clip)



    encoder_optimizer.step()

    decoder_optimizer.step()



    return all_chars_loss / n_totals





def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)





embedding_size = 500

hidden_size = 500



embedding = nn.Embedding(len(all_chars), embedding_size)



encoderrnn = EncoderRNN(hidden_size, embedding).to(device)

decoderrnn = DecoderRNN(hidden_size, len(all_chars), embedding).to(device)



learning_rate = 0.005

encoder_optimizer = optim.Adam(encoderrnn.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adam(decoderrnn.parameters(), lr=learning_rate)





#encoderrnn = torch.load('../input/couplet-para/couplet_encoder_para')

#decoderrnn = torch.load('../input/couplet-para/couplet_decoder_para')





def sample(uplink):

    with torch.no_grad():  # no need to track history in sampling



        output_name = ""



        input_variable, lengths = inputVar([uplink])



        encoder_outputs, encoder_hidden = encoderrnn(input_variable, lengths)



        decoder_input = torch.tensor([[char2id['sos']]], device=device)



        decoder_hidden = encoder_hidden



        for t in range(len(uplink)):

            decoder_output, decoder_hidden = decoderrnn(decoder_input, decoder_hidden)



            _, topi = decoder_output.topk(1)

            decoder_input = torch.tensor([[topi[0][0]]], device=device)

            decoder_input = decoder_input.to(device)



            char = id2char[topi[0][0].item()]

            if (char == 'eos'):

                break

            output_name += char



        return output_name





print_every = 1000

all_losses = []



start = time.time()



best_total_loss = 999

total_loss = 0



iter = 1



while True:



    loss = train(get_batch_data())



    if math.isnan(loss):

        print("nan")

        break



    total_loss += loss



    if iter % print_every == 0:

        average_loss = total_loss / print_every

        total_loss = 0

        print('%s (%d %.4f) %.4f' % (

            timeSince(start), iter * batch_size, ((iter * batch_size) / len(couplet_list)), average_loss))

        all_losses.append(average_loss)

        if average_loss < best_total_loss:

            torch.save(encoderrnn, "couplet_encoder_para")

            torch.save(decoderrnn, "couplet_decoder_para")

            best_total_loss = average_loss



        if average_loss < 0.1:

            break



        if (iter * batch_size) / len(couplet_list) > 15:

            break



        now = time.time()

        s = now - start

        h = math.floor((s / 60) / 60)

        if h > 8:

            break



        temp_str = sample(couplet_list[0][0])

        print(temp_str)



    iter += 1



# import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker

#

# plt.figure()

# plt.plot(all_losses)

# plt.show()





for i in range(len(couplet_list)):

    print(couplet_list[i][0], "--", sample(couplet_list[i][0]), "--", couplet_list[i][1])