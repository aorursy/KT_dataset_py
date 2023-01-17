# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np

import h5py

import json

import torch

#from scipy.misc import imread, imresize

from skimage.io import imread

from skimage.transform import resize as imresize



from tqdm import tqdm

from collections import Counter

from random import seed, choice, sample







def init_embedding(embeddings):

    """

    Fills embedding tensor with values from the uniform distribution.



    :param embeddings: embedding tensor

    """

    bias = np.sqrt(3.0 / embeddings.size(1))

    torch.nn.init.uniform_(embeddings, -bias, bias)





def load_embeddings(emb_file, word_map):

    """

    Creates an embedding tensor for the specified word map, for loading into the model.



    :param emb_file: file containing embeddings (stored in GloVe format)

    :param word_map: word map

    :return: embeddings in the same order as the words in the word map, dimension of embeddings

    """



    # Find embedding dimension

    with open(emb_file, 'r') as f:

        emb_dim = len(f.readline().split(' ')) - 1



    vocab = set(word_map.keys())



    # Create tensor to hold embeddings, initialize

    embeddings = torch.FloatTensor(len(vocab), emb_dim)

    init_embedding(embeddings)



    # Read embedding file

    print("\nLoading embeddings...")

    for line in open(emb_file, 'r'):

        line = line.split(' ')



        emb_word = line[0]

        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))



        # Ignore word if not in train_vocab

        if emb_word not in vocab:

            continue



        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)



    return embeddings, emb_dim





def clip_gradient(optimizer, grad_clip):

    """

    Clips gradients computed during backpropagation to avoid explosion of gradients.



    :param optimizer: optimizer with the gradients to be clipped

    :param grad_clip: clip value

    """

    for group in optimizer.param_groups:

        for param in group['params']:

            if param.grad is not None:

                param.grad.data.clamp_(-grad_clip, grad_clip)





def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,

                    bleu4, is_best):

    """

    Saves model checkpoint.



    :param data_name: base name of processed dataset

    :param epoch: epoch number

    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score

    :param encoder: encoder model

    :param decoder: decoder model

    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning

    :param decoder_optimizer: optimizer to update decoder's weights

    :param bleu4: validation BLEU-4 score for this epoch

    :param is_best: is this checkpoint the best so far?

    """

    state = {'epoch': epoch,

             'epochs_since_improvement': epochs_since_improvement,

             'bleu-4': bleu4,

             'encoder': encoder,

             'decoder': decoder,

             'encoder_optimizer': encoder_optimizer,

             'decoder_optimizer': decoder_optimizer}

    filename = 'checkpoint_' + data_name + '.pth.tar'

    torch.save(state, '../'+filename)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint

    if is_best:

        torch.save(state, '../BEST_' + filename)





class AverageMeter(object):

    """

    Keeps track of most recent, average, sum, and count of a metric.

    """



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





def adjust_learning_rate(optimizer, shrink_factor):

    """

    Shrinks learning rate by a specified factor.



    :param optimizer: optimizer whose learning rate must be shrunk.

    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.

    """



    print("\nDECAYING learning rate.")

    for param_group in optimizer.param_groups:

        param_group['lr'] = param_group['lr'] * shrink_factor

    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))





def accuracy(scores, targets, k):

    """

    Computes top-k accuracy, from predicted and true labels.



    :param scores: scores from the model

    :param targets: true labels

    :param k: k in top-k accuracy

    :return: top-k accuracy

    """



    batch_size = targets.size(0)

    _, ind = scores.topk(k, 1, True, True)

    correct = ind.eq(targets.view(-1, 1).expand_as(ind))

    correct_total = correct.view(-1).float().sum()  # 0D tensor

    return correct_total.item() * (100.0 / batch_size)

import torch

from torch.utils.data import Dataset

import h5py

import json

import os





class CaptionDataset(Dataset):

    """

    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    """



    def __init__(self, data_folder, data_name, split, transform=None):

        """

        :param data_folder: folder where data files are stored

        :param data_name: base name of processed datasets

        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'

        :param transform: image transform pipeline

        """

        self.split = split

        assert self.split in {'TRAIN', 'VAL', 'TEST'}



        # Open hdf5 file where images are stored

        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')

        self.imgs = self.h['images']



        # Captions per image

        self.cpi = self.h.attrs['captions_per_image']



        # Load encoded captions (completely into memory)

        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:

            self.captions = json.load(j)



        # Load caption lengths (completely into memory)

        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:

            self.caplens = json.load(j)



        # PyTorch transformation pipeline for the image (normalizing, etc.)

        self.transform = transform



        # Total number of datapoints

        self.dataset_size = len(self.captions)



    def __getitem__(self, i):

        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)

        if self.transform is not None:

            img = self.transform(img)



        caption = torch.LongTensor(self.captions[i])



        caplen = torch.LongTensor([self.caplens[i]])



        if self.split is 'TRAIN':

            return img, caption, caplen

        else:

            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score

            all_captions = torch.LongTensor(

                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

            return img, caption, caplen, all_captions



    def __len__(self):

        return self.dataset_size

import torch

from torch import nn

import torchvision



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class Encoder(nn.Module):

    """

    Encoder.

    """



    def __init__(self, encoded_image_size=14):

        super(Encoder, self).__init__()

        self.enc_image_size = encoded_image_size



        #vgg19 = torchvision.models.vgg19(pretrained=True)  # pretrained ImageNet VGG19

        

        # load pretrained model vgg19

        vgg19 = torchvision.models.vgg19(pretrained=False) #load model

        vgg19.load_state_dict(torch.load('/kaggle/input/vgg19dcbb9e9dpth/vgg19-dcbb9e9d.pth'))  # load weights

        

        # Remove linear and pool layers (since we're not doing classification)

        modules = list(vgg19.children())[0][:-1]

        self.vgg19 = nn.Sequential(*modules)



        self.fine_tune()



    def forward(self, images):

        """

        Forward propagation.



        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)

        :return: encoded images

        """

        out = self.vgg19(images)  # (batch_size, 512, image_size/32, image_size/32)

        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 512)

        return out



    def fine_tune(self, fine_tune=True):

        """

        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.



        :param fine_tune: Allow?

        """

        for p in self.vgg19.parameters():

            p.requires_grad = False

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4

        # for c in list(self.vgg19.children())[5:]:

        #     for p in c.parameters():

        #         p.requires_grad = fine_tune





class Attention(nn.Module):

    """

    Attention Network.

    """



    def __init__(self, encoder_dim, decoder_dim, attention_dim):

        """

        :param encoder_dim: feature size of encoded images

        :param decoder_dim: size of decoder's RNN

        :param attention_dim: size of the attention network

        """

        super(Attention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image

        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output

        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed

        #self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights



    def forward(self, encoder_out, decoder_hidden):

        """

        Forward propagation.



        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)

        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)

        :return: attention weighted encoding, weights

        """

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)

        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        #att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        alpha = self.softmax(att)  # (batch_size, num_pixels)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)



        return attention_weighted_encoding, alpha





class DecoderWithAttention(nn.Module):

    """

    Decoder.

    """



    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):

        """

        :param attention_dim: size of attention network

        :param embed_dim: embedding size

        :param decoder_dim: size of decoder's RNN

        :param vocab_size: size of vocabulary

        :param encoder_dim: feature size of encoded images

        :param dropout: dropout

        """

        super(DecoderWithAttention, self).__init__()



        self.encoder_dim = encoder_dim

        self.attention_dim = attention_dim

        self.embed_dim = embed_dim

        self.decoder_dim = decoder_dim

        self.vocab_size = vocab_size

        self.dropout = dropout



        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network



        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell

        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution



    def init_weights(self):

        """

        Initializes some parameters with values from the uniform distribution, for easier convergence.

        """

        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.fc.bias.data.fill_(0)

        self.fc.weight.data.uniform_(-0.1, 0.1)



    def load_pretrained_embeddings(self, embeddings):

        """

        Loads embedding layer with pre-trained embeddings.



        :param embeddings: pre-trained embeddings

        """

        self.embedding.weight = nn.Parameter(embeddings)



    def fine_tune_embeddings(self, fine_tune=True):

        """

        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).



        :param fine_tune: Allow?

        """

        for p in self.embedding.parameters():

            p.requires_grad = fine_tune



    def init_hidden_state(self, encoder_out):

        """

        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.



        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)

        :return: hidden state, cell state

        """

        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)

        c = self.init_c(mean_encoder_out)

        return h, c



    def forward(self, encoder_out, encoded_captions, caption_lengths):

        """

        Forward propagation.



        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)

        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)

        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)

        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices

        """



        batch_size = encoder_out.size(0)

        encoder_dim = encoder_out.size(-1)

        vocab_size = self.vocab_size



        # Flatten image

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)



        # Sort input data by decreasing lengths; why? apparent below

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        encoder_out = encoder_out[sort_ind]

        encoded_captions = encoded_captions[sort_ind]



        # Embedding

        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)



        # Initialize LSTM state

        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)



        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>

        # So, decoding lengths are actual lengths - 1

        decode_lengths = (caption_lengths - 1).tolist()



        # Create tensors to hold word predicion scores and alphas

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)



        # At each time-step, decode by

        # attention-weighing the encoder's output based on the decoder's previous hidden state output

        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],

                                                                h[:batch_size_t])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)

            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(

                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),

                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds

            alphas[:batch_size_t, t, :] = alpha



        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
import time

import torch.backends.cudnn as cudnn

import torch.optim

import torch.utils.data

import torchvision.transforms as transforms

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu

from tqdm import tqdm as tqdm



# Data parameters

data_folder = '/kaggle/input/flickr30k-image-captioning/data_processing_first_step'  # folder with data files saved by create_input_files.py

data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files



# Model parameters

emb_dim = 512  # dimension of word embeddings

attention_dim = 512  # dimension of attention linear layers

decoder_dim = 512  # dimension of decoder RNN

dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead



# Training parameters

start_epoch = 0

epochs = 120  # number of epochs to train for (if early stopping is not triggered)

epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU

batch_size = 64

workers = 1  # for data-loading; right now, only 1 works with h5py

encoder_lr = 1e-4  # learning rate for encoder if fine-tuning

decoder_lr = 4e-4  # learning rate for decoder

grad_clip = 5.  # clip gradients at an absolute value of

alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

best_bleu4 = 0.  # BLEU-4 score right now

print_freq = 100  # print training/validation stats every __ batches

fine_tune_encoder = False  # fine-tune encoder?

checkpoint = None  # path to checkpoint, None if none





def main_train():

    """

    Training and validation.

    """



    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map



    # Read word map

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

    with open(word_map_file, 'r') as j:

        word_map = json.load(j)



    # Initialize / load checkpoint

    if checkpoint is None:

        decoder = DecoderWithAttention(attention_dim=attention_dim,

                                       embed_dim=emb_dim,

                                       decoder_dim=decoder_dim,

                                       vocab_size=len(word_map),

                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),

                                             lr=decoder_lr)

        encoder = Encoder()

        encoder.fine_tune(fine_tune_encoder)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),

                                             lr=encoder_lr) if fine_tune_encoder else None



    else:

        checkpoint = torch.load(checkpoint)

        start_epoch = checkpoint['epoch'] + 1

        epochs_since_improvement = checkpoint['epochs_since_improvement']

        best_bleu4 = checkpoint['bleu-4']

        decoder = checkpoint['decoder']

        decoder_optimizer = checkpoint['decoder_optimizer']

        encoder = checkpoint['encoder']

        encoder_optimizer = checkpoint['encoder_optimizer']

        if fine_tune_encoder is True and encoder_optimizer is None:

            encoder.fine_tune(fine_tune_encoder)

            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),

                                                 lr=encoder_lr)



    # Move to GPU, if available

    decoder = decoder.to(device)

    encoder = encoder.to(device)



    # Loss function

    criterion = nn.CrossEntropyLoss().to(device)



    # Custom dataloaders

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(

        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),

        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(

        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),

        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)



    # Epochs

    for epoch in range(start_epoch, epochs):



        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20

        if epochs_since_improvement == 20:

            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:

            adjust_learning_rate(decoder_optimizer, 0.8)

            if fine_tune_encoder:

                adjust_learning_rate(encoder_optimizer, 0.8)



        # One epoch's training

        train(train_loader=train_loader,

              encoder=encoder,

              decoder=decoder,

              criterion=criterion,

              encoder_optimizer=encoder_optimizer,

              decoder_optimizer=decoder_optimizer,

              epoch=epoch)

    

        # One epoch's validation

        recent_bleu4 = validate(val_loader=val_loader,

                                encoder=encoder,

                                decoder=decoder,

                                criterion=criterion)



        # Check if there was an improvement

        is_best = recent_bleu4 > best_bleu4

        best_bleu4 = max(recent_bleu4, best_bleu4)



        if not is_best:

            epochs_since_improvement += 1

            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:

            epochs_since_improvement = 0



        # Save checkpoint

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,

                        decoder_optimizer, recent_bleu4, is_best)





def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):

    """

    Performs one epoch's training.



    :param train_loader: DataLoader for training data

    :param encoder: encoder model

    :param decoder: decoder model

    :param criterion: loss layer

    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)

    :param decoder_optimizer: optimizer to update decoder's weights

    :param epoch: epoch number

    """



    decoder.train()  # train mode (dropout and batchnorm is used)

    encoder.train()



    batch_time = AverageMeter()  # forward prop. + back prop. time

    data_time = AverageMeter()  # data loading time

    losses = AverageMeter()  # loss (per word decoded)

    top5accs = AverageMeter()  # top5 accuracy



    start = time.time()



    # Batches

    for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):

        data_time.update(time.time() - start)



        # Move to GPU, if available

        imgs = imgs.to(device)

        caps = caps.to(device)

        caplens = caplens.to(device)



        # Forward prop.

        imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)



        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

        targets = caps_sorted[:, 1:]



        # Remove timesteps that we didn't decode at, or are pads

        # pack_padded_sequence is an easy trick to do this

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data

        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data



        # Calculate loss

        loss = criterion(scores, targets)



        # Add doubly stochastic attention regularization

        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()



        # Back prop.

        decoder_optimizer.zero_grad()

        if encoder_optimizer is not None:

            encoder_optimizer.zero_grad()

        loss.backward()



        # Clip gradients

        if grad_clip is not None:

            clip_gradient(decoder_optimizer, grad_clip)

            if encoder_optimizer is not None:

                clip_gradient(encoder_optimizer, grad_clip)



        # Update weights

        decoder_optimizer.step()

        if encoder_optimizer is not None:

            encoder_optimizer.step()



        # Keep track of metrics

        top5 = accuracy(scores, targets, 5)

        losses.update(loss.item(), sum(decode_lengths))

        top5accs.update(top5, sum(decode_lengths))

        batch_time.update(time.time() - start)



        start = time.time()



        # Print status

        if i % print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'

                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'

                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),

                                                                          batch_time=batch_time,

                                                                          data_time=data_time, loss=losses,

                                                                          top5=top5accs))





def validate(val_loader, encoder, decoder, criterion):

    """

    Performs one epoch's validation.



    :param val_loader: DataLoader for validation data.

    :param encoder: encoder model

    :param decoder: decoder model

    :param criterion: loss layer

    :return: BLEU-4 score

    """

    decoder.eval()  # eval mode (no dropout or batchnorm)

    if encoder is not None:

        encoder.eval()



    batch_time = AverageMeter()

    losses = AverageMeter()

    top5accs = AverageMeter()



    start = time.time()



    references = list()  # references (true captions) for calculating BLEU-4 score

    hypotheses = list()  # hypotheses (predictions)



    # explicitly disable gradient calculation to avoid CUDA memory error

    # solves the issue #57

    with torch.no_grad():

        # Batches

        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):



            # Move to device, if available

            imgs = imgs.to(device)

            caps = caps.to(device)

            caplens = caplens.to(device)



            # Forward prop.

            if encoder is not None:

                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)



            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

            targets = caps_sorted[:, 1:]



            # Remove timesteps that we didn't decode at, or are pads

            # pack_padded_sequence is an easy trick to do this

            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data

            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data



            # Calculate loss

            loss = criterion(scores, targets)



            # Add doubly stochastic attention regularization

            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()



            # Keep track of metrics

            losses.update(loss.item(), sum(decode_lengths))

            top5 = accuracy(scores, targets, 5)

            top5accs.update(top5, sum(decode_lengths))

            batch_time.update(time.time() - start)



            start = time.time()



            if i % print_freq == 0:

                print('Validation: [{0}/{1}]\t'

                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,

                                                                                loss=losses, top5=top5accs))



            # Store references (true captions), and hypothesis (prediction) for each image

            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -

            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]



            # References

            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder

            for j in range(allcaps.shape[0]):

                img_caps = allcaps[j].tolist()

                img_captions = list(

                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],

                        img_caps))  # remove <start> and pads

                references.append(img_captions)



            # Hypotheses

            _, preds = torch.max(scores_copy, dim=2)

            preds = preds.tolist()

            temp_preds = list()

            for j, p in enumerate(preds):

                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads

            preds = temp_preds

            hypotheses.extend(preds)



            assert len(references) == len(hypotheses)



        # Calculate BLEU-4 scores

        bleu4 = corpus_bleu(references, hypotheses)



        print(

            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(

                loss=losses,

                top5=top5accs,

                bleu=bleu4))



    return bleu4
main_train()
from nltk.stem.porter import PorterStemmer

from nltk.corpus import wordnet

from itertools import chain, product

import json

import numpy as np

from nltk.translate.bleu_score import corpus_bleu







def _generate_enums(hypothesis, reference, preprocess=str.lower):

    # hypothesis_list = list(enumerate(preprocess(hypothesis).split()))

    # reference_list = list(enumerate(preprocess(reference).split()))

    hypothesis_list = list(enumerate([preprocess(str(w)) for w in hypothesis]))

    reference_list = list(enumerate([preprocess(str(w)) for w in reference]))

    

    return hypothesis_list, reference_list





def exact_match(hypothesis, reference):

    hypothesis_list, reference_list = _generate_enums(hypothesis, reference)

    return _match_enums(hypothesis_list, reference_list)







def _match_enums(enum_hypothesis_list, enum_reference_list):

    word_match = []

    string_ = []

    for i in range(len(enum_hypothesis_list))[::-1]:

        for j in range(len(enum_reference_list))[::-1]:

            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:

                word_match.append(

                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])

                )

                # string_.append(enum_hypothesis_list[i][1], enum_reference_list[j][1])

                (enum_hypothesis_list.pop(i)[1], enum_reference_list.pop(j)[1])

                break

    return word_match, enum_hypothesis_list, enum_reference_list





def _enum_stem_match(

    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer()

):

    stemmed_enum_list1 = [

        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list

    ]



    stemmed_enum_list2 = [

        (word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list

    ]



    word_match, enum_unmat_hypo_list, enum_unmat_ref_list = _match_enums(

        stemmed_enum_list1, stemmed_enum_list2

    )



    enum_unmat_hypo_list = (

        list(zip(*enum_unmat_hypo_list)) if len(enum_unmat_hypo_list) > 0 else []

    )



    enum_unmat_ref_list = (

        list(zip(*enum_unmat_ref_list)) if len(enum_unmat_ref_list) > 0 else []

    )



    enum_hypothesis_list = list(

        filter(lambda x: x[0] not in enum_unmat_hypo_list, enum_hypothesis_list)

    )



    enum_reference_list = list(

        filter(lambda x: x[0] not in enum_unmat_ref_list, enum_reference_list)

    )

    return word_match, enum_hypothesis_list, enum_reference_list





def stem_match(hypothesis, reference, stemmer=PorterStemmer()):

    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)

    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)







def _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet):

    word_match = []

    for i in range(len(enum_hypothesis_list))[::-1]:

        hypothesis_syns = set(

            chain(

                *[

                    [

                        lemma.name()

                        for lemma in synset.lemmas()

                        if lemma.name().find("_") < 0

                    ]

                    for synset in wordnet.synsets(enum_hypothesis_list[i][1])

                ]

            )

        ).union({enum_hypothesis_list[i][1]})

        for j in range(len(enum_reference_list))[::-1]:

            if enum_reference_list[j][1] in hypothesis_syns:

                word_match.append(

                    (enum_hypothesis_list[i][0], enum_reference_list[j][0])

                )

                enum_hypothesis_list.pop(i), enum_reference_list.pop(j)

                break

    return word_match, enum_hypothesis_list, enum_reference_list





def wordnetsyn_match(hypothesis, reference, wordnet=wordnet):

    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)

    return _enum_wordnetsyn_match(

        enum_hypothesis_list, enum_reference_list, wordnet=wordnet

    )







def _enum_allign_words(

    enum_hypothesis_list, enum_reference_list, stemmer=PorterStemmer(), wordnet=wordnet

):

    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(

        enum_hypothesis_list, enum_reference_list

    )



    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(

        enum_hypothesis_list, enum_reference_list, stemmer=stemmer

    )



    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(

        enum_hypothesis_list, enum_reference_list, wordnet=wordnet

    )



    return (

        sorted(

            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]

        ),

        enum_hypothesis_list,

        enum_reference_list,

    )





def allign_words(hypothesis, reference, stemmer=PorterStemmer(), wordnet=wordnet):

    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)

    return _enum_allign_words(

        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet

    )







def _count_chunks(matches):

    i = 0

    chunks = 1

    while i < len(matches) - 1:

        if (matches[i + 1][0] == matches[i][0] + 1) and (

            matches[i + 1][1] == matches[i][1] + 1

        ):

            i += 1

            continue

        i += 1

        chunks += 1

    return chunks





def single_meteor_score(

    reference,

    hypothesis,

    preprocess=str.lower,

    stemmer=PorterStemmer(),

    wordnet=wordnet,

    alpha=0.85,

    beta=0.2,

    gamma=0.6,

):

    enum_hypothesis, enum_reference = _generate_enums(

        hypothesis, reference, preprocess=preprocess

    )

    translation_length = len(enum_hypothesis)

    reference_length = len(enum_reference)

    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)

    matches_count = len(matches)

    try:

        precision = float(matches_count) / translation_length

        recall = float(matches_count) / reference_length

        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        chunk_count = float(_count_chunks(matches))

        frag_frac = chunk_count / matches_count

    except ZeroDivisionError:

        return (0., 0, 0, 0,0)

    penalty = gamma * frag_frac ** beta

    single_meteor = (1 - penalty) * fmean

    return (single_meteor, matches_count, reference_length, translation_length, chunk_count)

    #return (single_meteor, precision, recall, penalty)





def meteor_score(

    references,

    hypothesis,

    preprocess=str.lower,

    stemmer=PorterStemmer(),

    wordnet=wordnet,

    alpha=0.85,

    beta=0.2,

    gamma=0.6,

):

    all_single_sentences = []

    for reference in references:

        all_single_sentences.append(single_meteor_score(

                reference,

                hypothesis,

                stemmer=stemmer,

                wordnet=wordnet,

                alpha=alpha,

                beta=beta,

                gamma=gamma,

            ))

    scores_ = [s[0] for s in all_single_sentences]

    idx = scores_.index(max(scores_))

    return all_single_sentences[idx]





def corpus_meteor(references, hypothesis, preprocess=str.lower, stemmer=PorterStemmer(), wordnet=wordnet, alpha=0.85, beta=0.2, gamma=0.6):

    all_score = np.zeros((len(hypothesis), 4))

    i = 0

    for ref, hyp in zip(references, hypothesis):        

        _, all_score[i][0], all_score[i][1],all_score[i][2],all_score[i][3] = meteor_score(ref, hyp)

        i += 1

    

    # all_score_cp = all_score.copy()

    # all_score_cp = all_score_cp**3

    # all_score_cp = all_score_cp.sum(axis=0)



    sum_all_score = all_score.sum(axis=0)

    matches_count = sum_all_score[0]

    len_ref = sum_all_score[1]

    len_hypo = sum_all_score[2]

    chunks = sum_all_score[3]



    if matches_count == 0:

        final_score = 0.

    else:

        # meteor score corpus

        P = matches_count / len_hypo  # precision

        R = matches_count / len_ref   # recall

        fmean = P*R/(alpha*P+(1-alpha)*R)   # Hamonic mean

        frag = chunks / matches_count



        penalty = gamma*(frag**beta) #discouting factor

        

        final_score = (1 - penalty) * fmean

    return final_score
import torch.backends.cudnn as cudnn

import torch.optim

import torch.utils.data

import torchvision.transforms as transforms

from nltk.translate.bleu_score import corpus_bleu

import torch.nn.functional as F

from tqdm import tqdm



# Parameters

data_folder = '/kaggle/input/flickr30k-image-captioning/data_processing_first_step'  # folder with data files saved by create_input_files.py

data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

checkpoint = '../BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

word_map_file = '/kaggle/input/flickr30k-image-captioning/data_processing_first_step/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead



# Load model

checkpoint = torch.load(checkpoint)

decoder = checkpoint['decoder']

decoder = decoder.to(device)

decoder.eval()

encoder = checkpoint['encoder']

encoder = encoder.to(device)

encoder.eval()



# Load word map (word2ix)

with open(word_map_file, 'r') as j:

    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}

vocab_size = len(word_map)

# Normalization transform

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                 std=[0.229, 0.224, 0.225])





def evaluate(beam_size):

    """

    Evaluation



    :param beam_size: beam size at which to generate captions for evaluation

    :return: BLEU-4 score

    """

    # DataLoader

    loader = torch.utils.data.DataLoader(

        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),

        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)



    # TODO: Batched Beam Search

    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!



    # Lists to store references (true captions), and hypothesis (prediction) for each image

    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -

    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = list()

    hypotheses = list()



    references_meteor = list()

    hypotheses_meteor = list()



    # For each image

    for i, (image, caps, caplens, allcaps) in enumerate(

            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size



        # Move to GPU device, if available

        image = image.to(device)  # (1, 3, 256, 256)



        # Encode

        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

        enc_image_size = encoder_out.size(1)

        encoder_dim = encoder_out.size(3)



        # Flatten encoding

        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        num_pixels = encoder_out.size(1)



        # We'll treat the problem as having a batch size of k

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)



        # Tensor to store top k previous words at each step; now they're just <start>

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)



        # Tensor to store top k sequences; now they're just <start>

        seqs = k_prev_words  # (k, 1)



        # Tensor to store top k sequences' scores; now they're just 0

        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)



        # Lists to store completed sequences and scores

        complete_seqs = list()

        complete_seqs_scores = list()



        # Start decoding

        step = 1

        h, c = decoder.init_hidden_state(encoder_out)



        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>

        while True:



            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)



            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)



            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)

            awe = gate * awe



            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)



            scores = decoder.fc(h)  # (s, vocab_size)

            scores = F.log_softmax(scores, dim=1)



            # Add

            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)



            # For the first step, all k points will have the same scores (since same k previous words, h, c)

            if step == 1:

                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)

            else:

                # Unroll and find top scores, and their unrolled indices

                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)



            # Convert unrolled indices to actual indices of scores

            prev_word_inds = top_k_words / vocab_size  # (s)

            next_word_inds = top_k_words % vocab_size  # (s)



            # Add new words to sequences

            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)



            # Which sequences are incomplete (didn't reach <end>)?

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if

                               next_word != word_map['<end>']]

            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))



            # Set aside complete sequences

            if len(complete_inds) > 0:

                complete_seqs.extend(seqs[complete_inds].tolist())

                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)  # reduce beam length accordingly



            # Proceed with incomplete sequences

            if k == 0:

                break

            seqs = seqs[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]

            c = c[prev_word_inds[incomplete_inds]]

            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)



            # Break if things have been going on too long

            if step > 50:

                break

            step += 1



        i = complete_seqs_scores.index(max(complete_seqs_scores))

        seq = complete_seqs[i]



        # References

        img_caps = allcaps[0].tolist()

        img_captions = list(

            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],

                img_caps))  # remove <start> and pads

        references.append(img_captions)





        # Hypotheses

        sentence = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]

        hypotheses.append(sentence)

        

        # meteor

#         references_meteor.append(' '.join([ ' '.join([rev_word_map[w] for w in cap]) for cap in img_captions]))

#         hypotheses_meteor.append(' '.join([rev_word_map[w] for w in sentence]))



        assert len(references) == len(hypotheses)

    

    # Calculate BLEU-4 scores

    bleu4 = corpus_bleu(references, hypotheses)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))

    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))

    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0))

    meteor = corpus_meteor(references, hypotheses)



    return bleu1, bleu2, bleu3, bleu4, meteor

beam_size = 5

blue1, blue2, blue3, blue4, meteor = evaluate(beam_size)

#blue1, blue2, blue3, blue4 = evaluate(beam_size)

print("BLUE-1: ", blue1)

print("BLUE-2: ", blue2)

print("BLUE-3: ", blue3)

print("BLUE-4: ", blue4)

print("meteor: ", meteor)