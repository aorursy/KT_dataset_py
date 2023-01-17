import warnings

warnings.filterwarnings("ignore")



%timeit

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time, random, sys, os

import sklearn.metrics

from sklearn.model_selection import KFold, StratifiedKFold



import torch

import torch.nn as nn

import torch.backends.cudnn as cudnn

import torch.optim as optim

from torch.utils.data import DataLoader



from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR

from torch.utils.data.dataset import Dataset

from math import cos, pi

import librosa

from scipy.io import wavfile

import torch.nn.functional as F



pd.options.display.max_rows = 500

pd.options.display.max_columns = 500
# set parameters

NUM_FOLD = 5

NUM_CLASS = 264

SEED = 42

NUM_EPOCH =10*5

NUM_CYCLE = 10*5

BATCH_SIZE = 32

LR = [1e-1, 1e-8]

FOLD_LIST = [1]

CROP_LENGTH = 1000000

FEATURE_PATH = '../input/birdsong-recognition/train_audio'

OUTPUT_DIR = "./"



cudnn.benchmark = True

starttime = time.time()
BIRD_CODE = {

    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,

    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,

    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,

    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,

    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,

    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,

    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,

    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,

    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,

    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,

    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,

    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,

    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,

    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,

    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,

    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,

    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,

    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,

    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,

    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,

    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,

    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,

    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,

    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,

    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,

    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,

    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,

    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,

    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,

    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,

    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,

    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,

    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,

    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,

    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,

    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,

    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,

    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,

    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,

    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,

    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,

    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,

    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,

    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,

    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,

    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,

    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,

    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,

    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,

    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,

    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,

    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,

    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263

}



INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}
def compute_gain(sound, fs, min_db=-80.0, mode='RMSE'):

    if fs <= 32000:

        n_fft = 2048

    elif fs <= 44100:

        n_fft = 4096

    else:

        raise Exception('Invalid fs {}'.format(fs))

    stride = n_fft // 2



    gain = []

    for i in range(0, len(sound) - n_fft + 1, stride):

        if mode == 'RMSE':

            g = np.mean(sound[i: i + n_fft] ** 2)

        elif mode == 'A_weighting':

            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])

            power_spec = np.abs(spec) ** 2

            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)

            g = np.sum(a_weighted_spec)

        else:

            raise Exception('Invalid mode {}'.format(mode))

        gain.append(g)



    gain = np.array(gain)

    gain = np.maximum(gain, np.power(10, min_db / 10))

    gain_db = 10 * np.log10(gain)



    return gain_db





def mix(sound1, sound2, r, fs):

    gain1 = np.max(compute_gain(sound1, fs))  # Decibel

    gain2 = np.max(compute_gain(sound2, fs))

    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)

    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    sound = sound.astype(np.float32)



    return sound





class WaveDataset(Dataset):

    def __init__(self, X, y,

                 crop=-1, crop_mode='original', padding=0,

                 mixup=False, scaling=-1, gain=-1,

                 fs=44100,

                 ):

        self.X = X

        self.y = y

        self.crop = crop

        self.crop_mode = crop_mode

        self.padding = padding

        self.mixup = mixup

        self.scaling = scaling

        self.gain = gain

        self.fs = fs



    def preprocess(self, sound):

        for f in self.preprocess_funcs:

            sound = f(sound)



        return sound



    def do_padding(self, snd):

        snd_new = np.pad(snd, self.padding, 'constant')

        return snd_new



    def do_crop(self, snd):

        if self.crop_mode=='random':

            shift = np.random.randint(0, snd.shape[0] - self.crop)

            snd_new = snd[shift:shift + self.crop]

        else:

            snd_new = snd

        return snd_new



    def do_gain(self, snd):

        snd_new = snd * np.power(10, random.uniform(-self.gain, self.gain) / 20.0)

        return snd_new



    def do_scaling(self, snd, interpolate='Nearest'):

        scale = np.power(self.scaling, random.uniform(-1, 1))

        output_size = int(len(snd) * scale)

        ref = np.arange(output_size) / scale

        if interpolate == 'Linear':

            ref1 = ref.astype(np.int32)

            ref2 = np.minimum(ref1+1, len(snd)-1)

            r = ref - ref1

            snd_new = snd[ref1] * (1-r) + snd[ref2] * r

        elif interpolate == 'Nearest':

            snd_new = snd[ref.astype(np.int32)]

        else:

            raise Exception('Invalid interpolation mode {}'.format(interpolate))



        return snd_new



    def do_mixup(self, snd, label, alpha=1):

        idx2 = np.random.randint(0, len(self.X))

        snd2, _ = librosa.core.load(os.path.join(self.X, self.y['ebird_code'][idx2], self.y['filename'][idx2]), res_type="kaiser_fast")

        label2 = np.zeros(264).astype(np.float32)

        label2[BIRD_CODE[self.y['ebird_code'][idx2]]] = 1.0

        if self.scaling!=-1:

            snd2 = self.do_scaling(snd2)

        snd2 = self.do_padding(snd2)

        snd2 = self.do_crop(snd2)



        rate = np.random.beta(alpha, alpha)

        snd_new = mix(snd, snd, rate, self.fs)

        label_new = label * rate + label2 * (1 - rate)

        return snd_new, label_new



    def __getitem__(self, index):

        snd, _ = librosa.core.load(os.path.join(self.X, self.y['ebird_code'][index], self.y['filename'][index]), res_type="kaiser_fast")

        # print(snd.shape)

        label = np.zeros(264).astype(np.float32)

        label[BIRD_CODE[self.y['ebird_code'][index]]] = 1.0

        if self.scaling!=-1:

            snd = self.do_scaling(snd)

        snd = self.do_padding(snd)

        snd = self.do_crop(snd)

        if self.mixup:

            snd, label = self.do_mixup(snd, label)

        if self.gain!=-1:

            snd = self.do_gain(snd)

        snd = snd.reshape([1, 1, -1]).astype(np.float32) / 32768.0

        return snd, label



    def __len__(self):

        return len(self.X)
def _one_sample_positive_class_precisions(scores, truth):

    """Calculate precisions for each true class for a single sample.



    Args:

      scores: np.array of (num_classes,) giving the individual classifier scores.

      truth: np.array of (num_classes,) bools indicating which classes are true.



    Returns:

      pos_class_indices: np.array of indices of the true classes for this sample.

      pos_class_precisions: np.array of precisions corresponding to each of those

        classes.

    """

    num_classes = scores.shape[0]

    pos_class_indices = np.flatnonzero(truth > 0)

    # Only calculate precisions if there are some true classes.

    if not len(pos_class_indices):

        return pos_class_indices, np.zeros(0)

    # Retrieval list of classes for this sample.

    retrieved_classes = np.argsort(scores)[::-1]

    # class_rankings[top_scoring_class_index] == 0 etc.

    class_rankings = np.zeros(num_classes, dtype=np.int)

    class_rankings[retrieved_classes] = range(num_classes)

    # Which of these is a true label?

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)

    retrieved_class_true[class_rankings[pos_class_indices]] = True

    # Num hits for every truncated retrieval list.

    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    # Precision of retrieval list truncated at each hit, in order of pos_labels.

    precision_at_hits = (

            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /

            (1 + class_rankings[pos_class_indices].astype(np.float)))

    return pos_class_indices, precision_at_hits





# All-in-one calculation of per-class lwlrap.



def calculate_per_class_lwlrap(truth, scores):

    """Calculate label-weighted label-ranking average precision.



    Arguments:

      truth: np.array of (num_samples, num_classes) giving boolean ground-truth

        of presence of that class in that sample.

      scores: np.array of (num_samples, num_classes) giving the classifier-under-

        test's real-valued score for each class for each sample.



    Returns:

      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each

        class.

      weight_per_class: np.array of (num_classes,) giving the prior of each

        class within the truth labels.  Then the overall unbalanced lwlrap is

        simply np.sum(per_class_lwlrap * weight_per_class)

    """

    assert truth.shape == scores.shape

    num_samples, num_classes = scores.shape

    # Space to store a distinct precision value for each class on each sample.

    # Only the classes that are true for each sample will be filled in.

    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))

    for sample_num in range(num_samples):

        pos_class_indices, precision_at_hits = (

            _one_sample_positive_class_precisions(scores[sample_num, :],

                                                  truth[sample_num, :]))

        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (

            precision_at_hits)

    labels_per_class = np.sum(truth > 0, axis=0)

    weight_per_class = labels_per_class / float(np.sum(labels_per_class))

    # Form average of each column, i.e. all the precisions assigned to labels in

    # a particular class.

    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /

                        np.maximum(1, labels_per_class))

    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions

    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)

    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples

    #                = np.sum(per_class_lwlrap * weight_per_class)

    return per_class_lwlrap, weight_per_class
class AverageMeter(object):

    """Computes and stores the average and current value"""



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





def cycle(iterable):

    """

    convert dataloader to iterator

    :param iterable:

    :return:

    """

    while True:

        for x in iterable:

            yield x





class CosineLR(_LRScheduler):

    """cosine annealing.

    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):

        self.step_size_min = step_size_min

        self.t0 = t0

        self.tmult = tmult

        self.epochs_since_restart = curr_epoch

        super(CosineLR, self).__init__(optimizer, last_epoch)



    def get_lr(self):

        self.epochs_since_restart += 1



        if self.epochs_since_restart > self.t0:

            self.t0 *= self.tmult

            self.epochs_since_restart = 0



        lrs = [self.step_size_min + (

                0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0)))

               for base_lr in self.base_lrs]



        return lrs
class ConvBnRelu(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1):

        super(ConvBnRelu, self).__init__()

        self.conv_bn_relu = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups, False),

            nn.BatchNorm2d(out_channel),

            nn.ReLU(True))



    def forward(self, x):

        return self.conv_bn_relu(x)





class Flatten(nn.Module):

    def forward(self, x):

        return x.view(x.size()[0], -1)





class EnvNetv2(nn.Module):

    def __init__(self, num_classes=1):

        super(EnvNetv2, self).__init__()

        self.conv1 = ConvBnRelu(1, 32, (1, 64), stride=(1, 2))

        self.conv2 = ConvBnRelu(32, 64, (1, 16), stride=(1, 2))

        self.conv3 = ConvBnRelu(1, 32, (8, 8))

        self.conv4 = ConvBnRelu(32, 32, (8, 8))

        self.conv5 = ConvBnRelu(32, 64, (1, 4))

        self.conv6 = ConvBnRelu(64, 64, (1, 4))

        self.conv7 = ConvBnRelu(64, 128, (1, 2))

        self.conv8 = ConvBnRelu(128, 128, (1, 2))

        self.conv9 = ConvBnRelu(128, 256, (1, 2))

        self.conv10 = ConvBnRelu(256, 256, (1, 2))

        self.maxpool1 = nn.MaxPool2d((1, 64), stride=(1, 64))

        self.maxpool2 = nn.MaxPool2d((5, 3), stride=(5, 3))

        self.maxpool3 = nn.MaxPool2d((1, 2), stride=(1, 2))

        self.gmp = nn.AdaptiveMaxPool2d((10, 1))

        self.flatten = Flatten()

        self.last_linear1 = nn.Sequential(

            nn.Linear(256 * 10, 1024),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),

            nn.ReLU(),

            nn.Dropout(p=0.1),

            nn.Linear(1024, num_classes),

        )

        self.last_linear2 = nn.Sequential(

            nn.Linear(256 * 10, 1024),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),

            nn.ReLU(),

            nn.Dropout(p=0.1),

            nn.Linear(1024, num_classes),

        )



    def forward(self, input):

        h = self.conv1(input)

        h = self.conv2(h)

        h = self.maxpool1(h)

        h = h.transpose(1, 2)

        h = self.conv3(h)

        h = self.conv4(h)

        h = self.maxpool2(h)

        h = self.conv5(h)

        h = self.conv6(h)

        h = self.maxpool3(h)

        h = self.conv7(h)

        h = self.conv8(h)

        h = self.maxpool3(h)

        h = self.conv9(h)

        h = self.conv10(h)

        h = self.gmp(h)

        h = self.flatten(h)

        h = self.last_linear1(h)

        return h
def train(train_loaders, model, optimizer, scheduler, epoch):

    train_loader = train_loaders

    kl_avr = AverageMeter()

    bce_avr = AverageMeter()

    lsigmoid = nn.LogSigmoid().cuda()

    lsoftmax = nn.LogSoftmax(dim=1).cuda()

    softmax = nn.Softmax(dim=1).cuda()

    criterion_kl = nn.KLDivLoss().cuda()

    criterion_bce = nn.BCEWithLogitsLoss().cuda()



    # switch to train mode

    model.train()



    # training

    preds = np.zeros([0, NUM_CLASS], np.float32)

    y_true = np.zeros([0, NUM_CLASS], np.float32)

    for i, (input, target) in enumerate(train_loader):

        # get batches

        input = torch.autograd.Variable(input.cuda())

        target = torch.autograd.Variable(target.cuda())



        # compute output

        output = model(input)

        kl = criterion_kl(lsoftmax(output), target)

        bce = criterion_bce(output, target)

        loss = bce

        pred = softmax(output)

        pred = pred.data.cpu().numpy()



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        scheduler.step(metrics=loss)  # metrics=loss



        # record log

        kl_avr.update(kl.data, input.size(0))

        bce_avr.update(bce.data, input.size(0))

        preds = np.concatenate([preds, pred])

        y_true = np.concatenate([y_true, target.data.cpu().numpy()])



    # calc metric

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)

    lwlrap = np.sum(per_class_lwlrap * weight_per_class)



    return kl_avr.avg.item(), lwlrap, bce_avr.avg.item()





def validate(val_loader, model):

    kl_avr = AverageMeter()

    bce_avr = AverageMeter()

    lsoftmax = nn.LogSoftmax(dim=1).cuda()

    softmax = torch.nn.Softmax(dim=1).cuda()

    criterion_kl = nn.KLDivLoss().cuda()

    criterion_bce = nn.BCEWithLogitsLoss().cuda()



    # switch to eval mode

    model.eval()



    # validate

    preds = np.zeros([0, NUM_CLASS], np.float32)

    y_true = np.zeros([0, NUM_CLASS], np.float32)

    for i, (input, target) in enumerate(val_loader):

        # get batches

        input = torch.autograd.Variable(input.cuda())

        target = torch.autograd.Variable(target.cuda())



        # compute output

        with torch.no_grad():

            output = model(input)

            kl = criterion_kl(lsoftmax(output), target)

            bce = criterion_bce(output, target)

            pred = softmax(output)

            pred = pred.data.cpu().numpy()



        # record log

        kl_avr.update(kl.data, input.size(0))

        bce_avr.update(bce.data, input.size(0))

        preds = np.concatenate([preds, pred])

        y_true = np.concatenate([y_true, target.data.cpu().numpy()])



    # calc metric

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)

    lwlrap = np.sum(per_class_lwlrap * weight_per_class)



    return kl_avr.avg.item(), lwlrap, bce_avr.avg.item()
# load table data

df_train = pd.read_csv("../input/birdsong-recognition/train.csv")

#     + pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")

#     + pd.read_csv("../input/xeno-canto-bird-recordings-extended-n-z/train_extended.csv")

df_test = pd.read_csv("../input/birdcall-check/test.csv")

sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")



# fold splitting

#folds = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_train))))

folds = list(StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(df_train, df_train["ebird_code"]))



# Training

log_columns = ['epoch', 'kl', 'bce', 'lwlrap', 'val_kl', 'val_bce', 'val_lwlrap', 'time']

for fold, (ids_train_split, ids_valid_split) in enumerate(folds):

    if fold+1 not in FOLD_LIST: continue

    print("fold: {}".format(fold + 1))

    train_log = pd.DataFrame(columns=log_columns)



    # build model

    model = EnvNetv2(NUM_CLASS).cuda()



    # prepare data loaders

    df_train_fold = df_train.iloc[ids_train_split].reset_index(drop=True)

    dataset_train = WaveDataset(FEATURE_PATH, df_train_fold,

                                crop=CROP_LENGTH, crop_mode='random', padding=CROP_LENGTH//2,

                                mixup=True, scaling=1.25, gain=6

                                )

    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE,

                                shuffle=True, num_workers=1, pin_memory=True,

                                )



    df_valid = df_train.iloc[ids_valid_split].reset_index(drop=True)

    dataset_valid = WaveDataset(FEATURE_PATH, df_valid, padding=CROP_LENGTH//2)

    valid_loader = DataLoader(dataset_valid, batch_size=1,

                                shuffle=False, num_workers=1, pin_memory=True,

                                )



    # set optimizer and loss

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR[0], momentum = 0.9, nesterov = True)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR[0])

    # scheduler = CosineLR(optimizer, step_size_min=LR[1], t0=len(train_loader) * NUM_CYCLE, tmult=1)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.0001, min_lr=0.00000001)

    # scheduler = CosineAnnealingLR(optimizer, T_max=10)



    # training

    best_val = 999

    for epoch in range(NUM_EPOCH):

        # train for one epoch

        kl, lwlrap, bce = train(train_loader, model, optimizer, scheduler, epoch)



        # evaluate on validation set

        val_kl, val_lwlrap, val_bce = validate(valid_loader, model)



        # print log

        endtime = time.time() - starttime

        print("Epoch: {}/{} ".format(epoch + 1, NUM_EPOCH)

                + "KL: {:.4f} ".format(kl)

                + "BCE: {:.4f} ".format(bce)

                + "LwLRAP: {:.4f} ".format(lwlrap)

                + "Valid KL: {:.4f} ".format(val_kl)

                + "Valid BCE: {:.4f} ".format(val_bce)

                + "Valid LWLRAP: {:.4f} ".format(val_lwlrap)

                + "sec: {:.1f}".format(endtime)

                )



        # save log and weights

        train_log_epoch = pd.DataFrame(

            [[epoch+1, kl, bce, lwlrap, val_kl, val_bce, val_lwlrap, endtime]], columns=log_columns)

        train_log = pd.concat([train_log, train_log_epoch])

        train_log.to_csv("{}/train_log_fold{}.csv".format(OUTPUT_DIR, fold+1), index=False)

        if (epoch+1)%NUM_CYCLE==0:

            torch.save(model.state_dict(), "{}/weight_fold_{}_epoch_{}.pth".format(OUTPUT_DIR, fold+1, epoch+1))

        if best_val > val_bce:

            torch.save(model.state_dict(), "{}/weight_fold_{}_epoch_best.pth".format(OUTPUT_DIR, fold+1))

            best_val = val_bce