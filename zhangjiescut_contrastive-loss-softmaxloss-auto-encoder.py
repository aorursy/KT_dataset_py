!pip install lap
# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import time
TRAIN_DF = '../input/humpback-whale-identification/train.csv'
SUB_Df = '../input/humpback-whale-identification/sample_submission.csv'
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
P2H = '../input/metadata/p2h.pickle'
P2SIZE = '../input/metadata/p2size.pickle'
BB_DF = "../input/metadata/bounding_boxes.csv"
img_shape = (224, 224, 1)  # The image shape used by the model
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy
def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p
def get_alldata():
    tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
    submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
    join = list(tagged.keys()) + submit
    return tagged, submit, join

def get_p2size(join):
    if isfile(P2SIZE):
        print("P2SIZE exists.")
        with open(P2SIZE, 'rb') as f:
            p2size = pickle.load(f)
    else:
        p2size = {}
        for p in tqdm(join):
            size = pil_image.open(expand_path(p)).size
            p2size[p] = size
    return p2size

def get_p2bb():
    p2bb = pd.read_csv(BB_DF).set_index("Image")
    return p2bb

# get data for simaese network and corresponding whale id, exclude new_whale
def get_p2ws(tagged):
    new_whale = 'new_whale'
    p2ws = {}
    for p, w in tagged.items():
        if w != new_whale:
            if p not in p2ws:
                p2ws[p] = []
            if w not in p2ws[p]:
                p2ws[p].append(w)
    return p2ws

# this is used for validation
def get_new_whale(tagged):
    new_whales = []
    for p, w in tagged.items():
        if w == 'new_whale':
            new_whales.append(p)
    np.random.seed(44)
    np.random.shuffle(new_whales)
    np.random.seed(None)
    return new_whales


def get_w2ps(p2ws):
    w2ps = {}
    for p, ws in p2ws.items():
        for w in ws:
            if w not in w2ps:
                w2ps[w] = []
            if p not in w2ps[w]:
                w2ps[w].append(p)
    return w2ps


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img


# convert whale id to numbers for softmax loss
def get_w2idx(train_soft, w2ps):
    train_soft_set = sorted(set(train_soft))
    w2ts_soft = {}
    for w, ps in w2ps.items():
        for p in ps:
            if p in train_soft_set:
                if w not in w2ts_soft:
                    w2ts_soft[w] = []
                if p not in w2ts_soft[w]:
                    w2ts_soft[w].append(p)
    for w, ts in w2ts_soft.items():
        w2ts_soft[w] = np.array(ts)

    w2idx = {}
    for idx, w in enumerate(w2ts_soft.keys()):
        if w not in w2idx:
            w2idx[w] = idx

    idx2w = {}
    for w, idx in w2idx.items():
        idx2w[idx] = w
    return w2ts_soft, w2idx, train_soft_set, idx2w


# resize image with unchanged aspect ratio using padding
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), pil_image.BICUBIC)
    new_image = pil_image.new('L', size)  # , (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def read_cropped_image(p, p2size, p2bb, augment):
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y

    img = read_raw_image(p).convert('L')

    bbox = (x0, y0, x1, y1)
    img = img.crop(bbox)
    img = letterbox_image(img, img_shape[:2])
    img = np.array(img).reshape(img_shape)
    img = img.astype(np.float32)
    if augment:
        theta = np.random.uniform(-10, 10)  # random rotation
        h, w = img.shape[0], img.shape[1]
        tx = np.random.uniform(-0.1, 0.1) * h
        ty = np.random.uniform(-0.05, 0.05) * w  # random shift
        zx, zy = np.random.uniform(0.9, 1.1, 2)  # random zoom
        shear = np.random.uniform(-10, 10)  # random shear
        img = apply_affine_transform(img, theta, tx, ty, shear, zx, zy)

    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_for_training(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, True)


def read_for_validation(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, False)


def split_train_test(w2ps):
    np.random.seed(44)
    train = []
    test = []
    train_soft = []
    for ps in w2ps.values():
        if len(ps) >= 8:
            np.random.shuffle(ps)
            test += ps[-3:]
            train += ps[:-3]
            train_soft += ps[:-3]
            #train += ps
            #train_soft += ps
        elif len(ps) > 1:
            train += ps
            train_soft += ps
        else:
            train_soft += ps
    np.random.seed(None)
    train_set = sorted(set(train))
    test_set = sorted(set(test))
    np.random.shuffle(train)
    np.random.shuffle(train_soft)

    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, ps in w2ps.items():
        for p in ps:
            if p in train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if p not in w2ts[w]:
                    w2ts[w].append(p)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    w2vs = {}  # Associate the image ids from train to each whale id.
    for w, ps in w2ps.items():
        for p in ps:
            if p in test_set:
                if w not in w2vs:
                    w2vs[w] = []
                if p not in w2vs[w]:
                    w2vs[w].append(p)
    for w, vs in w2vs.items():
        w2vs[w] = np.array(vs)

    t2i = {}  # The position in train of each training image id
    for i, t in enumerate(train):
        t2i[t] = i

    v2i = {}
    for i, v in enumerate(test):
        v2i[v] = i

    return train, test, train_set, test_set, w2ts, w2vs, t2i, v2i, train_soft


def map_per_image(label, predictions):
    try:
        return 1.0 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m

# for cv validation
def val_score(test, threshold, known, p2ws, score_val):
    new_whale = 'new_whale'
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    predictions = []
    for i, p_ in enumerate(tqdm(test)):
        t = []
        s = set()
        a = score_val[i, :]
        for j in list(reversed(np.argsort(a))):
            p = known[j]
            if a[j] < threshold and new_whale not in s:
                pos[len(t)] += 1
                s.add(new_whale)
                t.append(new_whale)
                if len(t) == 5:
                    break
            for w in p2ws[p]:
                assert w != new_whale
                if w not in s:
                    if a[j] > 1.0:
                        vtop += 1
                    elif a[j] >= threshold:
                        vhigh += 1
                    s.add(w)
                    t.append(w)
                    if len(t) == 5:
                        break
            if len(t) == 5:
                break
        if new_whale not in s:
            pos[5] += 1
        assert len(t) == 5 and len(s) == 5
        predictions.append(t[:5])
    return predictions


def get_random_test_data(test, w2vs, v2i):
    np.random.seed(10)
    score = -1 * np.random.random_sample(size=(len(test), len(test)))
    np.random.seed(None)
    for vs in w2vs.values():
        idxs = [v2i[v] for v in vs]
        for i in idxs:
            for j in idxs:
                score[i, j] = 10000.0
    match = []
    unmatch = []
    _, _, x = lapjv(score)  # Solve the linear assignment problem
    y = np.arange(len(x), dtype=np.int32)

    # Compute a derangement for matching whales
    for vs in w2vs.values():
        d = vs.copy()
        while True:
            random.shuffle(d)
            if not np.any(vs == d):
                break
        for ab in zip(vs, d):
            match.append(ab)

    # Construct unmatched whale pairs from the LAP solution.
    for i, j in zip(x, y):
        if i == j:
            print(score)
            print(x)
            print(y)
            print(i, j)
        assert i != j
        unmatch.append((test[i], test[j]))

    # print(len(self.match), len(train), len(self.unmatch), len(train))
    assert len(match) == len(test) and len(unmatch) == len(test)
    return match, unmatch
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, MaxPooling2D, Reshape,Multiply
from keras.models import Model
import keras.backend as K
from keras.layers import Dropout, UpSampling2D
import tensorflow as tf


def subblock(x, filter, block, num, **kwargs):
    #y = BatchNormalization()(x)
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(x)  # Reduce the number of features to 'filter'
    #y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y)  # Extend the feature field
    #y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y)  # no activation # Restore the number of original features
    #y = BatchNormalization()(y)

    spatial_attention = Conv2D(K.int_shape(y)[-1] // 2, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                               name=block + '_' + str(num) + 'sa_conv1')(y)
    spatial_attention = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', name=block + '_' + str(num) + 'sa_conv2')(spatial_attention)

    channel_attention = GlobalAveragePooling2D(name=block + '_' + str(num) + 'ca_gmp')(y)
    channel_attention = Reshape(target_shape=(-1, K.int_shape(channel_attention)[-1]), name=block + '_' + str(num) + 'ca_reshape1')(channel_attention)
    channel_attention = Dense(K.int_shape(channel_attention)[-1], activation='sigmoid', name=block + '_' + str(num) + 'ca_dense1')(channel_attention)
    channel_attention = Dense(K.int_shape(channel_attention)[-1], activation='sigmoid', name=block + '_' + str(num) + 'ca_dense2')(channel_attention)
    channel_attention = Reshape(target_shape=(-1, 1, K.int_shape(channel_attention)[-1]), name=block + '_' + str(num) + 'ca_reshape2')(channel_attention)

    y = Multiply(name=block + '_' + str(num) + 'ml1')([y, channel_attention])
    y = Multiply(name=block + '_' + str(num) + 'ml2')([y, spatial_attention])

    y = Add()([x, y])  # Add the bypass connection
    y = Activation('relu')(y)
    return y


def decoder_model(inp):
    kwargs = {'padding': 'same'}
    net = Dense(4096, activation=None, name='dec_den_1')(inp)
    net = Reshape(target_shape=(4, 4, 256), name='dec_rs_1')(net)

    net = UpSampling2D(name='dec_us1')(net)
    net = Conv2D(128, (2, 2), padding='valid', activation='relu', name='dec_conv0')(net)
    net = Conv2D(128, (3, 3), padding='same', activation='relu', name='dec_conv1')(net)
    for i in range(3):
        net = subblock(net, 128, 'dec_1', i, **kwargs)

    net = UpSampling2D(name='dec_us2')(net)
    net = Conv2D(64, (3, 3), padding='same', activation='relu', name='dec_conv2')(net)
    for i in range(3):
        net = subblock(net, 64, 'dec_2', i, **kwargs)

    net = UpSampling2D(name='dec_us3')(net)
    net = Conv2D(32, (3, 3), padding='same', activation='relu', name='dec_conv3')(net)
    for i in range(3):
        net = subblock(net, 32, 'dec_3', i, **kwargs)

    net = UpSampling2D(name='dec_us4')(net)
    net = Conv2D(16, (3, 3), padding='same', activation='relu', name='dec_conv4')(net)
    for i in range(3):
        net = subblock(net, 16, 'dec_4', i, **kwargs)

    net = UpSampling2D(name='dec_us5')(net)
    net = Conv2D(3, (3, 3), padding='same', activation='relu', name='dec_conv5')(net)
    for i in range(3):
        net = subblock(net, 3, 'dec_5', i, **kwargs)

    net = UpSampling2D(name='dec_us6')(net)
    net = Conv2D(1, (3, 3), padding='same', activation='relu', name='dec_conv6')(net)
    net = Conv2D(1, (3, 3), padding='same', activation=None, name='dec_conv7')(net)

    return net


def build_model(lr, l2, img_shape=(224, 224, 1), activation='sigmoid'):
    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape)  
    x = Conv2D(64, (9, 9), strides=2, activation='relu', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  
    for _ in range(2):
        #x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  
    #x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu', **kwargs)(x)  
    #x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 64, '1', i, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) 
    #x = BatchNormalization()(x)
    x = Conv2D(256, (1, 1), activation='relu', **kwargs)(x) 
    #x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 64, '2', i,  **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = BatchNormalization()(x)
    x = Conv2D(384, (1, 1), activation='relu', **kwargs)(x) 
    #x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 96, '3', i,  **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  
    #x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu', **kwargs)(x)
    #x = BatchNormalization()(x)
    for i in range(4):
        x = subblock(x, 128, '4', i,  **kwargs)

    x = GlobalMaxPooling2D()(x) 
    branch_model = Model(inp, x)

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:])
    xb_inp = Input(shape=branch_model.output_shape[1:])

    x1 = Lambda(lambda x: x[0] * x[1])([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1])([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x))(x3)
    x = Concatenate()([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')
    
    # for classification
    x_inp_ = Input(shape=branch_model.output_shape[1:])
    x_all = Dropout(0.5)(x_inp_)
    x_all = Dense(512, activation='relu', kernel_regularizer=regul)(x_all)
    x_all = Dropout(0.5)(x_all)
    x_all = Dense(512, activation='relu', kernel_regularizer=regul)(x_all)
    x_all = Dense(5004, activation='softmax')(x_all)
    soft_model = Model(x_inp_, x_all, name='soft')

    ########################
    #  auto encoder
    ########################
    dec_inp = Input(shape=branch_model.output_shape[1:])
    net = decoder_model(dec_inp)
    dec_model = Model(dec_inp, net, name='decoder')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    img_c = Input(shape=img_shape)  # softmax
    img_d = Input(shape=img_shape)  # decoder
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    xc = branch_model(img_c)  # softmax
    xd = branch_model(img_d)  # decoder

    y_decoder = dec_model(xd)  # decoder

    x = head_model([xa, xb])
    y_softmax = soft_model(xc)
    model = Model([img_a, img_b, img_c, img_d], [x, y_softmax, y_decoder])
    model.compile(optim, loss=['binary_crossentropy', 'categorical_crossentropy', decoder_loss], metrics=['acc'],
                  loss_weights=[1, 0.5, 0.5])
    return model, branch_model, head_model, dec_model, soft_model


def decoder_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))
import time
import os
import sys
from keras.callbacks import Callback
from keras.utils import Sequence
tagged, submit, join = get_alldata()

p2size = get_p2size(join)

p2bb = get_p2bb()

p2ws = get_p2ws(tagged)

new_whales = get_new_whale(tagged)

w2ps = get_w2ps(p2ws)

train, test, train_set, test_set, w2ts, w2vs, t2i, v2i, train_soft = split_train_test(w2ps)

match_test, unmatch_test = get_random_test_data(test, w2vs, v2i)

w2ts_soft, w2idx, train_soft_set, idx2w = get_w2idx(train_soft, w2ps)

model, branch_model, head_model, dec_model, soft_model = build_model(64e-5, 0.0002)
new_whale = 'new_whale'

p2wts = {}
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        if p in train_set:
            if p not in p2wts:
                p2wts[p] = []
            if w not in p2wts[p]:
                p2wts[p].append(w)
known = sorted(list(p2wts.keys()))

# Dictionary of picture indices
kt2i = {}
for i, p in enumerate(known): kt2i[p] = i

class TestingData(Sequence):
    def __init__(self, batch_size=64):
        super(TestingData, self).__init__()
        self.batch_size = batch_size
        self.match = match_test
        self.unmatch = unmatch_test
        # np.random.seed(10)
        # self.score = -1 * np.random.random_sample(size=(len(test), len(test)))
        # np.random.seed(None)
        # self.batch_size = batch_size
        # for vs in w2vs.values():
        #     idxs = [v2i[v] for v in vs]
        #     for i in idxs:
        #         for j in idxs:
        #             self.score[
        #                 i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        # self.get_test_data()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        d = np.zeros((size,) + img_shape, dtype=K.floatx())
        e = np.zeros((size, 5004), dtype=K.floatx())
        f = np.zeros((size,) + img_shape, dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_validation(self.match[j][0], p2size, p2bb)
            b[i, :, :, :] = read_for_validation(self.match[j][1], p2size, p2bb)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_validation(self.unmatch[j][0], p2size, p2bb)
            b[i + 1, :, :, :] = read_for_validation(self.unmatch[j][1], p2size, p2bb)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        for i in range(size):
            d[i, :, :, :] = read_for_validation(test[(start + i) % len(test)], p2size, p2bb)
            e[i, w2idx[p2ws[test[(start + i) % len(test)]][0]]] = 1
        return [a, b, d, f], [c, e, f]

    # def get_test_data(self):
    #     self.match = []
    #     self.unmatch = []
    #     _, _, x = lapjv(self.score)  # Solve the linear assignment problem
    #     y = np.arange(len(x), dtype=np.int32)
    #
    #     # Compute a derangement for matching whales
    #     for vs in w2vs.values():
    #         d = vs.copy()
    #         while True:
    #             random.shuffle(d)
    #             if not np.any(vs == d): break
    #         for ab in zip(vs, d): self.match.append(ab)
    #
    #     # Construct unmatched whale pairs from the LAP solution.
    #     for i, j in zip(x, y):
    #         if i == j:
    #             print(self.score)
    #             print(x)
    #             print(y)
    #             print(i, j)
    #         assert i != j
    #         self.unmatch.append((test[i], test[j]))
    #
    #     # print(len(self.match), len(train), len(self.unmatch), len(train))
    #     assert len(self.match) == len(test) and len(self.unmatch) == len(test)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size

class TrainingData(Sequence):
    def __init__(self, score, train_soft, join, steps=1000, batch_size=64):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.train_soft = train_soft
        self.join = join
        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        d = np.zeros((size,) + img_shape, dtype=K.floatx())  # softmax loss x
        e = np.zeros((size, 5004), dtype=K.floatx())         # softmax loss y
        f = np.zeros((size,) + img_shape, dtype=K.floatx())  # decoder x, y
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0], p2size, p2bb)
            b[i, :, :, :] = read_for_training(self.match[j][1], p2size, p2bb)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0], p2size, p2bb)
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1], p2size, p2bb)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        for i in range(size):
            d[i, :, :, :] = read_for_training(self.train_soft[(start + i) % len(self.train_soft)], p2size, p2bb)
            e[i, w2idx[p2ws[self.train_soft[(start + i) % len(self.train_soft)]][0]]] = 1
        for i in range(size):
            f[i, :, :, :] = read_for_training(self.join[(start + i) % len(self.join)], p2size, p2bb)
        return [a, b, d, f], [c, e, f]
    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        np.random.seed(None)
        np.random.shuffle(self.train_soft)
        np.random.shuffle(self.join)
        self.steps -= 1
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size
# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i], p2size, p2bb)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size
def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score = score_reshape(score, features)
    return features, score


class cv_callback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 4:
            return

        cv_test = test + new_whales[:len(test)]
        # Evaluate the model.
        fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        fsubmit = branch_model.predict_generator(FeatureGen(cv_test), max_queue_size=20, workers=10, verbose=0)
        score_val = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        score_val = score_reshape(score_val, fknown, fsubmit)
        predictions = val_score(cv_test, args.threshold, known, p2wts, score_val)
        labels = [tagged[p] for p in test]
        labels_newwhale = ['new_whale' for p in new_whales[:len(test)]]
        labels = labels + labels_newwhale

        print('cv score: ' + str(map_per_set(labels, predictions)))
def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories
    np.random.seed(None)
    np.random.shuffle(train)
    # Compute the match score for each picture pair
    features, score = compute_score()

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), train_soft, join, steps=step, batch_size=32),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6,
        verbose=1, validation_data=TestingData(), callbacks=[cv_callback()]).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)
histories = []
steps = 0


# epoch -> 10
make_steps(10, 1000)
ampl = 100.0
for _ in range(2):
    print('noise ampl.  = ', ampl)
    make_steps(5, ampl)
    ampl = max(1.0, 100 ** -0.1 * ampl)
# epoch -> 150
for _ in range(18): make_steps(5, 1.0)
# epoch -> 200
set_lr(model, 16e-5)
for _ in range(10): make_steps(5, 0.5)
# epoch -> 240
set_lr(model, 4e-5)
for _ in range(8): make_steps(5, 0.25)
# epoch -> 250
set_lr(model, 1e-5)
for _ in range(2): make_steps(5, 0.25)
# epoch -> 300
weights = model.get_weights()
model, branch_model, head_model, dec_model, soft_model = build_model(64e-5, 0.0002)
model.set_weights(weights)
for _ in range(10): make_steps(5, 1.0)
# epoch -> 350
set_lr(model, 16e-5)
for _ in range(10): make_steps(5, 0.5)
# epoch -> 390
set_lr(model, 4e-5)
for _ in range(8): make_steps(5, 0.25)
# epoch -> 400
set_lr(model, 1e-5)
for _ in range(2): make_steps(5, 0.25)
model.save('standard.model')
def prepare_submission(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                for w in p2ws[h]:
                    assert w != new_whale
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break;
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos
def prepare_submission_softmax(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = sm_submit[i, :]
            for j in list(reversed(np.argsort(a))):
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                s.add(idx2w[j])
                t.append(idx2w[j])
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos
tic = time.time()

if True:
    # Evaluate the model with siamese network.
    fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)
    prepare_submission(args.threshold, 'submission.csv')
else:
    # Evaluate the model with classification model.
    fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
    sm_submit = soft_model.predict(fsubmit, batch_size=128)
    prepare_submission_softmax(args.threshold, 'submission.csv')
toc = time.time()
print("Submission time: ", (toc - tic) / 60.)




