import sys

sys.path.append('../input/draw-rna')
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.lines as mlines



from ipynb.draw import draw_struct

import seaborn as sns



plt.style.use('ggplot')
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
# we define the columns separately (this is gonna be usefull later)

err_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']

mes_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train.info()
test.info()
#taking a sample from our training data

    #samples with good confidence 

train_hsnr = train.query('signal_to_noise > 5')

sample = train_hsnr.iloc[49, :]

seq = sample['sequence']

struc = sample['structure']



#ploting the structure with different measurements as alpha



fig, axs = plt.subplots(3, 2, figsize=(20,20))

axs = axs.flatten()

    #for figure Legend

_measured = mlines.Line2D([], [], color='blue', linestyle='None', marker='o', markersize=15, label='Measured')

_unmeasured = mlines.Line2D([], [], color='red', linestyle='None', marker='o', markersize=15, label='Unmeasured')



for i, mes_col in enumerate(mes_cols):

    measure = np.array(sample[mes_col])

    #the last 39 bases aren't measured

    unmeasured = len(seq) - len(measure)

    #normalized the measurement vector (alpha require [0-1] range values)

    norm = (measure - measure.min()) / ( measure.max() - measure.min())

    #padding with ones to have same length

    alpha = np.concatenate((norm, np.ones(unmeasured)))

    #this is to distiguich measured/unmeasured bases

    color = np.concatenate((np.zeros(len(measure)), np.ones(unmeasured)))

    draw_struct(seq, struc, c=color, cmap='bwr', alpha=alpha, ax=axs[i])

    axs[i].set_title(mes_col)

    axs[i].legend(handles=[_measured, _unmeasured])

    

axs[-1].axis('off')
ax = train['signal_to_noise'].plot(kind='hist', bins=40, figsize=(15,3))

train['signal_to_noise'].plot(kind='kde', ax=ax, secondary_y=True)
#pd.DataFrame(train.query('signal_to_noise == 0').reactivity.values.tolist()).head(2)
#pd.DataFrame(train.query('signal_to_noise == 0').reactivity_error.values.tolist()).head(2)
train['signal_to_noise'].describe()
def calc_snr(sample):

    """This function takes a row and return signal to noise

        ratio accross all measurments"""

    ratios = np.zeros(5)

    for i , (deg, err) in enumerate(zip(mes_cols, err_cols)):

        ratio = (np.array(sample[deg]) / np.array(sample[err])).mean()

        ratios[i] = ratio

    return ratios.mean()
train['snr'] = train.apply(calc_snr, axis=1)
train[['signal_to_noise', 'snr']].sample(10)
train.eval('signal_to_noise - snr').mean()
train.SN_filter.hist(bins=3)
train['avg_err'] = train[err_cols].applymap(np.mean).mean(axis=1)
train['avg_err'].plot.hist(bins=10)

train['avg_err'].describe()
train['avg_err'].quantile(0.94)
#mask = train.query('signal_to_noise > 1')[mes_cols].apply(lambda row: np.any([np.any(np.array(c) == 0) for c in row]), axis=1)

#filtred = train.query('signal_to_noise > 1')[mask]
fig, axs = plt.subplots(5,1, figsize=(10, 15))

axs.flatten()

sample = train[train.SN_filter == 1].sample(1).iloc[0]



for i, err_col in enumerate(err_cols):

    axs[i].plot(sample[err_col],color='red', drawstyle='steps-mid')

    axs[i].set_title(err_col)
fig, axs = plt.subplots(5,1, figsize=(10, 15))

axs.flatten()



for i, err_col in enumerate(err_cols):

    errs = np.array(train[train.signal_to_noise > 1][err_col].values.tolist())

    for err in errs:

        axs[i].plot(err,color='black', alpha=0.01, zorder=-32)

    errs_avg = errs.mean(axis=0)

    errs_std = errs.std(axis=0)

    axs[i].errorbar(np.arange(68), errs_avg, yerr=errs_std, color='red', ecolor='yellow',  drawstyle='steps-mid')

    axs[i].set_ylim(0, 0.7)

    axs[i].set_title(err_col)
#Calculating the variances for each measurements

all_var = np.array(train.query('SN_filter == 1')[err_cols].applymap(lambda c: np.array(c) ** 2).values.tolist()) # shape: (1589, 5, 68)



#averaging along the sequence and samples axis

mse = all_var.mean(2).mean(0)



#square root and column-wise mean

mcrmse = np.sqrt(mse).mean()

mcrmse
fig, axs = plt.subplots(5,1, figsize=(10, 15))

axs.flatten()



for i, mes_col in enumerate(mes_cols):

    mess = np.array(train[train.SN_filter == 1][mes_col].values.tolist())

    for mes in mess:

        axs[i].plot(mes,color='black', alpha=0.01, zorder=-32)

    mess_avg = mess.mean(axis=0)

    mess_std = mess.std(axis=0)

    axs[i].errorbar(np.arange(68), mess_avg, yerr=mess_std, color='lime', ecolor='yellow', drawstyle='steps-mid')

    axs[i].set_ylim(0, 4)

    axs[i].set_title(err_col)
fig, axs = plt.subplots(5,1, figsize=(10, 15))

axs.flatten()

sample = train[train.SN_filter == 1].sample(1).iloc[0]



for i, (mes_col, err_col) in enumerate(zip(mes_cols, err_cols)):

    err = np.array(sample[err_col])

    mes = np.array(sample[mes_col])

    axs[i].errorbar(np.arange(68), mes, yerr=err, color='blue', ecolor='red', drawstyle='steps-mid', barsabove=True)

    axs[i].set_title(mes_col)
sample_id = 'id_20dec87f6'

sample_dist = np.load('../input/stanford-covid-vaccine/bpps/' + sample_id + '.npy')
plt.figure(figsize=(8,8))

plt.imshow(sample_dist)

plt.colorbar()
def is_loop(sub_struc):

    balance = 0

    for c in sub_struc:

        if balance < 0:

            return True

        if c == b'(':

            balance += 1

        if c == b')':

            balance -= 1

    return balance != 0

    

def sample_struc(_id, min_prob = 0.4):

    dist = np.load('../input/stanford-covid-vaccine/bpps/' + _id + '.npy')

    struc = np.chararray(dist.shape[0])

    struc[:] = '.'

    dist = np.tril(dist)

    while(True):

        if dist.max() < min_prob:

            break

        args = np.argwhere(dist == dist.max())

        for [i, j] in args:

            if not is_loop(struc[j:i]) and struc[i] == b'.':

                #print([j, i, dist.max()])

                struc[i] = ')'

                struc[j] = '('

            dist[i, j] = 0

    return struc.tostring().decode("utf-8") 

        
sample_struct = sample_struc(sample_id)

sample_struct
train.query('id == @sample_id').iloc[0]['structure']
ds = pd.concat([train, test])

ds['sample_struc'] = ds['id'].apply(sample_struc)

shape = ds.query('structure == sample_struc').shape
ds.query('structure == sample_struc').shape[0] / ds.shape[0]
true = list()

for i, c in enumerate(true_struc):

    if c == '(':

        balance = 0

        for j , _c in enumerate(true_struc[i + 1:]):

            if _c == ')' and balance == 0:

                true.append(((i, i + j + 1), sample_dist[i][i + j + 1]))

                break;

            if _c == ')':

                balance -= 1

            if _c == '(':

                balance += 1

true.sort(key=lambda x: x[1], reverse=True)
smpld = list()

dist = np.tril(sample_dist)

min_prob = 0.1

while(True):

    if dist.max() < min_prob:

        break

    args = np.argwhere(dist == dist.max())

    for [i, j] in args:

        smpld.append(((j,i), dist.max()))

        dist[i, j] = 0