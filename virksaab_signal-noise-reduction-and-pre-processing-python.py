# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy import signal as sps
from sklearn.model_selection import train_test_split
from statistics import mode

plt.style.use('ggplot')
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# PARAMETERS
BATCH_SIZE = 8
EPOCHS = 10
PARENT_DATA_DIR_PATH = '../input'
METADATA_TRAIN_FILE_PATH = os.path.join(PARENT_DATA_DIR_PATH, "metadata_train.csv")
TRAIN_DATA_FILE_PATH = os.path.join(PARENT_DATA_DIR_PATH, "train.parquet")
metadata_train = pd.read_csv(METADATA_TRAIN_FILE_PATH)
print("#samples:", len(metadata_train))
metadata_train.head()
# For equal number of samples for each class
target0DF = metadata_train[metadata_train.target == 0]
target1DF = metadata_train[metadata_train.target == 1]
metadata_train = pd.concat([target0DF.iloc[:len(target1DF), :], target1DF])
metadata_train.target.value_counts()
gridDF = metadata_train.groupby('id_measurement')
metadataList = []
for name, group in gridDF:
    if len(group.signal_id) == 3:
    #     print(name, group)
        ids = list(map(lambda x: str(x), group.signal_id))
        target = mode(group.target)
    #     print("IDs:", ids)
    #     print("Target:", target)
        metadataList.append({'IDs':ids, 'target': target})
metadataList[:5]
trainMeta, valMeta = train_test_split(metadataList, test_size=.10)
print("trainMDDF shape:", len(trainMeta))
print("valMDDF shapeL", len(valMeta))
# VERIFY THAT TRAIN AND VALIDATION SET HAVE BOTH CLASSES DATA.
t0 = 0
t1 = 0
for pair in trainMeta:
    if pair['target'] == 0:
        t0 += 1
    else:
        t1 += 1
print("train set: #t0={}, #t1={}".format(t0, t1))
t0 = 0
t1 = 0
for pair in valMeta:
    if pair['target'] == 0:
        t0 += 1
    else:
        t1 += 1
print("val set: #t0={}, #t1={}".format(t0, t1))
data = pq.read_pandas(TRAIN_DATA_FILE_PATH).to_pandas()
data.head()
def get_data_batch(dataDF, metaList, batchsize=2, loop=False, denoise=True, combine_phases=True):
    """
    Args:
    dataDF (pandas.DataFrame): The train.parquet file dataframe
    metaList(pandas.DataFrame): The training metadata file
    batchsize(int): Number of samples in a batch
    loop(boolean): True, For training with keras fit_generator; False, for validation.
    denoise(boolean): True, reduce signal noise
    combine_phase(boolean): True, return shape:(batchsize, 800000, 1); False, return shape:(batchsize, 800000, 3)
    """
    # set filter
    b, a = sps.butter(3, 0.5, btype='highpass', analog=False)
    while True:
        counter = 0
        signals_list = []
        target_list = []
        try:
            for index in range(len(metaList)):
                counter += 1
                sample_ids = metaList[index]['IDs']
                sample_target = metaList[index]['target']
                # OneHot encoding
                sample_targetOH = np.zeros(2)
                sample_targetOH[sample_target] = 1
                sample_signal = dataDF[sample_ids]
                if denoise:
                    # DeNoise
                    for colname in sample_signal:
                        noisy_signal = sample_signal[colname]
                        sample_signal[colname] = sps.filtfilt(b, a, noisy_signal)
                if combine_phases:
                    # combine phases to one signal
                    sample_signal = sample_signal.mean(1)
                    signals_list.append(np.expand_dims(sample_signal.values.reshape(-1, 1), 0))
                    target_list.append(np.expand_dims(sample_targetOH, 0))
                    if counter == batchsize:
                        yield np.concatenate(signals_list), np.concatenate(target_list)
                        counter = 0
                        signals_list.clear()
                        target_list.clear()
                else:
                    signals_list.append(np.expand_dims(sample_signal.values, 0))
                    target_list.append(np.expand_dims(sample_targetOH, 0))
                    if counter == batchsize:
                        yield np.concatenate(signals_list), np.concatenate(target_list)
                        counter = 0
                        signals_list.clear()
                        target_list.clear()
        except:
            pass
        if not loop:
            break
sb, tb = next(get_data_batch(data, trainMeta, batchsize=BATCH_SIZE, loop=False))
for sample_signal, sample_target in zip(sb, tb):
    plt.figure(figsize=(15,5))
    plt.plot(sample_signal);
    plt.title("Label: "+str(np.argmax(sample_target)))
    plt.show()
sb, tb = next(get_data_batch(data, trainMeta, batchsize=BATCH_SIZE, loop=False))
print("Sample signal batch:", sb.shape)
print("Sample target batch:", tb.shape) # Target are OneHot encodeed
