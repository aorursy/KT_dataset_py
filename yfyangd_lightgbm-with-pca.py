import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import os
print(os.listdir("../input"))
#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    

df_train = import_data("../input/train.csv")
df_train.shape
df_train.head()
plt.style.use('ggplot')
plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
sns.countplot(df_train['event'])
plt.xlabel("State of the pilot", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Target repartition", fontsize=15)

plt.subplot(1,3,2)
sns.countplot('experiment', hue='event', data=df_train)
plt.xlabel("Experiment and state of the pilot", fontsize=12)
plt.ylabel("Count (log)", fontsize=12)
plt.yscale('log')
plt.title("Experiments", fontsize=15)

plt.subplot(1,3,3)
sns.countplot('event', hue='seat', data=df_train)
plt.xlabel("Seat and state of the pilot", fontsize=12)
plt.ylabel("Count (log)", fontsize=12)
plt.yscale('log')
plt.title("Seat", fontsize=15)
plt.show()
# Just looking at a single trial for now
subset = df_train.loc[(df_train['crew'] == 1) & (df_train['experiment'] == 'CA')]
subset.sort_values(by='time')

# remove the high frequency signals by scipy
from scipy import signal
b, a = signal.butter(8,0.05)
y = signal.filtfilt(b, a, subset['r'], padlen=150)

plt.style.use('seaborn')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("Respiratoin Trend", fontsize=15)
plt.plot(subset['r'][3000:4024])
plt.subplot(1,2,2)
plt.plot(y[3000:4024])
plt.title("Respiratoin Trend after remove the high frequency signals", fontsize=15)
from biosppy.signals import ecg, resp

out = resp.resp(y,sampling_rate=256, show=False)

plt.plot(out['resp_rate_ts'], out['resp_rate'])
plt.ylabel('Respiratory frequency [Hz]')
plt.xlabel('Time [s]')
plt.title("Respiratoin Rate", fontsize=15)
plt.style.use('bmh')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.violinplot(x='event', y='ecg', data=df_train.sample(50000))
plt.ylabel("Electrocardiogram Signal (µV)", fontsize=12)
plt.xlabel("Event", fontsize=12)
plt.title("Electrocardiogram signal influence", fontsize=15)

plt.subplot(1,2,2)
sns.distplot(df_train['ecg'], label='Train set')
plt.legend()
plt.xlabel("Electrocardiogram Signal (µV)", fontsize=12)
plt.title("Electrocardiogram Signal Distribution", fontsize=15)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
b, a = signal.butter(8,0.05)
y = signal.filtfilt(b, a, subset['ecg'], padlen=150)
plt.plot(y[3000:4024])
plt.title("Electrocardiogram Trend", fontsize=15)

#Convert ECG into heart rate data by Biosppy to detect the R waves
plt.subplot(1,2,2)
out = ecg.ecg(signal=subset['ecg'], sampling_rate=256, show=False)
plt.plot(out['heart_rate_ts'], out['heart_rate'])
plt.ylabel('Heart Rate (BPM)')
plt.xlabel('Time [s]');
plt.title("Electrocardiogram Heart Rate", fontsize=15)
eeg_features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]

plt.figure(figsize=(20,25))

i = 0
for egg in eeg_features:
    i += 1
    plt.subplot(5, 4, i)
    sns.boxplot(x='event', y=egg, data=df_train.sample(50000), showfliers=False)
plt.figure(figsize=(20,25))
plt.title('EEG features distributions')
i = 0

for eeg in eeg_features:
    i += 1
    plt.subplot(5, 4, i)
    sns.distplot(df_train.sample(10000)[eeg], label='Train set', hist=False)
    plt.xlim((-500, 500))
    plt.legend()
    plt.xlabel(eeg, fontsize=12)
df_train['fp1_f7'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f7_t3'] = df_train['eeg_f7'] - df_train['eeg_t3']
df_train['t3_t5'] = df_train['eeg_t3'] - df_train['eeg_t5']
df_train['t5_o1'] = df_train['eeg_t5'] - df_train['eeg_o1']
df_train['fp1_f3'] = df_train['eeg_fp1'] - df_train['eeg_f7']
df_train['f3_c3'] = df_train['eeg_f3'] - df_train['eeg_c3']
df_train['c3_p3'] = df_train['eeg_c3'] - df_train['eeg_p3']
df_train['p3_o1'] = df_train['eeg_p3'] - df_train['eeg_o1']

df_train['fz_cz'] = df_train['eeg_fz'] - df_train['eeg_cz']
df_train['cz_pz'] = df_train['eeg_cz'] - df_train['eeg_pz']
df_train['pz_poz'] = df_train['eeg_pz'] - df_train['eeg_poz']

df_train['fp2_f8'] = df_train['eeg_fp2'] - df_train['eeg_f8']
df_train['f8_t4'] = df_train['eeg_f8'] - df_train['eeg_t4']
df_train['t4_t6'] = df_train['eeg_t4'] - df_train['eeg_t6']
df_train['t6_o2'] = df_train['eeg_t6'] - df_train['eeg_o2']
df_train['fp2_f4'] = df_train['eeg_fp2'] - df_train['eeg_f4']
df_train['f4_c4'] = df_train['eeg_f4'] - df_train['eeg_c4']
df_train['c4_p4'] = df_train['eeg_c4'] - df_train['eeg_p4']
df_train['p4_o2'] = df_train['eeg_p4'] - df_train['eeg_o2']
df_train.head()
df_train.shape
df_train.info()
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
LE2 = LabelEncoder()

df_train['experiment'] = LE1.fit_transform(df_train['experiment'])
df_train['event'] = LE2.fit_transform(df_train['event'])
df_train.head()
df_train['pilot'] = 100 * df_train['seat'] + df_train['crew']
print("Number of pilots : ", len(df_train['pilot'].unique()))
df_train.head()
from sklearn.preprocessing import MinMaxScaler
def normalize_by_pilots(df):
    pilots = df["pilot"].unique()
    for pilot in tqdm(pilots):
        ids = df[df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        df.loc[ids, features_m] = scaler.fit_transform(df.loc[ids, features_m])
    return df
features_m = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr",'fp1_f7', 'f7_t3', 't3_t5', 't5_o1', 'fp1_f3', 'f3_c3', 'c3_p3', 'p3_o1', 'fz_cz', 'cz_pz',
                'pz_poz', 'fp2_f8', 'f8_t4', 't4_t6', 't6_o2', 'fp2_f4', 'f4_c4', 'c4_p4', 'p4_o2']
df_train = normalize_by_pilots(df_train)
df_train.head()
y = df_train['event']
df_train.drop(['event'], axis=1, inplace=True)
df_train.shape
df_train.head()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
lista = range(1,47)
aa=[]
bb=[]
for f in tqdm(lista):
    aa.append(f)
    pca = PCA(n_components=f).fit(df_train)
    
a = 0
for e in lista:
    a=a+pca.explained_variance_ratio_[e-1]
    bb.append(a)
fig, ax = plt.subplots(figsize = (10,6))
plot = plt.plot(aa, bb, '-o')
ax.set_xlabel("Dimensions")
ax.set_ylabel("Variance_ratio"); 
# Apply PCA for dimension reduction
pca = PCA(n_components=10).fit(df_train)
X_pca = pca.transform(df_train)
print(sum(pca.explained_variance_ratio_)) 
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size = 0.2)
from imblearn.over_sampling import SMOTE
X_train, y_train = SMOTE().fit_resample(X_train, y_train.ravel())
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
         }

XGB = XGBClassifier()

# XGB = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
XGB.fit(X_train,y_train)
print ("accuracy_score on testing data of XGBoost: {:.4f}".format(accuracy_score(y_val, XGB.predict(X_val))))
