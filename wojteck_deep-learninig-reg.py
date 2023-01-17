import seaborn as sns

import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn.svm import SVR, NuSVR

from sklearn.kernel_ridge import KernelRidge

import pandas as pd

import numpy as np

import os

import gc

import warnings

warnings.filterwarnings("ignore")



DATA_DIR = "../input"

TEST_DIR = r'../input/test'



ld = os.listdir(TEST_DIR)

sizes = np.zeros(len(ld))



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy.stats import pearsonr

from scipy import stats

from sklearn.kernel_ridge import KernelRidge



import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from tsfresh.feature_extraction import feature_calculators

from tqdm import tqdm



%matplotlib inline

sns.set_style('darkgrid')
def classic_sta_lta(x, length_sta, length_lta):

    

    sta = np.cumsum(x ** 2)



    # Zamiana na float

    sta = np.require(sta, dtype=np.float)



    # Kopia dla LTA

    lta = sta.copy()



    # Obliczanie STA i LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta



    # Uzupełnienie zerami

    sta[:length_lta - 1] = 0



    # Aby nie dzielić przez 0 ustawiamy 0 na małe liczby typu float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny



    return sta / lta
def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)
percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]

hann_windows = [50, 150, 1500, 15000]

spans = [300, 3000, 30000, 50000]

windows = [10, 50, 100, 500, 1000, 10000]

borders = list(range(-4000, 4001, 1000))

peaks = [10, 20, 50, 100]

coefs = [1, 5, 10, 50, 100]

lags = [10, 100, 1000, 10000]

autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]
def gen_features(x, zero_mean=False):

    if zero_mean==True:

        x = x-x.mean()

    strain = {}

    strain['mean'] = x.mean()

    strain['std']=x.std()

    strain['max']=x.max()

    strain['kurtosis']=x.kurtosis()

    strain['skew']=x.skew()

    zc = np.fft.fft(x)

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    strain['min']=x.min()

    strain['sum']=x.sum()

    strain['mad']=x.mad()

    strain['median']=x.median()

    

    strain['mean_change_abs'] = np.mean(np.diff(x))

    strain['mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    strain['abs_max'] = np.abs(x).max()

    strain['abs_min'] = np.abs(x).min()

    

    strain['avg_first_50000'] = x[:50000].mean()

    strain['avg_last_50000'] = x[-50000:].mean()

    strain['avg_first_10000'] = x[:10000].mean()

    strain['avg_last_10000'] = x[-10000:].mean()

    

    strain['min_first_50000'] = x[:50000].min()

    strain['min_last_50000'] = x[-50000:].min()

    strain['min_first_10000'] = x[:10000].min()

    strain['min_last_10000'] = x[-10000:].min()

    

    strain['max_first_50000'] = x[:50000].max()

    strain['max_last_50000'] = x[-50000:].max()

    strain['max_first_10000'] = x[:10000].max()

    strain['max_last_10000'] = x[-10000:].max()

    

    strain['max_to_min'] = x.max() / np.abs(x.min())

    strain['max_to_min_diff'] = x.max() - np.abs(x.min())

    strain['count_big'] = len(x[np.abs(x) > 500])

           

    strain['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

    strain['mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])

    strain['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])

    strain['mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    

    strain['q95'] = np.quantile(x, 0.95)

    strain['q99'] = np.quantile(x, 0.99)

    strain['q05'] = np.quantile(x, 0.05)

    strain['q01'] = np.quantile(x, 0.01)

    

    strain['abs_q95'] = np.quantile(np.abs(x), 0.95)

    strain['abs_q99'] = np.quantile(np.abs(x), 0.99)

    strain['abs_q05'] = np.quantile(np.abs(x), 0.05)

    strain['abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    for autocorr_lag in autocorr_lags:

        strain['autocorrelation_' + str(autocorr_lag)] = feature_calculators.autocorrelation(x, autocorr_lag)

    

    # percentiles on original and absolute values

    for p in percentiles:

        strain['percentile_'+str(p)] = np.percentile(x, p)

        strain['abs_percentile_'+str(p)] = np.percentile(np.abs(x), p)

    

    strain['abs_mean'] = np.abs(x).mean()

    strain['abs_std'] = np.abs(x).std()

    

    strain['quantile_0.95']=np.quantile(x, 0.95)

    strain['quantile_0.99']=np.quantile(x, 0.99)

    strain['quantile_0.05']=np.quantile(x, 0.05)

    strain['realFFT_mean']=realFFT.mean()

    strain['realFFT_std']=realFFT.std()

    strain['realFFT_max']=realFFT.max()

    strain['realFFT_min']=realFFT.min()

    strain['imagFFT_mean']=imagFFT.mean()

    strain['imagFFT_std']=realFFT.std()

    strain['imagFFT_max']=realFFT.max()

    strain['imaglFFT_min']=realFFT.min()

    

    strain['std_first_50000']=x[:50000].std()

    strain['std_last_50000']=x[-50000:].std()

    strain['std_first_25000']=x[:25000].std()

    strain['std_last_25000']=x[-25000:].std()

    strain['std_first_10000']=x[:10000].std()

    strain['std_last_10000']=x[-10000:].std()

    strain['std_first_5000']=x[:5000].std()

    strain['std_last_5000']=x[-5000:].std()

        

    strain['Hilbert_mean'] = np.abs(hilbert(x)).mean()

    strain['Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    strain['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    strain['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    strain['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    strain['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    strain['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    strain['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    strain['Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    moving_average_700_mean = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    strain['exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    strain['exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    strain['exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)

    no_of_std = 3

    strain['MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    strain['MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    

    strain['iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    strain['q999'] = np.quantile(x,0.999)

    strain['q001'] = np.quantile(x,0.001)

    strain['ave10'] = stats.trim_mean(x, 0.1)

        

    for window in windows:

        x_roll_std = x.rolling(window).std().dropna().values

        x_roll_mean = x.rolling(window).mean().dropna().values

        

        strain['ave_roll_std_' + str(window)] = x_roll_std.mean()

        strain['std_roll_std_' + str(window)] = x_roll_std.std()

        strain['max_roll_std_' + str(window)] = x_roll_std.max()

        strain['min_roll_std_' + str(window)] = x_roll_std.min()

        strain['q01_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.01)

        strain['q05_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.05)

        strain['q95_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.95)

        strain['q99_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.99)

        strain['av_change_abs_roll_std_' + str(window)] = np.mean(np.diff(x_roll_std))

        strain['av_change_rate_roll_std_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        strain['abs_max_roll_std_' + str(window)] = np.abs(x_roll_std).max()

        

        for p in percentiles:

            strain['percentile_roll_std_' + str(p) + '_window_' + str(window)] = np.percentile(x_roll_std, p)

            strain['percentile_roll_mean_' + str(p) + '_window_' + str(window)] = np.percentile(x_roll_mean, p)

        

        strain['ave_roll_mean_' + str(window)] = x_roll_mean.mean()

        strain['std_roll_mean_' + str(window)] = x_roll_mean.std()

        strain['max_roll_mean_' + str(window)] = x_roll_mean.max()

        strain['min_roll_mean_' + str(window)] = x_roll_mean.min()

        strain['q01_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.01)

        strain['q05_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.05)

        strain['q95_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.95)

        strain['q99_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.99)

        strain['av_change_abs_roll_mean_' + str(window)] = np.mean(np.diff(x_roll_mean))

        strain['av_change_rate_roll_mean_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        strain['abs_max_roll_mean_' + str(window)] = np.abs(x_roll_mean).max()

        

        

    return pd.Series(strain)
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

X_train = pd.DataFrame()

y_train = pd.Series()



for df in tqdm(train_df):

    features = gen_features(df['acoustic_data'])

    X_train = X_train.append(features, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]), ignore_index=True)



X_train.head()
del train_df

X_test = pd.DataFrame()



for i, f in tqdm(enumerate(ld)):

    df = pd.read_csv(os.path.join(TEST_DIR, f))

    features = gen_features(df['acoustic_data'])

    X_test = X_test.append(features, ignore_index=True)
corelations = np.abs(X_train.corrwith(y_train)).sort_values(ascending=False)

corelations_df = pd.DataFrame(data=corelations, columns=['corr'])

print("Number of high corelated values: ",corelations_df[corelations_df['corr']>=0.55]['corr'].count())



high_corr = corelations_df[corelations_df['corr']>=0.55]

print(high_corr)

high_corr_labels = high_corr.reset_index()['index'].values

#print(high_corr_labels)
X_train_high_corr = X_train[high_corr_labels]

X_test_high_corr = X_test[high_corr_labels]
from sklearn.preprocessing import MinMaxScaler



scaler = StandardScaler()

scaler.fit(X_train_high_corr)

X_train_scaled = pd.DataFrame(scaler.transform(X_train_high_corr), columns=X_train_high_corr.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test_high_corr), columns=X_test_high_corr.columns)

p_columns = []

p_corr = []

p_values = []



for col in X_train_scaled.columns:

    p_columns.append(col)

    p_corr.append(abs(pearsonr(X_train_scaled[col], y_train.values)[0]))

    p_values.append(abs(pearsonr(X_train_scaled[col], y_train.values)[1]))



df = pd.DataFrame(data={'column': p_columns, 'corr': p_corr, 'p_value': p_values}, index=range(len(p_columns)))

df.sort_values(by=['corr', 'p_value'], inplace=True)

df.dropna(inplace=True)

df = df.loc[df['p_value'] <= 0.05]



drop_cols = []



for col in X_train_scaled.columns:

    if col not in df['column'].tolist():

        drop_cols.append(col)



print(drop_cols)

print('--------------------')

print(X_train_high_corr.columns.values)

        

X_train_scaled = X_train_scaled.drop(labels=drop_cols, axis=1)

X_test_scaled = X_test_scaled.drop(labels=drop_cols, axis=1)



X_train_scaled_minmax = X_train_scaled_minmax.drop(labels=drop_cols, axis=1)

X_test_scaled_minmax = X_test_scaled_minmax.drop(labels=drop_cols, axis=1)
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Dropout

from sklearn.model_selection import train_test_split

NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(128, kernel_initializer='RandomUniform',input_dim = X_train_scaled.shape[1], activation='relu'))

NN_model.add(Dropout(0.5))

# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='RandomUniform',activation='relu'))

NN_model.add(Dropout(0.5))

NN_model.add(Dense(256, kernel_initializer='RandomUniform',activation='relu'))

NN_model.add(Dropout(0.5))

NN_model.add(Dense(128, kernel_initializer='RandomUniform',activation='relu'))

# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='RandomUniform',activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.summary()
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]

NN_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

predictions_DNN = NN_model.predict(X_test_scaled)

submission_DNN = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), dtype={

    'acoustic_data': np.int16, 'time_to_failure': np.float32})

submission_DNN['time_to_failure'] = predictions_DNN

submission_DNN.to_csv('result_DNN.csv', index=False)