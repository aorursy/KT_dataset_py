import datetime
from datetime import timedelta
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D,
                                     Dense, Dropout, Input, Lambda, Multiply, Embedding, Flatten, 
                                     concatenate, TimeDistributed, Reshape)

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from pathlib import Path
import gc
%matplotlib inline


# Some paths


BASE_PATH = Path("../input/m5-forecasting-accuracy/")
DIMENSION_PATH = "../input/m5-sales-hierarchy-dataset/dimension.parquet"
SALES_TRAIN_VALIDATION_PATH = BASE_PATH / "sales_train_validation.csv"
SAMPLE_SUBMISSION_PATH = BASE_PATH / "sample_submission.csv"
SUBMISSION_PATH = "submission.csv"

# Some dates

TRAIN_START_DATE = pd.to_datetime("2011-01-29")
TRAIN_END_DATE = pd.to_datetime("2016-04-24")

# Try more history later? One year more, let's see... Trying less...
# Due to memory issues, the start date is only 2014...
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2016-04-24"
PRED_STEPS = 28

# Will drop these dates
CHRISTMAS_DATES = ["2012-12-25", "2013-12-25", "2014-12-25", "2015-12-25"]


# Change these to be in different "run" modes.
# TODO: Improve the conf values mngt...
DEBUG = False
LOAD = False
SUBMISSION = True 
# TODO: Is this the correct one? Seems to be...
FIRST_N_SAMPLES = 30940



# Model HP

BATCH_SIZE = 2**2
EPOCHS = 5

# Finish implementing this...
class DataProcessing:

    
    pass 



def get_time_block_series(series_array, date_to_index, start_date, end_date):

    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]


def transform_series_encode(series_array):
    # Should there be scale transformation?
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_mean = series_array.mean(axis=1).reshape(-1,1)
    series_std = series_array.std(axis=1).reshape(-1,1)
    # TODO: What about std? Should the mean be saved? Use these for predicting...
    # Transform these into a tf.Tensor?
    # Was 0.01. There seems to be an impact on the error function...
    series_array = (series_array - series_mean) # To avoid NaN? Changed to smaller.
    series_array = series_array.reshape((series_array.shape[0],series_array.shape[1], 1))
    return series_array, series_mean, series_std

def transform_series_decode(series_array, encode_series_mean, encode_series_std):

    # Should there be scale transformation?
    series_array = np.log1p(np.nan_to_num(series_array)) # filling NaN with 0
    series_array = (series_array - encode_series_mean) # To avoid NaN?
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array

def untransform_series_decode(series_array, encode_series_mean, encode_series_std):
    series_array = series_array.reshape(series_array.shape[0], series_array.shape[1])
    series_array = series_array + encode_series_mean
    # unlog the data, clip the negative part if smaller than 0
    series_array = np.expm1(series_array)
    series_array = np.clip(series_array, 0.0, None)
    return  series_array


def predict_sequences(input_sequences, batch_size):
    history_sequences = input_sequences.copy()
    # initialize output (pred_steps time steps)
    pred_sequences = np.zeros((history_sequences.shape[0], pred_steps, 1))   

    # TODO: pred_steps should be capitalized.
    for i in tqdm(range(pred_steps), "Predicting"):

        # record next time step prediction (last time step of model output)
        last_step_pred = model.predict(history_sequences, batch_size)[:, -1, 0]
        # For debug, remove later...
        pred_sequences[:, i, 0] = last_step_pred

        # add the next time step prediction along with corresponding exogenous features
        # to the history tensor
        # TODO: Why from one? So that it can be concatenated... Alright.
        last_step_exog = input_sequences[:, [(-pred_steps+1)+i], 1:]
        last_step_tensor = np.concatenate([last_step_pred.reshape((-1,1,1)), 
                                           last_step_exog], axis=-1)
        history_sequences = np.concatenate([history_sequences, last_step_tensor], axis=1)
        del last_step_pred
        del last_step_tensor
        del last_step_exog
        gc.collect()

    return pred_sequences



def get_data_encode_decode(series_array, exog_array, first_n_samples,
                           date_to_index, enc_start, enc_end, pred_start=None, pred_end=None, pred=False):

    exog_inds = date_to_index[enc_start:pred_end]

    # sample of series from enc_start to enc_end  
    encoder_input_data = get_time_block_series(series_array, date_to_index, 
                                               enc_start, enc_end)[:first_n_samples]
    encoder_input_data, encode_series_mean, encode_series_std = transform_series_encode(encoder_input_data)


    if not pred:
        # sample of series from pred_start to pred_end 
        decoder_target_data = get_time_block_series(series_array, date_to_index, 
                                                    pred_start, pred_end)[:first_n_samples]
        decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean, encode_series_std)

        # we append a lagged history of the target series to the input data, 
        # so that we can train with teacher forcing
        lagged_target_history = decoder_target_data[:,:-1,:1]
        encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

    # we add the exogenous features corresponding to day after input series
    # values to the input data (exog should match day we are predicting)
    if not pred:
        # Why does that start 1?
        exog_input_data = exog_array[:first_n_samples, exog_inds, :][:, 1:,:]
    else:
        exog_input_data = exog_array[:first_n_samples, exog_inds, :][:, :,:]
    # encoder_input_data = np.concatenate([encoder_input_data, exog_input_data], axis=-1)

    if not pred:
        return encoder_input_data, exog_array, decoder_target_data
    else:
        return encoder_input_data, encode_series_mean, encode_series_std

def predict_and_save(encoder_input_data, sample_ind, batch_size=2**4, enc_tail_len=30 * 6, decoder_target_data=1):

    label = ids[sample_ind]
    encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:]
    pred_series = predict_sequences(encode_series, batch_size)

    encode_series = encode_series.reshape(-1,1)
    pred_series = pred_series.reshape(-1,1)  

    if isinstance(decoder_target_data, np.ndarray):
        target_series = decoder_target_data[sample_ind, :, :1].reshape(-1, 1)
        encode_series_tail = np.concatenate([encode_series[-enc_tail_len:], target_series[:1]])
    else:
        encode_series_tail = encode_series[-enc_tail_len:]


    x_encode = encode_series_tail.shape[0]

    fig = plt.figure(figsize=(10,6))   

    plt.plot(range(1,x_encode+1),encode_series_tail)
    plt.plot(range(x_encode,x_encode+pred_steps),pred_series,color='teal',linestyle='--')

    plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
    plt.legend(['Encoding Series','Target Series',f'Predictions for {label}'])

    if isinstance(decoder_target_data, np.ndarray):
        plt.plot(range(x_encode, x_encode + pred_steps), target_series, color='orange')
        plt.legend(['Encoding Series', 'Target Series', 'Predictions'])
    else:
        plt.legend(['Encoding Series', 'Predictions'])

    fig.savefig(f"predictions_{sample_ind}.png")


    # To clear the figure
    plt.clf()

# TODO: Move these into a function/class and some refactoring...

df = pd.read_csv(SALES_TRAIN_VALIDATION_PATH)

ids = df["id"]




pred_length= timedelta(PRED_STEPS)

first_day = pd.to_datetime(DATA_START_DATE) 
last_day = pd.to_datetime(DATA_END_DATE)

val_pred_start = last_day - pred_length + timedelta(1)
val_pred_end = last_day

train_pred_start = val_pred_start - pred_length
train_pred_end = val_pred_start - timedelta(days=1)


enc_length = train_pred_start - first_day

train_enc_start = first_day
train_enc_end = train_enc_start + enc_length - timedelta(1)

val_enc_start = train_enc_start + pred_length
val_enc_end = val_enc_start + enc_length - timedelta(1)

print('Train encoding:', train_enc_start, '-', train_enc_end)
print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')
print('Val encoding:', val_enc_start, '-', val_enc_end)
print('Val prediction:', val_pred_start, '-', val_pred_end)

print('\nEncoding interval:', enc_length.days)
print('Prediction interval:', pred_length.days)

columns = df.columns
ids = df["id"]
date_columns = columns[columns.str.contains("d_")]
dates_s = pd.date_range(TRAIN_START_DATE, TRAIN_END_DATE, freq="1d")
date_to_index = pd.Series(index=dates_s, 
                          data=range(len(date_columns)))

christmas_index = [date_to_index[d] for d in CHRISTMAS_DATES]
date_columns = list(set(date_columns) - set(date_columns[christmas_index]))
series_array = df[date_columns].values
# More features 

# DOW encoded => try categorical embedding later?
dow_ohe = pd.get_dummies(dates_s.dayofweek)


# TODO: Other later???
dow_array = np.expand_dims(dow_ohe.values, axis=0) # add sample dimension
dow_array = np.tile(dow_array, (df.shape[0], 1,1)) # repeat OHE array along sample dimension


month_ohe = pd.get_dummies(dates_s.month)
month_ohe = np.expand_dims(month_ohe.values, axis=0) # add sample dimension
month_ohe = np.tile(month_ohe, (df.shape[0],1,1)) # repeat OHE array along sample dimension

# Sales hierarchy 
dimension_df = pd.read_parquet(DIMENSION_PATH, columns=["location", "department", "category"]) # TODO: Try more later...
dimension_array = pd.get_dummies(dimension_df).values
dimension_array = np.expand_dims(dimension_array, axis=1) # add timesteps dimension
dimension_array = np.tile(dimension_array, (1, dow_array.shape[1], 1)) # repeat OHE array along timesteps dimension 


year_array = pd.get_dummies(dates_s.year).values
year_array = np.expand_dims(year_array, axis=0)  # add timesteps dimension
year_array = np.tile(year_array, (df.shape[0],1,1))

# Add the other ones later, for now memory issue...
exog_array = np.concatenate([dow_array, dimension_array], axis=-1)


dow_array.shape[1]
# Cleanup

del df
del dimension_df
del year_array
del dimension_array
del month_ohe
del dow_array
gc.collect()
embedding_df = pd.read_parquet(DIMENSION_PATH, columns=["location", "department", "category"])
location_s = embedding_df[["location"]].values
location_s = np.expand_dims(location_s, axis=1)
location_s = np.tile(location_s, (1, 1913, 1))
for col in ["location", "department", "category"]:
    print(col, embedding_df[col].nunique())
def embedding_mapping(df):
    
    res = []
    for col in ["location", "department", "category"]:
        raw_vals = embedding_df[col].unique()
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i     
        mapping = df[col].map(val_map).values
        res.append(mapping)
    return res
res = embedding_mapping(embedding_df)
def build_model(pred_steps=PRED_STEPS):
    n_filters = 16 # 32 
    filter_width = 2 # Was 2
    # Maybe tree less? Was 8,  tried 4, trying 16... Overfitting for sure...
    # Due to limited memory, will be restrained to 4
    dilation_rates = [2**i for i in range(4)] * 2 
    
    
    # Numerical input
    history_seq = Input(shape=(None, 1))
                        
                        
    # Embeddings inputs
    # dow_input = Input(shape=(None, ), name='dow')
    location_input = Input(shape=(None, ), name='location')
    # department_input = Input(shape=(None, ), name='department')
    # category_input = Input(shape=(None, ), name='category')
    # Simple heuristic for embedding dimensions: number of unique categories divided by 2.

    # dow_emb = (Embedding(7, 7 // 2)(dow_input))
    location_emb = (Embedding(3, 3//2)(location_input))    
    # department_emb = (Embedding(7, 7 // 2)(department_input))                        
    # category_emb =(Embedding(3, 3 // 2)(category_input))                        
    

    # x = concatenate([history_seq, dow_emb, location_emb, department_emb, category_emb])
    x = concatenate([history_seq, location_emb])
    print(x.shape)

    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(32, 5, padding='same', activation='relu')(x) 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(32, 1, padding='same', activation='relu')(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(.2)(out)
    out = Conv1D(1, 1, padding='same')(out)

    def slice(x, seq_length):
        return x[:, -seq_length:, :]

    pred_seq_train = Lambda(slice, arguments={'seq_length': pred_steps})(out)

    model = Model([history_seq, location_input], pred_seq_train)
    model.compile(Adam(), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])   
    return model
    
model = build_model()
# TODO: Break this into a separate model and train loop

def train_model(model, batch_size=BATCH_SIZE, epochs=EPOCHS):



    encoder_input_data, _location_s, decoder_target_data = \
        get_data_encode_decode(series_array, location_s, FIRST_N_SAMPLES, date_to_index, 
                               train_enc_start, train_enc_end, train_pred_start, train_pred_end)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=3, min_lr=0.001)

    checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5")

    print(encoder_input_data.shape)
    %pdb
    history = model.fit([encoder_input_data, _location_s], decoder_target_data,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS),
                        
                        # validation_split=0.2,
                        # callbacks=[checkpoint, reduce_lr])
    # Plot the model's architecture
    tensorflow.keras.utils.plot_model(model, 'model_architecture.png', show_shapes=True) 
    return model, encoder_input_data, decoder_target_data

if LOAD:
    model = tf.keras.models.load_model("model.h5")
else:
    model = build_model()
    model, encoder_input_data, decoder_target_data = train_model(model)
if not submission:

    if debug:
        predict_and_save(encoder_input_data, 2486, decoder_target_data=decoder_target_data, batch_size=1)

    else:
        # TODO: Plot more?
        for i in range(0, first_n_samples, 1000):
            predict_and_save(encoder_input_data, i, decoder_target_data=decoder_target_data, batch_size=1)

if submission:
    # Two years was too much for the RAM...
    cmp_enc_start = TRAIN_END_DATE - timedelta(days=28 * 3)

    cmp_enc_end = TRAIN_END_DATE


    encoder_input_data, encode_series_mean, encode_series_std = get_data_encode_decode(
                            series_array, exog_array, first_n_samples, date_to_index, 
                            cmp_enc_start, cmp_enc_end, pred=True)

    # TODO: Change how many batches are predicted once this works...
    pred_series = predict_sequences(encoder_input_data, batch_size=2**4)

    # visualize one sample to check the prediction
    predict_and_save(encoder_input_data, 100)


    # reverse the transformation
    pred_series_transformed = untransform_series_decode(pred_series, encode_series_mean, encode_series_std)

    # check the time frame
    print('encode_input_first_day:', cmp_enc_start.date())
    print('encode_input_last_day:', cmp_enc_end.date())

    columns = [f"F{id}" for id in range(1, 29)]
    sumbmission_df = pd.DataFrame(pred_series_transformed, columns=columns)
    sumbmission_df["id"] = ids
    del pred_series_transformed
    gc.collect()

    # Read samples_df and make the submission_df final DataFrame.

    samples_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    sumbmission_df = pd.concat([samples_df.loc[lambda df: df["id"].str.contains("eval")], sumbmission_df], sort=False)
    del samples_df
    gc.collect()
    # Removed compression so that I can make a submission from the kernel
    sumbmission_df.to_csv(SUBMISSION_PATH, index=False)
sumbmission_df.head()