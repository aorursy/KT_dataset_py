#the basics

import pandas as pd, numpy as np

import math, re, gc, random, os, sys

from matplotlib import pyplot as plt



#for maximum aesthetics

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



#tensorflow deep learning basics

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K



#for model evaluation

from sklearn.model_selection import train_test_split



#no warnings

import warnings

warnings.filterwarnings('ignore')
SEED = 34



def seed_everything(seed):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    

seed_everything(SEED)
#load files into memory as Pandas DataFrames

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')
#sneak peak at training features

print(train_features.shape)

if ~ train_features.isnull().values.any(): print('No missing values')

train_features.head()
#sneak peak at train targets

print(train_targets_scored.shape)

if ~ train_targets_scored.isnull().values.any(): print('No missing values')

train_targets_scored.head()
#sneak peak at non scored train targets

print(train_targets_nonscored.shape)

if ~ train_targets_nonscored.isnull().values.any(): print('No missing values')

train_targets_nonscored.head()
#sneak peak at test features

print(test_features.shape)

if ~ test_features.isnull().values.any(): print('No missing values')

test_features.head()
train = train_features.merge(train_targets_scored, on='sig_id', how='left')

train = train.merge(train_targets_nonscored, on='sig_id', how='left')
fig = px.histogram(train, x='cp_type', histfunc='count',

                  height=500, width=500)

fig.show()
control_ids = train.loc[train['cp_type'] == 'ctl_vehicle', 'sig_id']

train_targets_scored.loc[train_targets_scored['sig_id'].isin(control_ids)].sum()[1:].sum()
cp_time_count = train['cp_time'].value_counts().reset_index()

cp_time_count.columns = ['cp_time', 'count']



fig = px.bar(cp_time_count, x='cp_time', y='count',

             height=500, width=600)

fig.show()
fig = px.histogram(train, x='cp_dose', height=500, width=600)

fig.show()
fig = make_subplots(rows=15, cols=1)



for i in range(1,15):

    fig.add_trace(

    go.Histogram(x=train[f'g-{i}'], name=f'g-{i}'),

    row=i, col=1)





fig.update_layout(height=1200, width=800, title_text="Gene Expression Features")

fig.show()
fig = make_subplots(rows=15, cols=1)



for i in range(1,15):

    fig.add_trace(

    go.Histogram(x=train[f'c-{i}'], name=f'c-{i}'),

    row=i, col=1)





fig.update_layout(height=1200, width=800, title_text="Cell Viability Features")

fig.show()
fig = make_subplots(rows=1, cols=2)



fig.add_trace(

        go.Histogram(x=train[train_targets_scored[1:].columns.tolist()].sum(axis=1), name='Training Unique Scored Targets per Sample'),

        row=1, col=1)



fig.add_trace(

        go.Histogram(x=train[train_targets_nonscored[1:].columns.tolist()].sum(axis=1), name='Training Unique Non-Scored Targets per Sample'),

        row=1, col=2)



fig.update_layout(height=400, width=1000, title_text="Unique Labels per Sample")

fig.show()
fig = px.bar(x=train[train_targets_scored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).values,

            y=train[train_targets_scored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).index,

            height=800, width=800, color=train[train_targets_scored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).values)



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, title='Training Scored Target Classification Counts')



fig.show()
fig = px.bar(x=train[train_targets_nonscored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).values,

            y=train[train_targets_nonscored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).index,

            height=800, width=800, color=train[train_targets_nonscored.columns[1:].tolist()].sum(axis=0).sort_values(ascending=False).values)



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, title='Training NonScored Target Classification Counts')



fig.show()
fig = px.imshow(train[train_features[1:].columns.tolist()].corr(method='pearson'), 

                title='Correlations Among Training Features',

                height=800, width=800)

fig.show()
fig = px.imshow(train[[col for col in train_features.columns if 'c-' in col]].corr(method='pearson'), 

                title='Correlations Among Training Features',

                height=800, width=800)

fig.show()
c_cols = [col for col in train_features.columns if 'c-' in col]

g_cols = [col for col in train_features.columns if 'g-' in col]



c_corrs = train[[*c_cols,*train_targets_scored]].corr(method='pearson')
threshold_bad = .85

bad_c_cols = []



for col in c_corrs.iloc[:len(c_cols), :len(c_cols)].columns:

    for pair in c_corrs.iloc[:len(c_cols):, :len(c_cols)][col].iteritems():

        if abs(pair[1]) > threshold_bad:

            if pair[0] not in bad_c_cols and pair[0] is not col: 

                bad_c_cols.append(pair[0])

            

print(f"{len(bad_c_cols)} c- columns with correlation to other c- columns above {threshold_bad}")

print('')

print(bad_c_cols)
threshold_good = .65

good_c_cols = []



for col in c_corrs.iloc[:len(c_cols), len(c_cols):].columns:

    for pair in c_corrs.iloc[:len(c_cols):, len(c_cols):][col].iteritems():

        if abs(pair[1]) > threshold_good:

            if pair[0] not in good_c_cols and pair[0] is not col: 

                good_c_cols.append(pair[0])

            

print(f"{len(good_c_cols)} c- columns with correlation to target above {threshold_good}")

print('')

print(good_c_cols)
c_cols_to_drop = [col for col in bad_c_cols if col not in good_c_cols]

print(len(c_cols_to_drop))

print(c_cols_to_drop)
great_cols = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,

        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,

        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,

        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,

        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,

        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,

       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,

       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,

       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,

       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,

       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,

       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,

       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,

       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,

       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,

       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,

       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,

       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,

       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,

       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,

       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,

       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,

       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,

       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,

       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,

       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,

       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,

       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,

       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,

       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,

       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,

       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,

       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,

       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,

       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,

       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,

       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,

       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,

       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,

       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,

       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,

       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,

       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,

       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,

       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,

       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,

       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,

       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,

       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,

       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,

       864, 867, 868, 870, 871, 873, 874]
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 2, 'D2': 3})

    del df['sig_id']

    return df
train_features = preprocess(train_features)

test_features = preprocess(test_features)
train_targets_scored = train_targets_scored.drop('sig_id', axis = 1)
train_targets_scored = train_targets_scored.loc[train_features['cp_type'] == 0].reset_index(drop=True)

train_features = train_features.loc[train_features['cp_type'] == 0].reset_index(drop=True)
sample_sub.loc[:, train_targets_scored.columns] = 0
USE_NN_ENSEMBLE = False

USE_PROCESSED = True
#basic training configuration

NUM_NETS = 1

EPOCHS = 30

BATCH_SIZE = 64

VERBOSE = 0
def build_model(num_columns, num_nodes = 1024, use_swish = False, use_mish = False,

                use_relu = False, use_selu = False, batch_norm = True, dropout = .4):

    model = tf.keras.Sequential()

    

    if use_swish:

        #first layer

        if batch_norm:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

            model.add(tf.keras.layers.Dropout(.2))

        else: model.add(tf.keras.layers.Dropout(.2, input_shape=(num_columns,)))

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='swish')))

        if batch_norm: model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

    

        #second layer

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='swish')))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

                      

    if use_mish:

        #first layer

        if batch_norm:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

            model.add(tf.keras.layers.Dropout(.2))

        else: model.add(tf.keras.layers.Dropout(.2, input_shape=(num_columns,)))

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes,

                                                 activation = tfa.activations.mish)))

        if batch_norm: model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

    

        #second layer

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes,

                                                 activation=tfa.activations.mish)))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

        

    if use_relu:

        #first layer

        if batch_norm:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

            model.add(tf.keras.layers.Dropout(.2))

        else: model.add(tf.keras.layers.Dropout(.2, input_shape=(num_columns,)))

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='relu')))

        if batch_norm: model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

    

        #second layer

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='relu')))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

        

    if use_selu:

        #first layer

        if batch_norm:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

            model.add(tf.keras.layers.Dropout(.2))

        else: model.add(tf.keras.layers.Dropout(.2, input_shape=(num_columns,)))

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='selu')))

        if batch_norm: model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

    

        #second layer

        model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_nodes, activation='selu')))

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(dropout))

    

    #output layer

    model.add(tf.keras.layers.Dense(206, activation='sigmoid'))

    

    #compiler

    model.compile(optimizer = tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period = 10),

                  loss = 'binary_crossentropy', metrics = ['AUC'])

              

    return model
preds = np.zeros((test_features.shape[0], 206)) 

histories = []



if USE_NN_ENSEMBLE:

    for j in range(NUM_NETS):



        #get datasets

        train_ds = train_features.values

        train_targets = train_targets_scored.values



        #create a validation set to evaluate our model(s) performance

        train_ds, val_ds, train_targets, val_targets = train_test_split(train_ds, train_targets, test_size = 0.1)



        #some callbacks we can use

        sv = tf.keras.callbacks.ModelCheckpoint(f'net-{j}.h5', monitor = 'val_loss', verbose = 0,

                                                save_best_only = True, save_weights_only = True, mode = 'min')

        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 

                                                              verbose=VERBOSE, epsilon=1e-4, mode='min')



        #print fold info

        model = build_model(train_features.shape[1], use_swish = True)

        history = model.fit(train_ds, train_targets,

                            validation_data = (val_ds, val_targets),

                            epochs = EPOCHS, batch_size = BATCH_SIZE, 

                            callbacks = [reduce_lr_loss, sv], verbose = VERBOSE)

        histories.append(history)



        #report training results

        print(f"Neural Net {j + 1}: Epochs={EPOCHS}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(min(history.history['loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

        print('')



        #predict out of fold

        model.load_weights(f'net-{j}.h5')

        pred = model.predict(test_features)

        preds += pred / NUM_NETS
preds_proc = np.zeros((test_features.shape[0], 206)) 

histories_proc = []



if USE_NN_ENSEMBLE & USE_PROCESSED:

    for j in range(NUM_NETS):

        

        #train_dataset = train_features.drop(columns=c_cols_to_drop)

        train_dataset = train_features.iloc[:, great_cols]

        

        #get datasets

        train_ds = train_dataset.values

        train_targets = train_targets_scored.values



        #create a validation set to evaluate our model(s) performance

        train_ds, val_ds, train_targets, val_targets = train_test_split(train_ds, train_targets, test_size = 0.1)



        #some callbacks we can use

        sv = tf.keras.callbacks.ModelCheckpoint(f'net-{j}.h5', monitor = 'val_loss', verbose = 0,

                                                save_best_only = True, save_weights_only = True, mode = 'min')

        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 

                                                              verbose=VERBOSE, epsilon=1e-4, mode='min')



        #print fold info

        model = build_model(train_dataset.shape[1], use_swish = True)

        history = model.fit(train_ds, train_targets,

                            validation_data = (val_ds, val_targets),

                            epochs = EPOCHS, batch_size = BATCH_SIZE, 

                            callbacks = [reduce_lr_loss, sv], verbose = VERBOSE)

        histories_proc.append(history)



        #report training results

        print(f"Neural Net {j + 1}: Epochs={EPOCHS}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(min(history.history['loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

        print('')



        #predict out of fold

        model.load_weights(f'net-{j}.h5')

        pred = model.predict(test_features)

        preds += pred / NUM_NETS
if USE_NN_ENSEMBLE:

    print(f"Average validation loss: {np.average([min(histories[i].history['val_loss']) for i in range(len(histories))])}")

    print(f"Average validation AUC: {np.average([max(histories[i].history['val_auc']) for i in range(len(histories))])}")

    

if USE_NN_ENSEMBLE & USE_PROCESSED:

    print(f"Average validation loss: {np.average([min(histories[i].history['val_loss']) for i in range(len(histories))])}")

    print(f"Average validation AUC: {np.average([max(histories[i].history['val_auc']) for i in range(len(histories))])}")
#define function to visualize learning curves

def plot_learning_curves(histories, num): 

    fig, ax = plt.subplots(figsize = (20, 10))



    #plot losses

    for i in range(num):

        plt.plot(histories[i].history['loss'], color = 'C0')

        plt.plot(histories[i].history['val_loss'], color = 'C1')

    

    #set master title

    fig.suptitle("Model Performance", fontsize=14)



if USE_NN_ENSEMBLE:

    plot_learning_curves(histories, NUM_NETS)

    plot_learning_curves(histories_proc, NUM_NETS)
USE_SKF_ENSEMBLE = True

USE_PROCESSED = True
#basic training configuration

FOLDS = 5

REPEATS = 2

BATCH_SIZE = 64

VERBOSE = 0
sys.path.append('../input/iterativestrat/iterative-stratification-master')

from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
skf_preds = np.zeros((test_features.shape[0], 206)) 

skf_histories = []

skf = RepeatedMultilabelStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)



if USE_SKF_ENSEMBLE:

    for f, (train_index, val_index) in enumerate(skf.split(train_features.values, train_targets_scored.values)):



        #get datasets

        train_ds = train_features.values[train_index]

        train_targets = train_targets_scored.values[train_index]

        val_ds = train_features.values[val_index]

        val_targets = train_targets_scored.values[val_index]



        #some callbacks we can use

        sv = tf.keras.callbacks.ModelCheckpoint(f'fold-{f}.h5', monitor = 'val_loss', verbose = 0,

                                                save_best_only = True, save_weights_only = True, mode = 'min')

        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,

                                                              verbose=VERBOSE, epsilon=1e-4, mode='min')



        #print fold info

        model = build_model(train_features.shape[1], use_swish = True)

        history = model.fit(train_ds, train_targets,

                            validation_data = (val_ds, val_targets),

                            epochs = EPOCHS, batch_size = BATCH_SIZE, 

                            callbacks = [reduce_lr_loss, sv], verbose = VERBOSE)

        print('')

        skf_histories.append(history)



        #report training results

        print(f"Fold {f + 1}: Epochs={EPOCHS}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(min(history.history['loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

        print('')



        #predict out of fold

        model.load_weights(f'fold-{f}.h5')

        pred = model.predict(test_features)

        skf_preds += pred / FOLDS / REPEATS
skf_preds_proc = np.zeros((test_features.shape[0], 206)) 

skf_histories_proc = []

rmskf = RepeatedMultilabelStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)



if USE_SKF_ENSEMBLE & USE_PROCESSED:

    for f, (train_index, val_index) in enumerate(rmskf.split(train_features.values, train_targets_scored.values)):



        #train_dataset = train_features.drop(columns=c_cols_to_drop)

        train_dataset = train_features.iloc[:, great_cols]

        

        #get datasets

        train_ds = train_dataset.values[train_index]

        train_targets = train_targets_scored.values[train_index]

        val_ds = train_dataset.values[val_index]

        val_targets = train_targets_scored.values[val_index]



        #some callbacks we can use

        sv = tf.keras.callbacks.ModelCheckpoint(f'fold-{f}.h5', monitor = 'val_loss', verbose = 0,

                                                save_best_only = True, save_weights_only = True, mode = 'min')

        reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,

                                                              verbose=VERBOSE, epsilon=1e-4, mode='min')



        #print fold info

        model = build_model(train_dataset.shape[1],

                            use_swish = True)



        history = model.fit(train_ds, train_targets, validation_data = (val_ds, val_targets),

                            epochs = EPOCHS, batch_size = BATCH_SIZE,

                            callbacks = [reduce_lr_loss, sv], verbose = VERBOSE)

        print('')

        skf_histories_proc.append(history)



        #report training results

        print(f"Fold {f + 1}: Epochs={EPOCHS}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(min(history.history['loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

        print('')



        #predict out of fold

        model.load_weights(f'fold-{f}.h5')

        pred = model.predict(test_features.iloc[:, great_cols])

        skf_preds_proc += pred / FOLDS / REPEATS
if USE_SKF_ENSEMBLE:

    print('#'*25)

    print('SKF Ensemble Results')

    print('#'*25); print('')

    print(f"Average validation loss: {np.average([min(skf_histories[i].history['val_loss']) for i in range(len(skf_histories))])}")

    print(f"Average validation AUC: {np.average([max(skf_histories[i].history['val_auc']) for i in range(len(skf_histories))])}")

    print('')

    

if USE_SKF_ENSEMBLE & USE_PROCESSED:

    print('#'*25)

    print('SKF Ensemble Results - Processed')

    print('#'*25); print('')

    print(f"Average validation loss: {np.average([min(skf_histories_proc[i].history['val_loss']) for i in range(len(skf_histories))])}")

    print(f"Average validation AUC: {np.average([max(skf_histories_proc[i].history['val_auc']) for i in range(len(skf_histories))])}")
print(sample_sub.shape)

sample_sub.head()
if USE_NN_ENSEMBLE:

    #sample_sub.loc[:, train_targets_scored.columns] = preds

    sample_sub.loc[:, train_targets_scored.columns] = preds_proc

    

if USE_SKF_ENSEMBLE:

    #sample_sub.loc[:, train_targets_scored.columns] = skf_preds

    sample_sub.loc[:, train_targets_scored.columns] = skf_preds_proc

    

sample_sub.head()
#sanity check

sample_sub.loc[test_features['cp_type'] == 1].head()
sample_sub.loc[test_features['cp_type'] == 1, train_targets_scored.columns] = 0
#last sanity check

sample_sub.head()
sample_sub.to_csv('submission.csv', index = False)

print('Submission saved')