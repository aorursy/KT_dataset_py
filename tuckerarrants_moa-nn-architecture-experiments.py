import warnings

warnings.filterwarnings('ignore')



#the basics

import pandas as pd, numpy as np

import math, re, gc, random, os, sys

from matplotlib import pyplot as plt

from tqdm import tqdm



#tensorflow deep learning basics

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K



#for model evaluation

from sklearn.model_selection import train_test_split, KFold
cfg = {

    

    'training_params': {

        'method' : 'random',

        'epochs' : 30,

        'skf_folds' : 5,

        'train_val_split' : .2,

        'skf_repeats' : 1,

        'random_repeats' : 20,

        'feature_selection' : None,

        'batch_size' : 64,

        'seed' : 34,

        'verbose' : 2

    },

    

    

    'model_params': {

        'layers': 2,

        'nodes': [1024, 1024],

        'activations': [tf.keras.activations.swish] * 2,

        'batch_norms': [True, True, True],

        'weight_norms' : [True, True, True],

        'dropouts' : [True, True, True],

        'dropout_rates' : [.2, .4, .4]

    }

}
#set how many hidden layers

cfg['model_params']['layers'] = 3



#select node count of layer

cfg['model_params']['nodes'] = [1024, 512, 256]



#select where to place dropout

cfg['model_params']['dropouts'] = [1, 0, 0, 1]



#select rate of dropouts

cfg['model_params']['dropout_rates'] = [.2, 0, 0, .4]



#select where to apply batch norm

cfg['model_params']['batch_norms'] = [1, 1, 1, 1]



#select where to apply weight norm

cfg['model_params']['weight_norms'] = [0, 0, 0]



#select swish activation for all layers

cfg['model_params']['activations'] = [tf.keras.activations.swish] * 3
def build_model(cfg, num_columns):

    

    #define for convenience

    model_cfg = cfg['model_params']

    

    #initialize empty model

    model = tf.keras.Sequential()

    

#############################################################

#### Hidden layers

#############################################################



    for layer in range(model_cfg['layers']):



        #add batch norm before activation

        if model_cfg['batch_norms'][layer] and not model_cfg['dropouts'][layer]:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

        

        #add batch norm before activation, then dropout

        if model_cfg['batch_norms'][layer] and model_cfg['dropouts'][layer]:

            model.add(tf.keras.layers.BatchNormalization(input_shape=(num_columns,)))

            model.add(tf.keras.layers.Dropout(model_cfg['dropout_rates'][layer]))

            

        #add only dropout without batch norm

        if not model_cfg['batch_norms'][layer] and model_cfg['dropouts'][layer]:

            model.add(tf.keras.layers.Dropout(model_cfg['dropout_rates'][layer],

                      input_shape=(num_columns,)))

            

        #add activation layer with weight normalization

        if model_cfg['weight_norms'][layer]:

            model.add(tfa.layers.WeightNormalization(tf.keras.layers.Dense(model_cfg['nodes'][layer], activation=model_cfg['activations'][layer])))

             

        #add activation layer without weight normalization

        if not model_cfg['weight_norms'][layer]:

            model.add(tf.keras.layers.Dense(model_cfg['nodes'][layer], activation=model_cfg['activations'][layer]))

            

################################################################

#### Output layer

################################################################



    if model_cfg['batch_norms'][model_cfg['layers']] and model_cfg['dropouts'][model_cfg['layers']]:

        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dropout(model_cfg['dropout_rates'][layer]))



    if model_cfg['batch_norms'][model_cfg['layers']] and not model_cfg['dropouts'][model_cfg['layers']]:

        model.add(tf.keras.layers.BatchNormalization())



    model.add(tf.keras.layers.Dense(206, activation='sigmoid'))

    

    #add compiler

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(lr = 5e-4), sync_period=10),

                  loss='binary_crossentropy', metrics=['AUC'])   

    

    return model
example_model = build_model(cfg, num_columns = 100)

example_model.summary()
#load files into memory as Pandas DataFrames

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 2, 'D2': 3})

    del df['sig_id']

    

    return df



#process datasets

train_features = preprocess(train_features)

test_features = preprocess(test_features)

train_targets_scored = train_targets_scored.drop('sig_id', axis = 1)
train_targets_scored = train_targets_scored.loc[train_features['cp_type'] == 0].reset_index(drop=True)

train_features = train_features.loc[train_features['cp_type'] == 0].reset_index(drop=True)
# https://www.kaggle.com/stanleyjzheng/multilabel-neural-network-improved

top_feats1 = [  1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,

        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,

        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,

        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,

        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,

        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,

        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,

       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,

       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,

       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,

       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,

       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,

       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,

       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,

       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,

       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,

       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,

       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,

       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,

       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,

       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,

       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,

       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,

       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,

       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,

       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,

       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,

       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,

       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,

       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,

       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,

       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,

       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,

       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,

       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,

       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,

       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,

       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,

       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,

       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,

       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,

       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,

       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,

       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,

       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,

       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,

       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,

       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,

       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,

       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,

       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,

       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,

       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,

       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,

       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,

       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,

       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,

       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,

       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,

       870, 871, 872, 873, 874]



print(len(top_feats1))
#https://www.kaggle.com/nicohrubec/pytorch-multilabel-neural-network

top_feats2 = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

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



print(len(top_feats2))
sys.path.append('../input/iterativestrat/iterative-stratification-master')

from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
def train_model(cfg):

    

#########################################################

#### Random training loop

#########################################################



    if cfg['training_params']['method'] is 'random':

        

        #define for convenience

        train_cfg = cfg['training_params']



        #save training results

        random_results = []

        

        for j in range(train_cfg['random_repeats']):



            #get datasets

            if train_cfg['feature_selection'] is not None:

                train_dataset = train_features.iloc[:, train_cfg['feature_selection']]    

            if train_cfg['feature_selection'] is None:

                train_dataset = train_features

                

            train_targets = train_targets_scored



            #create a validation set to evaluate our model(s) performance

            train_ds, val_ds, train_targets, val_targets = train_test_split(train_dataset, train_targets,

                                                                            test_size = train_cfg['train_val_split'])



            #some callbacks we can use

            sv = tf.keras.callbacks.ModelCheckpoint(f'net-{j}.h5', monitor = 'val_loss', verbose = 0,

                                                    save_best_only = True, save_weights_only = True, mode = 'min')



            reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 

                                                                  verbose=train_cfg['verbose'], epsilon=1e-4, mode='min')

            

            if train_cfg['feature_selection'] is not None:

                model = build_model(cfg, num_columns = len(train_cfg['feature_selection']))

            else: model = build_model(cfg, num_columns = 875)   

                

            history = model.fit(train_ds, train_targets,

                                validation_data = (val_ds, val_targets),

                                epochs = train_cfg['epochs'], batch_size = train_cfg['batch_size'], 

                                callbacks = [reduce_lr_loss], verbose = train_cfg['verbose'])

            print('')

            random_results.append(min(history.history['val_loss']))

            

            print(f"Neural Net {j + 1}: Epochs={train_cfg['epochs']}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(min(history.history['loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

            print('')

        

        return random_results



#########################################################

#### StratKFold training loop

#########################################################



    if cfg['training_params']['method'] is 'skf':

        

        #define for convenience

        train_cfg = cfg['training_params']

        

        #get stratified kfold split object

        rmskf = RepeatedMultilabelStratifiedKFold(n_splits=train_cfg['skf_folds'], n_repeats=train_cfg['skf_repeats'],

                                                  random_state=train_cfg['seed'])

        

        #save training results

        skf_results = []

        

        for f, (train_index, val_index) in enumerate(rmskf.split(train_features.values,

                                                                 train_targets_scored.values)):



            #get features

            if train_cfg['feature_selection'] is not None:

                train_dataset = train_features.iloc[train_index, train_cfg['feature_selection']].values

                val_dataset = train_features.iloc[val_index, train_cfg['feature_selection']].values

            if train_cfg['feature_selection'] is None:

                train_dataset = train_features[train_index].values

                val_dataset = train_features[val_index].values

            

            #get targets

            train_targets = train_targets_scored.values[train_index]

            val_targets = train_targets_scored.values[val_index]



            #some callbacks we can use

            sv = tf.keras.callbacks.ModelCheckpoint(f'fold-{f}.h5', monitor = 'val_loss', verbose = 0,

                                                    save_best_only = True, save_weights_only = True, mode = 'min')

            

            reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,

                                                                  verbose=train_cfg['verbose'], epsilon=1e-4, mode='min')

        

            if train_cfg['feature_selection'] is not None: model = build_model(cfg, num_columns = len(train_cfg['feature_selection']))

            else: model = build_model(cfg, num_columns = 875)   

                

            history = model.fit(train_dataset, train_targets,

                                validation_data = (val_dataset, val_targets),

                                epochs = train_cfg['epochs'],

                                batch_size = train_cfg['batch_size'], 

                                callbacks = [reduce_lr_loss],

                                verbose = train_cfg['verbose'])

            

            print('')

            skf_results.append(min(history.history['val_loss']))

            

            print(f"Fold {f + 1}: Epochs={train_cfg['epochs']}, Train AUC={round(max(history.history['auc']), 5)}, Train loss={round(max(history.history['val_loss']), 5)}, Validation AUC={round(max(history.history['val_auc']), 5)}, Validation loss={round(min(history.history['val_loss']), 5)}")  

            print('')



        return skf_results
cfg1 = cfg.copy()
#model specific

cfg1['model_params']['layers'] = 2

cfg1['model_params']['nodes'] = [1024, 1024]

cfg1['model_params']['dropouts'] = [1, 1, 1]

cfg1['model_params']['dropout_rates'] = [.2, .4, .4]

cfg1['model_params']['batch_norms'] = [1, 1, 1]

cfg1['model_params']['weight_norms'] = [1, 1, 1]

cfg1['model_params']['activations'] = [tf.keras.activations.swish] * 2



#training specific

cfg1['training_params']['method'] = 'skf'    

cfg1['training_params']['epochs'] = 30  

cfg1['training_params']['skf_folds'] = 5   

cfg1['training_params']['train_val_split'] = .2

cfg1['training_params']['skf_repeats'] = 1

cfg1['training_params']['random_repeats'] = 5

cfg1['training_params']['feature_selection'] = None

cfg1['training_params']['verbose'] = 0
cfg2 = cfg.copy()
#model specific

cfg2['model_params']['layers'] = 2

cfg2['model_params']['nodes'] = [1024, 1024]

cfg2['model_params']['dropouts'] = [1, 1, 1]

cfg2['model_params']['dropout_rates'] = [.2, .4, .4]

cfg2['model_params']['batch_norms'] = [1, 1, 1]

cfg2['model_params']['weight_norms'] = [1, 1, 1]

cfg2['model_params']['activations'] = [tf.keras.activations.swish] * 2



#training specific

cfg2['training_params']['method'] = 'skf'    

cfg2['training_params']['epochs'] = 30  

cfg2['training_params']['skf_folds'] = 5   

cfg2['training_params']['train_val_split'] = .2

cfg2['training_params']['skf_repeats'] = 1

cfg2['training_params']['random_repeats'] = 5

cfg2['training_params']['feature_selection'] = top_feats1

cfg2['training_params']['verbose'] = 0
cfg3 = cfg.copy()
#model specific

cfg3['model_params']['layers'] = 2

cfg3['model_params']['nodes'] = [1024, 1024]

cfg3['model_params']['dropouts'] = [1, 1, 1]

cfg3['model_params']['dropout_rates'] = [.2, .4, .4]

cfg3['model_params']['batch_norms'] = [1, 1, 1]

cfg3['model_params']['weight_norms'] = [1, 1, 1]

cfg3['model_params']['activations'] = [tf.keras.activations.swish] * 2



#training specific

cfg3['training_params']['method'] = 'skf'    

cfg3['training_params']['epochs'] = 30  

cfg3['training_params']['skf_folds'] = 5   

cfg3['training_params']['train_val_split'] = .2

cfg3['training_params']['skf_repeats'] = 1

cfg3['training_params']['random_repeats'] = 5

cfg3['training_params']['feature_selection'] = top_feats2

cfg3['training_params']['verbose'] = 0
cfgs = [cfg1, cfg2, cfg3]

results = []



for cfg in cfgs:

    histories = train_model(cfg)

    results.append(histories)
for f, result in enumerate(results):

    print(f"Model {f + 1} validation loss = {round(np.average(result), 7)}, STD = {round(np.std(result), 7)}")