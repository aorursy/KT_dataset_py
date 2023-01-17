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
"""Create Keras model"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm

def create_model(input_dim, output_dim):
    # create model
    model = Sequential()
    
    # input layer
    model.add(Dense(100, input_dim=input_dim, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    # hidden layer
    model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    
    # output layer
    model.add(Dense(output_dim, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
"""Utility logic for handling data set"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from pandas.api.types import CategoricalDtype


APPLICANT_NUMERIC = ['annual_inc', 'dti', 'age_earliest_cr', 'loan_amnt', 'installment']
APPLICANT_CATEGORICAL = ['application_type', 'emp_length', 'home_ownership', 'addr_state', 'term']
CREDIT_NUMERIC = ['acc_now_delinq', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
                  'bc_util', 'delinq_2yrs', 'delinq_amnt', 'fico_range_high', 'fico_range_low',
                  'last_fico_range_high', 'last_fico_range_low', 'open_acc', 'pub_rec', 'revol_util',
                  'revol_bal', 'tot_coll_amt', 'tot_cur_bal', 'total_acc', 'total_rev_hi_lim',
                  'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats',
                  'num_bc_tl', 'num_il_tl', 'num_rev_tl_bal_gt_0', 'pct_tl_nvr_dlq',
                  'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
                  'total_il_high_credit_limit', 'all_util', 'loan_to_income',
                  'installment_pct_inc', 'il_util', 'il_util_ex_mort', 'total_bal_il', 'total_cu_tl']
LABEL = ['grade']

class LendingClubModelHelper:
    """Provides utility functions for training data"""
    def __init__(self):
        self.lcdata = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_sm = None
        self.y_sm = None
        self.x_train_new=None
        self.x_test_new=None
        self.model = None

    def read_csv(self, filename, columns):
        """Read in lending club data"""

        # ...skip the columns we're not going to use to preserve memory
        self.lcdata = pd.read_csv(filename, usecols=columns)

        # Set an ordering on our grade category so order of grades doesn't appear randome in graphs
        grade_categories = [g for g in "ABCDEFG"]
        self.lcdata["grade"] = self.lcdata["grade"].astype("category", CategoricalDtype(categories=grade_categories))

        # Sanity check that we're working with cleaned data
        bad_rows = self.lcdata.isnull().T.any().T.sum()
        if bad_rows > 0:
            print("Rows with null/NaN values: {}".format(bad_rows))
            print("Columns with null/NaN values:")
            print(pd.isnull(self.lcdata).sum() > 0)
            print("Dropping bad rows...")
            self.lcdata.dropna(axis=0, how='any', inplace=True)
            print("Rows with null/NaN values: {}".format(self.lcdata.isnull().T.any().T.sum()))
        
    def split_data(self, continuous_cols, categorical_cols, label_col, test_size=0.2, row_limit=None):
        """Divide the data in to X and y dataframes and train/test split"""

        # Subset to get feature data
        x_df = self.lcdata.loc[:, continuous_cols + categorical_cols]

        # Update our X dataframe with categorical values replaced by one-hot encoded values
        x_df = encode_categorical(x_df, categorical_cols)

        # Ensure all numeric features are on the same scale
        for col in continuous_cols:
            x_df[col] = (x_df[col] - x_df[col].mean()) / x_df[col].std()

        # Specify the target labels and flatten the array
        y = pd.get_dummies(self.lcdata[label_col])

        # When requested, limit the amount of data that will be used
        # Using entire data set can be painfully slow without a GPU!
        if row_limit != None:
            rows = np.random.binomial(1, 0.1, size=len(self.lcdata)).astype('bool')
            x_df = x_df[rows]
            y = y[rows]
            print("Using only a sample of {} observations".format(x_df.shape[0]))
            #data = self.lcdata.sample(int(row_limit))
        else:
            print("Using the full set of {} observations".format(self.lcdata.shape[0]))
            #data = self.lcdata

        # Create train and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_df, y, test_size=test_size, random_state=23)
        
        print("x_train contains {} rows and {} features".format(self.x_train.shape[0], self.x_train.shape[1]))
    
        from imblearn.over_sampling import SMOTENC
        sm= SMOTENC(sampling_strategy={5:25000,6:20000},random_state=42,categorical_features=[x for x in range(44,113)])
        self.x_sm, self.y_sm = sm.fit_resample(self.x_train.to_numpy(), self.y_train.to_numpy())
        #'Individual','Joint App','1 year','10+ years','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','< 1 year'
        # ,'ANY','MORTGAGE','OWN','RENT','AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI'
        # ,'MN','MO','MS','MT' , 'NC'  ,'ND'  ,'NE' , 'NH' , 'NJ' , 'NM' , 'NV' , 'NY' , 'OH'  ,'OK'  ,'OR'  ,'PA'  ,'RI'  ,'SC'  ,'SD' , 'TN',
        #'TX' , 'UT' , 'VA'  ,'VT'  ,'WA'  ,'WI' , 'WV','WY','36 months','60 months'
        print("x_sm contains {} rows and {} features".format(self.x_sm.shape[0], self.x_sm.shape[1]))

        from sklearn.feature_selection import SelectFromModel
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import ExtraTreesClassifier
        etc = ExtraTreesClassifier(n_estimators=100)
        etc.fit(self.x_sm, self.y_sm)
        # make predictions for test data and evaluate
        y_pred = etc.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy of ExtraTreesClassifier: %.2f%%" % (accuracy * 100.0))
        #feature selection    
        selection = SelectFromModel(etc,prefit=True)
        self.x_train_new = selection.transform(self.x_sm)
        self.x_test_new = selection.transform(self.x_test)
        print("x_train_new contains {} rows and {} features".format(self.x_train_new.shape[0], self.x_train_new.shape[1]))

    
    def train_model(self, model_func, gpu_enabled=True):
        """Create and train the neural network model"""

        # Create model using provided model function
        self.model = model_func(self.x_train_new.shape[1], self.y_sm.shape[1])

        # Model converges faster on larger data set with larger batches
        epochs = 30 if gpu_enabled else 45

        # GPU is actually *slower* than CPU when using small batch size
        batch_sz = 1024 if gpu_enabled else 64

        print("Beginning model training with batch size {} and {} epochs".format(batch_sz, epochs))

        checkpoint = ModelCheckpoint("lc_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        # train the model
        history = self.model.fit(self.x_train_new,
                        self.y_sm,
                        epochs=epochs,  
                        batch_size=batch_sz,
                        validation_split=0.2,
                        verbose=2,
                        callbacks=[checkpoint])
        
        # revert to the best model encountered during training
        self.model = load_model("lc_model.hdf5")
        return history

def encode_categorical(frame, categorical_cols):
    """Replace categorical variables with one-hot encoding in-place"""
    for col in categorical_cols:
        # use get_dummies() to do one hot encoding of categorical column
        frame = frame.merge(pd.get_dummies(frame[col]), left_index=True, right_index=True)
        
        # drop the original categorical column
        frame.drop(col, axis=1, inplace=True)
  
    return frame
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

def plot_correlation_matrix(data):
    """Plots a correlation matrix of the data set"""
    sns.reset_orig()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(data.corr(), cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticklabels(data, rotation=60)
    ax.set_yticklabels(data)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def plot_history(history, save_file = None):
    """Plost accuracy and loss for a Keras history object"""

    plt.rcParams["figure.figsize"] = [16, 5]
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.framealpha'] = 0.7
    plt.rcParams["legend.fontsize"] = 14

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid.'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid.'], loc='upper left')

    # adjust size
    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
        plt.close()

def fmt_pct(x, pos):
    """Format as percentage"""
    return '{:.0%}'.format(x)

def plot_confusion_matrix(conf_matrix, classes,
                          title='Confusion matrix', save_file = None):
    """
    This function prints and plots the confusion matrix.
    The style is *roughly* similar to the AWS machine learning confusion matrix
    """

    correct_mask = np.ones(conf_matrix.shape, dtype=bool)
    wrong_mask = np.zeros(conf_matrix.shape, dtype=bool)
    for i in range(conf_matrix.shape[0]):
        correct_mask[i,i] = False
        wrong_mask[i,i] = True
        row_sum = sum(conf_matrix[i])
        for j in range(conf_matrix.shape[1]):
            conf_matrix[i, j] = conf_matrix[i, j] / row_sum

    correct_matrix = np.ma.masked_array(conf_matrix, mask=correct_mask)
    wrong_matrix = np.ma.masked_array(conf_matrix, wrong_mask)

    fig,ax = plt.subplots(figsize=(8, 8))
    blue_map = colors.LinearSegmentedColormap.from_list('custom blue', ['#f1eef6', '#025a90'], N=10)
    blue_map.set_under(color='white')

    red_map = colors.LinearSegmentedColormap.from_list('custom blue', ['#feeddd', '#a83500'], N=10)
    red_map.set_under(color='white')
    
    plot_correct = ax.imshow(correct_matrix,interpolation='nearest',cmap=blue_map, vmin=0.000001, vmax=1)
    plot_wrong = ax.imshow(wrong_matrix,interpolation='nearest',cmap=red_map, vmin=0.000001, vmax=1)
    
    colorbar_wrong = plt.colorbar(plot_wrong, shrink=0.35, orientation='horizontal', pad=-0.11, format=ticker.FuncFormatter(fmt_pct))
    colorbar_correct = plt.colorbar(plot_correct, shrink=0.35, orientation='horizontal', pad=0.03)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels([''] + classes)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_yticklabels([''] + classes)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    colorbar_correct.ax.text(-0.3,0.25,'CORRECT',rotation=0)
    colorbar_wrong.ax.text(-0.35,0.25,'INCORRECT',rotation=0)
    plt.setp(colorbar_correct.ax.get_xticklabels(), visible=False)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        cell_label = "{:.1%}".format(conf_matrix[i, j])
        plt.text(j, i, cell_label,
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=72)
        plt.close()
"""Trains neural net for Lending Club dataset and saves the results"""

import numpy as np
import pandas as pd
from keras import backend as K
pd.set_option("display.max_columns", None)
helper = LendingClubModelHelper()

# Read in lending club data 
helper.read_csv("../input/lc2017q32018loanscleaned/lc-2018-loans.csv", 
                APPLICANT_NUMERIC +
                APPLICANT_CATEGORICAL +
                CREDIT_NUMERIC +
                LABEL)

# Divide the data set into training and test sets
helper.split_data(APPLICANT_NUMERIC + CREDIT_NUMERIC,
                  APPLICANT_CATEGORICAL,
                  LABEL,
                  test_size=0.2,
                  row_limit = None)

history = helper.train_model(create_model, True)

output = 'output/'
if not os.path.exists(output):
    os.makedirs(output)
helper.model.save("{}lc_model.h5".format(output))  # creates a HDF5 file 'lc_model.h5'
#np.savetxt("{}x_test.csv".format(output), helper.x_test[:100].to_numpy(), delimiter=',')
#np.savetxt("{}y_test.csv".format(output), helper.y_test[:100].to_numpy(), delimiter=',')
#y_pred = helper.model.predict(helper.x_test_new[:100])
#np.savetxt("{}y_pred.csv".format(output), y_pred, delimiter=',')
plot_history(history, "{}accuracy_loss.png".format(output))

# Save confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

y_pred = helper.model.predict(helper.x_test_new)
y_pred_classes = pd.DataFrame((y_pred.argmax(1)[:,None] == np.arange(y_pred.shape[1])), \
                                columns=helper.y_test.columns, \
                                index=helper.y_test.index)
y_test_vals = helper.y_test.idxmax(1)
y_pred_vals = y_pred_classes.idxmax(1)
f1 = f1_score(y_test_vals, y_pred_vals, average='weighted')
acc= accuracy_score(y_test_vals, y_pred_vals)
print("Test Set f1_score: {:.2%}".format(f1))
print("Test Set Accuracy: {:.2%}".format(acc))


cfn_matrix = confusion_matrix(y_test_vals, y_pred_vals)
plot_confusion_matrix(cfn_matrix.astype(float), [l for l in "ABCDEFG"], save_file="{}confusion_matrix.png".format(output))