# Import all the necessary libraries

import os

import datetime
# Have a look at our data folder

topDir = '/kaggle'  #defaults to '/kaggle' in kaggle kernels, different if on own system e.g. '/home/user/kaggle'

os.chdir(topDir)    #changes our python working directory to the top directory of our kaggle files

print(os.listdir(os.path.join(topDir, 'input')))  #see what's in the input folder (where data is in)
# 
train_path = os.path.join(topDir, '')  #path to training data file/folder

test_path = os.path.join(topDir, '')   #path to test data file/folder
# Get training data

def get_X_data(path):

    

    return X_data

X_train = get_X_data(train_path)
# Get training data labels

def get_Y_data(path):

    

    return Y_data

Y_train = get_Y_data(train_path)
# Design our model architecture here

def Model():

    

    return model
# Set some model compile parameters

optimizer = 'adam'

loss      = 'binary_crossentropy'

metrics   = ['accuracy']



# Compile our model

model = Model()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# Runtime data augmentation
# Runtime custom callbacks
# Finally run the model!!

model.fit(X=X_train, Y=Y_train)
# Get test data

X_test = get_X_data(test_path)
# Use model to predict test labels

Y_hat = model.predict(X_test)
# Create submission DataFrame

sub = pd.DataFrame()



timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")

print('Submission output to: sub-{}.csv'.format(timestamp))

sub.to_csv('sub-{}.csv'.format(timestamp), index=False)