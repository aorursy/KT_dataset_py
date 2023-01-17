import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import average_precision_score, f1_score

from sklearn.metrics import recall_score, accuracy_score

        

train_plus = pd.read_csv("/kaggle/input/nslkdd/KDDTrain+.txt", header=None)

test_plus = pd.read_csv("/kaggle/input/nslkdd/KDDTest+.txt", header=None)

test_minus_twentyone = pd.read_csv("/kaggle/input/nslkdd/KDDTest-21.txt", header=None)



colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 

            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 

            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',

            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 

            'srv_rerror_rate',  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 

            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',

            'dst_host_srv_rerror_rate', 'class']



train_plus.drop(42, axis=1, inplace=True)

test_plus.drop(42, axis=1, inplace=True)

test_minus_twentyone.drop(42, axis=1, inplace=True)



train_plus.columns = colnames

test_plus.columns = colnames

test_minus_twentyone.columns = colnames



# Concat the df's together

df = pd.concat([train_plus, test_plus, test_minus_twentyone], ignore_index=True)



# Convert the attack label to a binary classification problem 0=normal 1=attack

df["attack"] = df["class"].apply(lambda x: 0 if x=="normal" else 1)



# Get the one-hot encoding

one_hot = pd.get_dummies(df[["protocol_type", "service", "flag"]])

df = df.join(one_hot)

df.drop(["protocol_type", "service", "flag"], inplace=True, axis=1)



# resplit the data

train_plus = df.iloc[0:125973, :]

test_plus = df.iloc[125973:148517, :]

test_minus_twentyone = df.iloc[148517:,:]



# Create the training target for each dataset

y_train = np.array(train_plus["attack"])

y_test = np.array(test_plus["attack"])

y_test_minus_twentyone = np.array(test_minus_twentyone["attack"])



# Scale the data

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train_plus.drop(["attack", "class"], axis=1))

X_test = scaler.transform(test_plus.drop(["attack", "class"], axis=1))

X_test_minus_twentyone = scaler.transform(test_minus_twentyone.drop(["attack", "class"], axis=1))



print(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")

print(f"X_test shape: {X_test.shape} y_test shape: {y_test.shape}")

print(f"X_test_minus_twentyone shape: {X_test_minus_twentyone.shape} y_test_minus_twentyone shape: {y_test_minus_twentyone.shape}")
# Create a helper function to plot the loss / accuracy



def plot_loss(history):



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(loss) + 1)



    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    plt.show()



    plt.clf()



    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']



    plt.plot(epochs, acc, 'bo', label='Training accuracy')

    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

    plt.title('Training and validation Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
# Define the layers

input_layer = tf.keras.Input(shape=(122))

x = tf.keras.layers.Dense(122, activation="relu")(input_layer)

x = tf.keras.layers.Dense(64, activation="relu")(x)

x = tf.keras.layers.Dense(32, activation="relu")(x)

output_layer = tf.keras.layers.Dense(2, activation="softmax")(x)



# Create the model

fcn = tf.keras.models.Model(input_layer, output_layer, name="Fully_Connected_Network")



# Compile the model

fcn.compile(optimizer = 'adam', loss="SparseCategoricalCrossentropy", metrics = ['accuracy'])

fcn.summary()



# Fit the model

history = fcn.fit(X_train, y_train, epochs=40, batch_size=256, validation_split = 0.3, verbose=False)





plot_loss(history)

tf.keras.utils.plot_model(fcn, show_shapes=True)
# Fit the model

history = fcn.fit(X_train, y_train, epochs=7, batch_size=256, verbose=False)
# Test+

preds = fcn.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict = {"FCN-Test+":accuracy}

recall_dict = {"FCN-Test+":recall}

precision_dict = {"FCN-Test+":precision}

f1_dict = {"FCN-Test+":f1}



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



# Test-21

preds = fcn.predict(X_test_minus_twentyone) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test_minus_twentyone, pred_labels), 3)    

recall = np.round(recall_score(y_test_minus_twentyone, pred_labels), 3)

precision = np.round(average_precision_score(y_test_minus_twentyone, pred_labels), 3)

f1 = np.round(f1_score(y_test_minus_twentyone, pred_labels), 3)







print(f"\nTest-21\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



accuracy_dict["FCN-Test-21"] = accuracy

recall_dict["FCN-Test-21"] = recall

precision_dict["FCN-Test-21"] = precision

f1_dict["FCN-Test-21"] = f1
encoded_size=32



# ENCODER

encoder_input_layer = tf.keras.Input(shape=(122))

dense_encoder = tf.keras.layers.Dense(122, activation="relu")(encoder_input_layer)

latent_output = tf.keras.layers.Dense(encoded_size, activation=None)(dense_encoder)

encoder = tf.keras.models.Model(encoder_input_layer, latent_output, name="Encoder")



# DECODER

decoder_input_layer = tf.keras.Input(shape=(encoded_size))

dense_decoder = tf.keras.layers.Dense(64, activation="relu")(decoder_input_layer)

decoder_output = tf.keras.layers.Dense(122, activation=None)(dense_decoder)

decoder = tf.keras.models.Model(decoder_input_layer, decoder_output, name="Decoder")



# AUTOENCODER

input_layer = tf.keras.Input(shape=(122))

encoder_layer = encoder(input_layer)

decoder_layer = decoder(encoder_layer)

autoencoder = tf.keras.models.Model(input_layer, decoder_layer, name="Autoencoder")



autoencoder.compile(optimizer='adam', loss="MSE")

autoencoder.summary()



# We are trying to recreate the input so the target is also X_train

autoencoder.fit(X_train, X_train, epochs=5, batch_size=256)



# Now we need to freeze the weights of the encoder

encoder.trainable = False



# Use the encoder with a FCN attached for classification

x = tf.keras.layers.Dense(32, activation="relu")(encoder_layer)

x = tf.keras.layers.Dense(16, activation="relu")(x)

fcn_output = tf.keras.layers.Dense(2, activation="softmax")(x)



autoencoder_fcn = tf.keras.models.Model(input_layer, fcn_output, name="Autoencoder_FCN")

autoencoder_fcn.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])

autoencoder_fcn.summary()

autoencoder_fcn.fit(X_train, y_train, epochs=5, batch_size=256)



loss, accuracy = autoencoder_fcn.evaluate(X_test,y_test, batch_size=256)

print("Accuracy", accuracy)
# Test+

preds = autoencoder_fcn.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["AE-FCN-Test+"] = accuracy

recall_dict["AE-FCN-Test+"] = recall

precision_dict["AE-FCN-Test+"] = precision

f1_dict["AE-FCN-Test+"] = f1



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



# Test-21

preds = autoencoder_fcn.predict(X_test_minus_twentyone) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test_minus_twentyone, pred_labels), 3)    

recall = np.round(recall_score(y_test_minus_twentyone, pred_labels), 3)

precision = np.round(average_precision_score(y_test_minus_twentyone, pred_labels), 3)

f1 = np.round(f1_score(y_test_minus_twentyone, pred_labels), 3)



print(f"\nTest-21\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



accuracy_dict["AE-FCN-Test-21"] = accuracy

recall_dict["AE-FCN-Test-21"] = recall

precision_dict["AE-FCN-Test-21"] = precision

f1_dict["AE-FCN-Test-21"] = f1
# Custom preprocessing required for the Seq2Seq model



# Reshape the data input for LSTM input

X_train = X_train.reshape(-1,1,122)

y_train = y_train.reshape(-1,1)

print(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")



X_test = X_test.reshape(-1,1,122)

y_test = y_test.reshape(-1,1)

print(f"X_test shape: {X_test.shape} y_test shape: {y_test.shape}")



X_test_minus_twentyone = X_test_minus_twentyone.reshape(-1,1,122)
# Layers

input_layer = tf.keras.Input(shape=(1,122), name="Input_Layer")

encoder_lstm_one = tf.keras.layers.LSTM(122, return_sequences=True, name="Encoder_LSTM_One", input_shape=(50,122))(input_layer)

encoder_lstm_two = tf.keras.layers.LSTM(122, return_sequences=True, name="Encoder_LSTM_Two")(encoder_lstm_one)

output, state_h, state_c = tf.keras.layers.LSTM(122, return_state=True, name="Encoder_LSTM_Three")(encoder_lstm_two)

encoder_state = [state_h, state_c]

output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="Encoder_Classifier")(output)



# Model

encoder = tf.keras.models.Model(input_layer, output_layer, name="Encoder")

encoder.compile(optimizer = 'adam', loss="MSE", metrics = ['accuracy'])





# DECODER

# Layers

decoder_lstm_one = tf.keras.layers.LSTM(122, name="Decoder_LSTM_One", input_shape=(50,122), return_sequences=True)(input_layer, initial_state=encoder_state)

decoder_lstm_two = tf.keras.layers.LSTM(122, name="Decoder_LSTM_Two", return_sequences=True)(decoder_lstm_one)

decoder_lstm_three = tf.keras.layers.LSTM(122, name="Decoder_LSTM_Three")(decoder_lstm_two)

decoder_output_layer = tf.keras.layers.Dense(2, activation="softmax", name="Decoder_Classifier")(decoder_lstm_three)



# Model

seq2seq = tf.keras.models.Model(input_layer, decoder_output_layer, name="Seq2Seq")

seq2seq.compile(optimizer = 'adam', loss="SparseCategoricalCrossentropy", metrics = ['accuracy'])

seq2seq.summary()



# Fit

seq2seq.fit(X_train, y_train, epochs=5)



# EVALUATE

loss, accuracy = seq2seq.evaluate(X_test,y_test)

print("Accuracy", accuracy)
# Reshape the test data back to normal

y_test = y_test.flatten()

y_test_minus_twentyone = y_test_minus_twentyone.flatten()



# Test+

preds = seq2seq.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["Seq2Seq-Test+"] = accuracy

recall_dict["Seq2Seq-Test+"] = recall

precision_dict["Seq2Seq-Test+"] = precision

f1_dict["Seq2Seq-Test+"] = f1



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



# Test-21

preds = seq2seq.predict(X_test_minus_twentyone) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test_minus_twentyone, pred_labels), 3)    

recall = np.round(recall_score(y_test_minus_twentyone, pred_labels), 3)

precision = np.round(average_precision_score(y_test_minus_twentyone, pred_labels), 3)

f1 = np.round(f1_score(y_test_minus_twentyone, pred_labels), 3)



print(f"\nTest-21\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



accuracy_dict["Seq2Seq-Test-21"] = accuracy

recall_dict["Seq2Seq-Test-21"] = recall

precision_dict["Seq2Seq-Test-21"] = precision

f1_dict["Seq2Seq-Test-21"] = f1
tf.keras.utils.plot_model(seq2seq)
# We currently have one-hot encoded columns for categorical variables

# Convert each group into a one hot vector in a single column



# Get a list of each group of columns

protocol_cols = [col for col in df.columns if 'protocol_' in col]

service_cols = [col for col in df.columns if 'service_' in col]

flag_cols = [col for col in df.columns if 'flag_' in col]



# Function to convert each group of cols into a single col

def vectorize_cols(col_list, dataframe):



    col = []

    # Iterate over each row

    for row in dataframe[col_list].itertuples():

        row_entry = []

        

        # Iterate over each column

        for i in range(1,len(col_list)+1):

            row_entry.append(row[i])

        

        col.append(row_entry)

    

    return col



df["flag"] = vectorize_cols(flag_cols, df)

df["service"] = vectorize_cols(service_cols, df)

df["protocol"] = vectorize_cols(protocol_cols, df)



# Drop the individual columns

df.drop(protocol_cols, inplace=True, axis=1)

df.drop(service_cols, inplace=True, axis=1)

df.drop(flag_cols, inplace=True, axis=1)



# df.loc[:,"flag"] = df["flag"].apply(lambda x: np.asarray(x))

# df.loc[:,"service"] = df["fl"].apply(lambda x: np.asarray(x))

# df.loc[:,"protocol"] = df["flag"].apply(lambda x: np.asarray(x))



# resplit the data

train_plus = df.iloc[0:125973, :]

test_plus = df.iloc[125973:148517, :]

test_minus_twentyone = df.iloc[148517:,:]



# Create the training target for each dataset

y_train = np.array(train_plus["attack"])

y_test = np.array(test_plus["attack"])

y_test_minus_twentyone = np.array(test_minus_twentyone["attack"])



# Scale the data with standard scaler 

# do not include the training targets or the already vectorized columns

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train_plus.drop(["attack", "class", "flag", "service", "protocol"], axis=1))

X_test = scaler.transform(test_plus.drop(["attack", "class", "flag", "service", "protocol"], axis=1))

X_test_minus_twentyone = scaler.transform(test_minus_twentyone.drop(["attack", "class", "flag", "service", "protocol"], axis=1))



# Discretize the data into 10 bins

X_train = np.digitize(X_train, np.arange(0,0.9,.1))

X_test = np.digitize(X_test, np.arange(0,0.9,.1))

X_test_minus_twentyone = np.digitize(X_test_minus_twentyone, np.arange(0,0.9,.1))



# One hot encode each value

X_train = tf.keras.utils.to_categorical(X_train, num_classes=10)

X_test = tf.keras.utils.to_categorical(X_test, num_classes=10)

X_test_minus_twentyone = tf.keras.utils.to_categorical(X_test_minus_twentyone, num_classes=10)



# Reshape the data ready to concatenate categorical data back in

X_train = X_train.reshape(125973, -1)

X_test = X_test.reshape(22544, -1)

X_test_minus_twentyone = X_test_minus_twentyone.reshape(11850, -1)



# Add the categorical data back in

# X_train

protocol = np.vstack(train_plus["protocol"].apply(lambda x: np.asarray(x)))

service = np.vstack(train_plus["service"].apply(lambda x: np.asarray(x)))

flag = np.vstack(train_plus["flag"].apply(lambda x: np.asarray(x)))

X_train = np.concatenate((X_train, protocol, service, flag), axis=1)



# X_test

protocol = np.vstack(test_plus["protocol"].apply(lambda x: np.asarray(x)))

service = np.vstack(test_plus["service"].apply(lambda x: np.asarray(x)))

flag = np.vstack(test_plus["flag"].apply(lambda x: np.asarray(x)))

X_test = np.concatenate((X_test, protocol, service, flag), axis=1)



# X_test_minus_twentyone

protocol = np.vstack(test_minus_twentyone["protocol"].apply(lambda x: np.asarray(x)))

service = np.vstack(test_minus_twentyone["service"].apply(lambda x: np.asarray(x)))

flag = np.vstack(test_minus_twentyone["flag"].apply(lambda x: np.asarray(x)))

X_test_minus_twentyone = np.concatenate((X_test_minus_twentyone, protocol, service, flag), axis=1)



print("We now have the correct shape described in the paper for the number of features")

print(X_train.shape, X_test.shape, X_test_minus_twentyone.shape)



# * We now have the correct shape described in the paper for the number of features

# * Convert each 8 bits into a pixel

# * Then pad the array's so we can have the correct size for a square image

# * We currently have data for 464 / 8 = 58 pixels

# * We need data for 8x8 = 64 pixels

# * 6x8 = 48 -> we must pad the current 464 dimension to 512 length with zeroes



X_train = np.pad(X_train, ((0,0), (0,48)), "constant", constant_values=0)

X_test = np.pad(X_test, ((0,0), (0,48)), "constant", constant_values=0)

X_test_minus_twentyone = np.pad(X_test_minus_twentyone, ((0,0), (0,48)), "constant", constant_values=0)

print("\nPad out the data so we can have 8x8")

print(X_train.shape, X_test.shape, X_test_minus_twentyone.shape)



# We need to convert each 8bit segment into a 0-255 value, reshape so we can.

X_train = X_train.reshape(-1, 64, 8).astype("int")

X_test = X_test.reshape(-1, 64, 8).astype("int")

X_test_minus_twentyone = X_test_minus_twentyone.reshape(-1, 64, 8).astype("int")



# Convert binary array to 0-255

X_train = np.packbits(X_train, axis=2)

X_test = np.packbits(X_test, axis=2)

X_test_minus_twentyone = np.packbits(X_test_minus_twentyone, axis=2)



# Reshape into an 8x8 image

X_train = X_train.reshape(-1, 8, 8)

X_test = X_test.reshape(-1, 8, 8)

X_test_minus_twentyone = X_test_minus_twentyone.reshape(-1, 8, 8)

print("\nNow we have the correct shape of an image")

print(X_train.shape, X_test.shape, X_test_minus_twentyone.shape)



# Have a look at the first example

print("\nVisualize the first observation")

plt.imshow(X_train[0,:,:], cmap='gray', vmin=0, vmax=255)

plt.show()



# The paper leaves out some information regarding preprocessing

# ResNet50 was trained on RGB images and requires three channels of data

# To get around this I will repeat the greyscale image three times



X_train = np.repeat(X_train[..., np.newaxis], 3, -1)

X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

X_test_minus_twentyone = np.repeat(X_test_minus_twentyone[..., np.newaxis], 3, -1)

print("\nRepeat the greyscale layer three times so we immitate an RGB image")

print(X_train.shape, X_test.shape, X_test_minus_twentyone.shape)
# Create and compile the base resnet model using imagenet weights

resnet = tf.keras.applications.ResNet50(

    include_top=False, weights=None, input_tensor=tf.keras.Input(shape=(8,8,3)), 

    input_shape=None, pooling="max")





inputs = tf.keras.Input(shape=(8,8,3))

resnet_layer = resnet(inputs)

x = tf.keras.layers.Dense(256, activation="relu")(resnet_layer)

x = tf.keras.layers.Dense(64, activation="relu")(x)

outputs = tf.keras.layers.Dense(2, activation="softmax")(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='SGD', loss="SparseCategoricalCrossentropy", metrics="accuracy")



model.summary()



model.fit(x=X_train, y=y_train.astype("bool"), epochs=50, batch_size=256)



loss, accuracy = model.evaluate(X_test,y_test, batch_size=256)

print("Accuracy Test+", accuracy)



loss, accuracy = model.evaluate(X_test_minus_twentyone, y_test_minus_twentyone, batch_size=256)

print("Accuracy Test-21", accuracy)



tf.keras.utils.plot_model(model)
# Test+

preds = model.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["ResNet-Test+"] = accuracy

recall_dict["ResNet-Test+"] = recall

precision_dict["ResNet-Test+"] = precision

f1_dict["ResNet-Test+"] = f1



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



# Test-21

preds = model.predict(X_test_minus_twentyone) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test_minus_twentyone, pred_labels), 3)

recall = np.round(recall_score(y_test_minus_twentyone, pred_labels), 3)

precision = np.round(average_precision_score(y_test_minus_twentyone, pred_labels), 3)

f1 = np.round(f1_score(y_test_minus_twentyone, pred_labels), 3)

print(f"\nTest+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")



accuracy_dict["ResNet-Test-21"] = accuracy

recall_dict["ResNet-Test-21"] = recall

precision_dict["ResNet-Test-21"] = precision

f1_dict["ResNet-Test-21"] = f1
import seaborn as sns



# Plot accuracies

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)}.values()),

           palette="viridis")

plt.xticks(rotation=45)

plt.title("NSL-KDD")

plt.ylabel("Accuracy")

plt.xlabel("Model")

plt.show()



# Plot Precision

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(precision_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(precision_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="cool_r")

plt.xticks(rotation=45)

plt.title("NSL-KDD")

plt.ylabel("Precision")

plt.xlabel("Model")

plt.show()



# Plot Recall

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(recall_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(recall_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="magma")

plt.xticks(rotation=45)

plt.title("NSL-KDD")

plt.ylabel("Recall")

plt.xlabel("Model")

plt.show()



# Plot F1

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(f1_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(f1_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="mako")

plt.xticks(rotation=45)

plt.title("NSL-KDD")

plt.ylabel("F1")

plt.xlabel("Model")

plt.show()