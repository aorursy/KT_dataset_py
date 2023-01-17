%load_ext autoreload

%autoreload 2
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



from sklearn.metrics import mean_squared_error

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam



from preprocess import DataPreprocessModule
data_preprocess_module = DataPreprocessModule(

    train_path='../input/hdb-resale-price-prediction/train.csv',

    test_path='../input/hdb-resale-price-prediction/test.csv')

X_train, X_val, X_test, y_train, y_val = data_preprocess_module.get_preprocessed_data()

print('Shape of X_train:', X_train.shape)

print('Shape of X_val:', X_val.shape)

print('Shape of X_test:', X_test.shape)

print('Shape of y_train:', y_train.shape)

print('Shape of y_val:', y_val.shape)
test_indices = X_test.index

preprocesser = data_preprocess_module.get_preprocessor()

X_train = preprocesser.fit_transform(X_train)

X_val = preprocesser.transform(X_val)

X_test = preprocesser.transform(X_test)
# Define RMSE

metric = lambda y1_real, y2_real: np.sqrt(mean_squared_error(y1_real, y2_real))

# Claculate exp(y) - 1 for all elements in y

y_trfm = lambda y: np.expm1(y)

# Define function to get score given model and data

def get_score(model, X, y):

    # Predict

    preds = model.predict(X)

    # Transform

    preds = y_trfm(preds)

    y = y_trfm(y)

    return metric(preds, y)
def build_MLP(layer_sizes, input_shape, loss, learning_rate=0.001, dropout=0):

    inp = Input(input_shape)

    x = inp

    for layer_size in layer_sizes:

        x = Dense(layer_size, activation='relu')(x)

        if dropout:

            x = Dropout(dropout)(x)

    out = Dense(1)(x)

    model = Model(inp, out)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)

    return model
# model1 = build_MLP(layer_sizes=[10, 10], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model1.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model1, X_val, y_val)
# model2 = build_MLP(layer_sizes=[50, 50], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model2.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model2, X_val, y_val)
# model3 = build_MLP(layer_sizes=[10, 10, 10], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model3.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model3, X_val, y_val)
# model4 = build_MLP(layer_sizes=[50, 50, 50], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model4.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model4, X_val, y_val)
# model5 = build_MLP(layer_sizes=[100, 100], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model5.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model5, X_val, y_val)
# model6 = build_MLP(layer_sizes=[100, 100, 100], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.3)

# model6.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model6, X_val, y_val)
# model7 = build_MLP(layer_sizes=[50, 50, 50], input_shape=(X_train.shape[1],),

#                    loss='mean_squared_error', learning_rate=1e-4, dropout=0.5)

# model7.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)

# get_score(model7, X_val, y_val)
model8 = build_MLP(layer_sizes=[50, 50, 50], input_shape=(X_train.shape[1],),

                   loss='mean_squared_error', learning_rate=1e-4, dropout=0.5)

history = model8.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val),

                     verbose=0)
plt.plot(history.epoch, history.history['loss'], label='Training Loss')

plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='best')

plt.savefig('loss_curves.png')
num_epochs = int(np.argmin(history.history['val_loss']) + 1)

print('Epoch with lowest valdiation loss:' , num_epochs)
model9 = build_MLP(layer_sizes=[50, 50, 50], input_shape=(X_train.shape[1],),

                   loss='mean_squared_error', learning_rate=1e-4, dropout=0.5)

model9.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val),

                     verbose=0)

get_score(model9, X_val, y_val)
preds_test = model9.predict(X_test).flatten()

preds_test = y_trfm(preds_test)



output = pd.DataFrame({'id': test_indices,

                       'resale_price': preds_test})

output.to_csv('submission.csv', index=False)