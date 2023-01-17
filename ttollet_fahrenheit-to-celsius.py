import keras # For our neural network



import numpy as np # To create the data



from sklearn.model_selection import train_test_split # For dividing our data

from sklearn.metrics import mean_squared_error # For evaluating our models



import matplotlib.pyplot as plt # To briefly vizualise some key points
l0 = keras.layers.Dense(units=1, input_shape=[1])



model = keras.models.Sequential([l0])



model.compile(loss='mean_squared_error', optimizer='Adam')
def to_celsius(f):

    # Applies the known equation for converting fahrenheit to celsius

    return ((f - 32) / 1.8)



# Create list of temperatures from 0 to 199 fahrenheit, and corresponding values in degrees celsius

X = np.arange(200)

y = []



for x in X:

    y.append(to_celsius(x))

    

y = np.array(y)



# Split the data in training and testing sets, for later evaluation of our model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
history = model.fit(X_train, y_train, epochs=5000, verbose=False) # Runs for a couple minutes
pred = model.predict(X_test)

pred_head = list(map((lambda x: x[0]), pred[:5].tolist()))



print('RMSE, Root mean squared error:', np.sqrt(mean_squared_error(y_test, pred)))



print('\nPrediction: {}\nActual: {}'.format(

    pred_head,

    y_test[:5])

)
plt.xlabel('Actual value in Celsius')

plt.ylabel('Model`s value in Celsius')

plt.plot(y_test[:5], pred_head)

plt.show()
plt.xlabel('Fahrenheit')

plt.ylabel('Celsius')

plt.plot(X,y)

plt.show()
print('Weights: {}'.format(l0.get_weights()))
print(10 * 0.55554414 + (-17.775848))



print(10 * l0.get_weights()[0] + l0.get_weights()[1])



print(model.predict(np.array([10])))
from sklearn.linear_model import LinearRegression
# Reshape data as required

lm_X_train = X_train.reshape(-1, 1)

lm_y_train = y_train.reshape(-1,1)

lm_X_test = X_test.reshape(-1,1)

lm_y_test = y_test.reshape(-1,1)
# Create model & predictions

linear_model = LinearRegression()

linear_model.fit(lm_X_train, lm_y_train) # Runs almost instantly

lm_pred = linear_model.predict(lm_X_test)
# Evaluate & compare



from decimal import Decimal



keras_rmse = '%.2E' % Decimal(np.sqrt(mean_squared_error(y_test, pred)))

lm_rmse = '%.2E' % Decimal(np.sqrt(mean_squared_error(y_test, lm_pred)))



print('Linear Model RMSE: ', lm_rmse)

print('Keras RMSE: ', keras_rmse)