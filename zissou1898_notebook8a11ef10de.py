lm = Sequential([ Dense(1, input_shape=(3,1))])

lm.compile(optimizer=SGD(lr=0.1), loss='mse')
lm.evaluate(x, y, verbose=0)