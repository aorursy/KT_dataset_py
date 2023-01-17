from keras import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("../input/train_V2.csv")
# use only 30% of the total data to reduce the size

features = ["boosts", "killPlace", "kills", "vehicleDestroys", "walkDistance"]

train_feature = train[features]

train_feature.describe(include="all")
x = train_feature.values
y = train["winPlacePerc"].values
print(x[:5])
print(y[:5])
# split into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y)
def build_model():
    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(5, ), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer=RMSprop(clipnorm=1.))
    
    return model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("estimator", KerasRegressor(build_fn=build_model, epochs=3, batch_size=128))
])
# cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=3)
pipeline.fit(x_train, y_train)
# load test data
test = pd.read_csv("../input/test_V2.csv")
test_features = test[features]
result = pipeline.predict(test_features)
tmp_data = {
    "Id": test["Id"],
    "winPlacePerc": result
}

pd_result = pd.DataFrame(tmp_data)
pd_result.to_csv("sample_submission.csv", index=False)
