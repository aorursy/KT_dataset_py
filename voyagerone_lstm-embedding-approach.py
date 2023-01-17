# Fehim Altınışık
# fehim.altinisik@gmail.com
# 160201010
import os
import sys
import math

from operator import attrgetter

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

print(device_lib.list_local_devices())
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
sample_submissions = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
print(item_categories.shape)
print(items.shape)
print(sales_train.shape)
print(sample_submissions.shape)
print(shops.shape)
print(test.shape)
with pd.option_context('display.max_rows', 5, 'display.max_columns', 185):
    display(sales_train.sample(5))
print(test["shop_id"].nunique())
print(test["item_id"].nunique())
print(len(test["shop_id"].unique()))
print(len(sales_train["shop_id"].unique()))
print(len(set(test["shop_id"]).intersection(set(sales_train["shop_id"]))))

print(len(test["item_id"].unique()))
print(len(sales_train["item_id"].unique()))
print(len(set(test["item_id"]).intersection(set(sales_train["item_id"]))))
sales_train = sales_train.assign(
    date_to_datetime=pd.to_datetime(sales_train["date"])
)
sales_train = sales_train.assign(
    date_to_datetime_year=sales_train["date_to_datetime"].dt.year,
    date_to_datetime_month=sales_train["date_to_datetime"].dt.month,
    date_to_datetime_day=sales_train["date_to_datetime"].dt.day
)
sales_train = sales_train.assign(
    item_cnt_day_as_int=sales_train["item_cnt_day"].astype(np.int32)
)
sales_train = sales_train.merge(items.loc[:, ["item_id", "item_category_id"]], on="item_id")
sales_train.head()
print(sales_train["item_category_id"].nunique())
def shop_feature(shop_data):
    features = {}
    
    first_record = shop_data["date_to_datetime"].min()
    last_record = shop_data["date_to_datetime"].max()
    
    features["first_record"] = first_record
    features["last_record"] = last_record
    
    lifetime = last_record - first_record
    features["shop_lifetime"] = int(lifetime.days / 30)
    
    return pd.Series(features, index=["first_record", "last_record", "shop_lifetime"])

shop_features = sales_train.groupby("shop_id").apply(shop_feature)
print(shop_features.head())
with pd.option_context('display.max_rows', 10, 'display.max_columns', 185):
    display(sales_train.set_index(["shop_id", "item_category_id"]).sort_values("date_to_datetime").head())
block_based_grouping = pd.pivot_table(sales_train, index=["shop_id", "item_category_id"], columns="date_block_num", values="item_cnt_day", aggfunc="mean")
print(block_based_grouping.shape)
block_based_grouping = block_based_grouping.fillna(method='ffill', axis="columns")
block_based_grouping = block_based_grouping.fillna(0)
with pd.option_context('display.max_rows', 10, 'display.max_columns', 185):
    display(block_based_grouping.sample(10))
    # display(testgroup.head())
def generate_training_series(data, size_of_sets):
        
    # index =  pd.Index((data.name, ) * 5)
    index =  np.array((data.name, ) * 7)
    index_extension =  np.arange(
                (
                    math.floor(data.shape[0] / size_of_sets) + 1
                )
        )
    
    new_index = np.column_stack((index, index_extension))
    
    # print(index)
    # print(index_extension)
    # print(new_index)
    
    # print(np.transpose(new_index))# new_index = pd.MultiIndex(new_index)
    new_index = pd.MultiIndex.from_arrays(np.transpose(new_index), names=('shop_id', 'item_id', "period_id"))
    # print(new_index)
    
    samples = pd.DataFrame(np.nan, index=new_index, columns=np.arange(size_of_sets).tolist() + ["y"])
    
    # print(samples)
    
    cursor = 0
    counter = 0
    
    while cursor < data.shape[0]:
        
        if cursor + size_of_sets > data.shape[0]:
            # print("x: {}:{}, y: {}".format(data.shape[0] - size_of_sets - 2, data.shape[0] - 2, data.shape[0] - 1))
            samples.loc[samples.index.get_level_values('period_id') == counter, samples.columns[0: size_of_sets]] = data.loc[data.shape[0] - size_of_sets - 1: data.shape[0] - 2].values
            samples.loc[samples.index.get_level_values('period_id') == counter, samples.columns[size_of_sets]] = data.loc[data.shape[0] - 1]
            break
            
        # print(samples.loc[samples.index.get_level_values('period_id') == counter, samples.columns[0: size_of_sets]])
        # print(data.loc[cursor: cursor + size_of_sets - 1])
        
        samples.loc[samples.index.get_level_values('period_id') == counter, samples.columns[0: size_of_sets]] = data.loc[cursor: cursor + size_of_sets - 1].values
        samples.loc[samples.index.get_level_values('period_id') == counter, samples.columns[size_of_sets]] = data.loc[cursor + size_of_sets]
            
        # print("x: {}:{}, y: {}".format(cursor, cursor + size_of_sets, cursor + size_of_sets + 1))
        
        cursor += size_of_sets
        counter += 1
    
    return samples
    
training_set = pd.concat({i: generate_training_series(row, 5) for i, row in block_based_grouping.iterrows()})
training_set= training_set.reset_index((0, 1), drop=True)
normalizer = Normalizer().fit(training_set[training_set.columns[0:5]])
training_set_normalized = normalizer.transform(training_set[training_set.columns[0:5]])
training_set_multi_input = np.concatenate((training_set_normalized, training_set.index.get_level_values(0).values.reshape(-1, 1), training_set.index.get_level_values(1).values.reshape(-1, 1)), axis=1)
x_train, x_test, y_train, y_test = train_test_split(training_set_multi_input, training_set[training_set.columns[5]], test_size=0.33, random_state=42)
embedding_size = len(sales_train["shop_id"].unique()) * len(sales_train["item_id"].unique())

lstm_input = keras.Input(shape=(5, 1), name="sales")
memory = layers.LSTM(120, input_shape=(5, 1))
x_memory = memory(lstm_input)
# memory_outputs = layers.Dense(1)(x_memory)

embedding_input= keras.Input(shape=(None,), name="shop_item")
embedding_features = layers.Embedding(embedding_size, 128)(embedding_input)
embedding_features = layers.LSTM(256)(embedding_features)

concat_x = layers.concatenate([embedding_features, x_memory])

outputs = layers.Dense(1, name="sale_count")(concat_x)

model = keras.Model(inputs=[lstm_input, embedding_input], outputs=outputs)

model.summary()
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
model.compile(
    loss=keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    optimizer=keras.optimizers.Adam(lr=0.005),
    metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],
)

history = model.fit(
    {"sales": np.expand_dims(x_train[:, 0:5], 2), "shop_item": (x_train[:, 5:7])},
    {"sale_count": y_train},
    batch_size=8,
    epochs=3,
    validation_split=0.2
)

test_scores = model.evaluate(
    {"sales": np.expand_dims(x_test[:, 0:5], 2), "shop_item": (x_test[:, 5:7])},
    y_test, verbose=2
)
print("Test loss:", test_scores[0])
print("Test mse:", test_scores[1])
# model.save("sales_forecast_model")
### Fetch Categories

test_cases = test.merge(items.loc[:, ["item_id", "item_category_id"]], on="item_id")
print(test_cases.shape)
print(test_cases.head())
test_cases_fetch_data = test_cases.merge(
    block_based_grouping.loc[:, block_based_grouping.columns[28:33]],
    left_on=["shop_id", "item_category_id"],
    right_index=True,
    how="left"
)
print(test_cases_fetch_data.shape)
with pd.option_context('display.max_rows', 10, 'display.max_columns', 185):
    display(test_cases_fetch_data.head(10))
print(test_cases[["shop_id", "item_category_id"]].duplicated().any())
test_cases_fetch_data = test_cases_fetch_data.fillna(method='ffill', axis="columns")
test_cases_fetch_data = test_cases_fetch_data.fillna(0)
prediction_block_data = normalizer.transform(test_cases_fetch_data[test_cases_fetch_data.columns[4:9]])
prediction_data = np.concatenate((prediction_block_data, test_cases_fetch_data[["shop_id", "item_category_id"]].values), axis=1)
print(prediction_data.shape)
test_scores = model.predict(
    {"sales": np.expand_dims(prediction_data[:, 0:5], 2), "shop_item": (prediction_data[:, 5:7])}
)
print(test_scores.shape)
submissions = pd.DataFrame(data=test_scores, index=test_cases_fetch_data["ID"].values, columns=["item_cnt_month"])
submissions.index.name = "ID"
print(submissions.head())
submissions.index = submissions.index.astype(np.int64)
# submissions.to_csv("submission.csv", index=True, index_label="ID", float_format="%.3f")