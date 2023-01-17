import pandas as pd

train_dataform = pd.read_csv(
    '/kaggle/input/digit-recognizer/train.csv',
)

# 試しに中身を見てみる
train_dataform.head()
test_dataform = pd.read_csv(
    '/kaggle/input/digit-recognizer/test.csv',
)

# 中身を見てみる
test_dataform.head()
# 訓練データから正解ラベルを落とす
train_input_dataform = train_dataform.drop(['label'], axis=1)
train_input_dataform.head()
# 訓練データのラベルだけを取ってきて正解データを作成する
train_output = train_dataform.label
train_output.head()
from tensorflow.keras.utils import to_categorical

# 正解を数字から one-hot 表現に変換
train_output_onehot = to_categorical(train_output.values, num_classes=10)
train_output_onehot
# 画像を表示してみる
%matplotlib inline
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 15))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(train_output[i]))
    # 28x28 に reshape してやる必要がある
    ax.imshow(train_input_dataform.values[i].reshape(28, 28), cmap='gray')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam

model = Sequential()

# シンプルに全結合レイヤーいくつか
model.add(Input(shape=(784, )))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation="softmax"))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

model.fit(
    train_input_dataform.values, 
    train_output_onehot,
    batch_size=2000, 
    epochs=5, 
    verbose=1,
)
from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# 訓練データで評価してみる
model.evaluate(train_input_dataform.values, train_output_onehot)
# 適当に訓練データでpredictして中身を見てみる
sample_results = model.predict(train_input_dataform, batch_size=10000)
sample_results
import numpy as np

# 予測
results = model.predict(test_dataform, batch_size=10000)

# np.array に変換
results_nparray = np.array([np.argmax(result) for result in results])
results_nparray
submit_dataframe = pd.DataFrame({
    'ImageId': pd.array(range(1, 28001)), # ImageId は1始まりらしい
    'Label': results_nparray
})
submit_dataframe.head()
submit_dataframe.to_csv('submission.csv', index=False)