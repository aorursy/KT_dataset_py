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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
# usr embedding
# 嵌入层就是一个查找表，它从整数索引（代表特定的单词）映射到密集向量。嵌入的维数是可以手动指定的，
# 就像指定多少层一样

# 创建embedding层时，权重会随机初始化，训练过程中会通过反向传播逐渐调整他们
embedding_layer = layers.Embedding(1000, 5)  # 该层只能用作模型中的第一层, 输出为5
# 如果将整数传递给嵌入层，则结果将使用嵌入表中的向量替换每个整数：
result = embedding_layer(tf.constant([1,2,3]))
result
# 对于文本或序列问题，Embedding层采用2D整数张量
# 注意传入的值及维度，输出的值及维度
tensor = tf.constant([[0,1,2],[3,4,5]])
print(tensor)
result = embedding_layer(tensor)
print(result)
# 从零开始学习嵌入
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)
# 词汇表中的“ _”代表空格
encoder = info.features['text'].encoder
encoder.subwords[:20]
# 电影评论的长度可以不同。我们将使用该padded_batch方法来规范评论的长度
train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)
# 请注意尾随零，因为该批次被填充到最长的示例中
train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()
embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)
import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()
# 检索学习的嵌入
# 接下来，让我们检索在训练中学习到的词嵌入。这将是一个形状矩阵(vocab_size, embedding-dimension)。

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
# 现在，我们将权重写入磁盘。要使用Embedding Projector ，
# 我们将以制表符分隔的格式上传两个文件：一个向量文件（包含嵌入）和一个元数据文件（包含单词）

import io

encoder = info.features['text'].encoder

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
  vec = weights[num+1] # skip 0, it's padding.
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()


