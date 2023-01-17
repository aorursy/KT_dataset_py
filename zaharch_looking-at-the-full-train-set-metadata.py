import numpy as np

import pandas as pd



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
data = pd.read_csv('/kaggle/input/train-set-metadata-for-dfdc/metadata', low_memory=False)
(data['pxl.hash'] == data['pxl.hash.orig']).value_counts()
data['md5'].value_counts().value_counts().head()
data['wav.hash'].value_counts().value_counts().head()
data['pxl.hash'].value_counts().value_counts().head()
data.head()
data.shape
data.label.value_counts()
data.split.value_counts()
set(data.original) - set(data.filename)
set(data.loc[data.original == 'NAN', 'filename']) - set(data.original)
data.loc[data.original != 'NAN', 'original'].value_counts().hist(bins=40)
data.loc[data.original != 'NAN', 'original'].value_counts().value_counts().head()
for col in data.columns:

    print(pd.crosstab(data[col],data['label']))
pd.crosstab(data['video.@display_aspect_ratio'],data['label'])
pd.crosstab([data['video.@display_aspect_ratio'], data['audio.@codec_time_base']],data['label'])