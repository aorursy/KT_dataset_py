from collections import Counter

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_json('../input/train.json')
# train_df['ingredients'] = train_df['ingredients'].apply(np.array) # Convert list of ingredients to numpy arrays
train_df.head()
cusine_list = list(train_df['cuisine'].unique())
len(cusine_list)
train_df.index = train_df['cuisine']
ingredients = train_df['ingredients'].values.tolist()
flatten = lambda l: [item for sublist in l for item in sublist]
ingredients = flatten(ingredients)
ingredient_counts = Counter(ingredients).most_common(70)
df = pd.DataFrame.from_dict(ingredient_counts)
df.set_index(0, drop=True, inplace=True)
df.plot(kind='bar', figsize=(15, 5))
ingredient_counts = Counter(ingredients).most_common()[-50:]
df = pd.DataFrame.from_dict(ingredient_counts)
df.set_index(0, drop=True, inplace=True)
df.plot(kind='bar', figsize=(15, 5))
