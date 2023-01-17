# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborne is a package built on top of matplotlib.
sns.set() # activate seaborn to override all the matplotlib graphics

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
raw_df = pd.read_csv("../input/star-categorization-giants-and-dwarfs/Star39552_balanced.csv")
raw_df
df = raw_df[['B-V', 'Amag', 'TargetClass']]
df
df.describe(include='all')
# Select Target
y = df['TargetClass']

# Select Features
x = df[['B-V','Amag']]
star_predictor = LogisticRegression()
star_predictor.fit(x, y)
print('the score is') 
print(star_predictor.score(x, y))
#Make Prediction

feature = [[1.130,15.792525]]

star_predictor.predict(feature)  # Target Class is 0 -> Dwarf
#Make Prediction

feature = [[0.227,17.159748]]

star_predictor.predict(feature)  # Target Class is 1 -> Giant