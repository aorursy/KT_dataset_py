# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from scipy.io import loadmat



mat_contents = loadmat('/kaggle/input/1056lab-cardiac-arrhythmia-detection/test/B07957.mat')

mat_contents
mat_contents.keys()
mat_contents['val']


mat_contents['val'][0]
import matplotlib.pyplot as plt



y = mat_contents['val'][0]

length = len(y)

x = np.linspace(0, length, length)



plt.style.use('ggplot')

plt.figure()

plt.plot(x, y)

plt.show()
from glob import glob



idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

y_ = []

for d in ['normal', 'af']:

    for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/' + d +'/*.mat')):

        filename = path.split('/')[-1]  # e.g. B05821.mat

        i = filename.split('.')[0]  # e.g. B05821

        idx_.append(i)

        mat_contents = loadmat(path)

        x = mat_contents['val'][0]

        len_.append(len(x))

        mean_.append(x.mean())

        std_.append(x.std())

        ste_.append(x.std()/np.sqrt(len(x)))

        max_.append(x.max())

        min_.append(x.min())

        if d == 'normal':

            y_.append(0)

        else:

            y_.append(1)
train_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value', 'y'])

train_df['length'] = len_

train_df['mean'] = mean_

train_df['standard deviation'] = std_

train_df['standard error'] = ste_

train_df['maximum value'] = max_

train_df['minimum value'] = min_

train_df['y'] = y_

train_df
from glob import glob



idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/test/*.mat')):

    filename = path.split('/')[-1]  # e.g. B05821.mat

    i = filename.split('.')[0]  # e.g. B05821

    idx_.append(i)

    mat_contents = loadmat(path)

    x = mat_contents['val'][0]

    len_.append(len(x))

    mean_.append(x.mean())

    std_.append(x.std())

    ste_.append(x.std()/np.sqrt(len(x)))

    max_.append(x.max())

    min_.append(x.min())

test_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value'])

test_df['length'] = len_

test_df['mean'] = mean_

test_df['standard deviation'] = std_

test_df['standard error'] = ste_

test_df['maximum value'] = max_

test_df['minimum value'] = min_

test_df
import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')

plt.figure()

sns.pairplot(train_df)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



corr = train_df.corr()



plt.style.use('ggplot')

plt.figure()

sns.heatmap(corr, square=True, annot=True)

plt.show()
from sklearn.ensemble import RandomForestClassifier



X_train = train_df.drop('y', axis=1).to_numpy()

y_train = train_df['y'].to_numpy()



model = RandomForestClassifier()

model.fit(X_train, y_train)
p_train = model.predict_proba(X_train)

p_train
y_train
X_test = test_df.to_numpy()



p_test = model.predict_proba(X_test)

p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-cardiac-arrhythmia-detection/sampleSubmission.csv', index_col=0)

submit_df['af'] = p_test[:,1]

submit_df
submit_df.to_csv('submission.csv')