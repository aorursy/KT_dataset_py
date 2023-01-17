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
#kappaを求めるモジュールをインポート

from sklearn.metrics import confusion_matrix, cohen_kappa_score

#データを読み込む

import pandas as pd

df = pd.read_csv('/kaggle/input/trialcsv2/kappa37_fixed2.csv')

#いらない行を削ぎ落とす

df = df.loc[:,['Feature_sh', '抽出済']]

#漢字のカラム名を英語に変える

df = df.rename(columns={'抽出済':'my_feature'})
#一旦見てみる

df
#データを置換する

df.loc[df['Feature_sh'] != 4, 'Feature_sh'] = 0

df.loc[df['Feature_sh'] == 4, 'Feature_sh'] = 1
#データを置換する

df.loc[df['my_feature'] == 0, 'my_feature'] = 20000

df.loc[df['my_feature'] != 20000, 'my_feature'] = 0

df.loc[df['my_feature'] == 20000, 'my_feature'] = 1
a = df['my_feature']

b = df['Feature_sh']

cohen_kappa_score(a, b)
model = sklearn.method(x_train, y_train)

y_test = model.predict(x_test)