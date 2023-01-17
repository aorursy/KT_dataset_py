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
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv') 

missing_value = train_data.isnull().sum() # 欠損値調査

print(missing_value)

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')  # テストデータ読み込み

test_data.isnull().sum() # 欠損値確認

print(train_data)
# 訓練データ・学習データの欠損値を'null'に置換する

for data in [train_data, test_data]:

    for col in ['keyword', 'location']:

        data[col] = data[col].fillna('null')

        data[col] = data[col].str.replace('%20',' ')

        #print(data[col])

train_data.isnull().sum() # 欠損値調査

test_data.isnull().sum() # 欠損値確認
import re # 正規表現による置換用

excep = ['#', '@'] # 除外ワード

train_label = [] # 学習に用いるラベル保存用

train_text = [] # 学習に用いるテキストデータ

train_text_SCDV = [] # SCDVに用いるデータ

# 5:みやすい置き換えワード



for count,text in enumerate(train_data['text']):

    text_new = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "URL", text) # URL潰し

    text_new = text_new.lower() # 大文字を小文字に

    text_new = text_new.replace('\n',' ') # 改行置き換え

    text_new_list = [] # テキストぶつ切り保管庫 

    for text_no,text_parts in enumerate(text_new.split(' ')):

        #print(text_parts)

        text_parts = re.sub(r'[^a-z0-9]', "", text_parts) # 記号消去

        text_parts = re.sub(r'([a-z])\1+', '\\1\\1', text_parts) # 2文字以上の文字重複を2文字やった

        #m = re.match(r'([a-z])\1',text_parts)

        text_parts = re.sub(r'[0-9].*[0-9]', "5", text_parts) # 数字を全てみやすいワードに置き換え

        text_parts = re.sub(r'[0-9](^[0-9])*', "", text_parts) # 5でないもの（単語に密着しているもの）を削除

        text_parts = text_parts.replace(".","") # ピリオドを消去

        if text_parts == '': # 無を獲得

            continue

        if text_no == 0 and text_parts == 'rt': # リツイート削除

            continue

        if text_parts[0] in excep: # トレンドやメンションがあるなら

            text_parts = text_parts[0] # それを記号のみにする（トレンド・メンションの有無が真偽に関わるかも）

        if (text_parts[0] != '&' or len(text_parts) <= 1): # HTTPタグを削除

            text_new_list.append(text_parts)

    text_new = ' '.join(text_new_list) # 結合

    train_text_SCDV.append(text_new_list)

    train_text.append(text_new) # 学習用にテキストを捕獲

    train_label.append(train_data['target'][count]) # ラベル保存



#print(train_data['text'])

#print(train_text)

    #print('ID:' + str(train_data['id'][count]) + ' 元テキスト:' + text)

    #print('ID:' + str(train_data['id'][count]) + ' テキスト:' + text_new)
from sklearn.svm import LinearSVC # サポートベクタークラシフィケーション データ量10kかつclass分類のため

from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer



model = make_pipeline(TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english'), LinearSVC()) # データ前処理とモデル生成のパイプライン

# tf-idfのオプション（0.9以上の頻出語は消去・2回以上使用されていない単語は除去）

# scikit-learnの公式から問題があるというfactあり



model.fit(train_text, train_data['target']) # textとtargetで学習実行



print('Train accuracy = %.3f' % model.score(train_text, train_data['target'])) # 学習精度たしかめー
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')  # テストデータ読み込み

test_data.isnull().sum() # 欠損値確認



test_text = [] # テストに用いるテキストデータ

# 前処理開始（正直前の処理と同じなんで・・・）

for count,text in enumerate(test_data['text']):

    text_new = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "URL", text) # URL潰し

    text_new = text_new.lower() # 大文字を小文字に

    text_new = text_new.replace('\n',' ') # 改行置き換え

    text_new_list = [] # テキストぶつ切り保管庫 

    for text_no,text_parts in enumerate(text_new.split(' ')):

        #print(text_parts)

        text_parts = re.sub(r'[^a-z0-9]', "", text_parts) # 記号消去

        text_parts = re.sub(r'([a-z])\1+', '\\1\\1', text_parts) # 2文字以上の文字重複を2文字やった

        #m = re.match(r'([a-z])\1',text_parts)

        text_parts = re.sub(r'[0-9].*[0-9]', "5", text_parts) # 数字を全てみやすいワードに置き換え

        text_parts = re.sub(r'[0-9](^[0-9])*', "", text_parts) # 5でないもの（単語に密着しているもの）を削除

        text_parts = text_parts.replace(".","") # ピリオドを消去

        if text_parts == '': # 無を獲得

            continue

        if text_no == 0 and text_parts == 'rt': # リツイート削除

            continue

        if text_parts[0] in excep: # トレンドやメンションがあるなら

            text_parts = text_parts[0] # それを記号のみにする（トレンド・メンションの有無が真偽に関わるかも）

        if (text_parts[0] != '&' or len(text_parts) <= 1): # HTTPタグを削除

            text_new_list.append(text_parts)

    

    text_new = ' '.join(text_new_list) # 結合

    test_text.append(text_new) # 学習用にテキストを捕獲



test_predicted = model.predict(test_text) # テスト開始

# チェックするならこの下を実行

#for check in test_predicted:

#    print(check)
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv') 

sub['target'] = list(map(int, test_predicted))

sub.to_csv('submission.csv', index=False)