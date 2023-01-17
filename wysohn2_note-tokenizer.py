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
from tokenizer import load_or_build, translate, retranslate, TOKEN_SOS, TOKEN_EOS



text_tokenizer = load_or_build('/kaggle/input/naver-blog-results/naver_posts_refined.csv')
test_str = """옆쪽에는 가위집이 있어 한번에 쭈욱~ 뜯어 주면 되는 편리함이 있어 좋은데요. 개인적으로 이런 편리함 칭찬해!"""



translated = translate(test_str)

print(translated)

print(retranslate(translated))
tokens = text_tokenizer.encode(translate(test_str))

print(tokens)

trans_str = text_tokenizer.decode(tokens)

print(trans_str)

print(retranslate(trans_str))