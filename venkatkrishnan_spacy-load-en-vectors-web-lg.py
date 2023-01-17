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
        
import spacy

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!python -m spacy download en_vectors_web_lg
!python -m spacy link en_vectors_web_lg en_vectors_web_lg_link
nlp = spacy.load('en_vectors_web_lg_link')
doc = nlp(u'My Name is Venkat Krishnan')
for token1 in doc:
    for token2 in doc:
        print(f'Token 1 {token1.text} and Token 2 {token2.text} Similarity score : {token1.similarity(token2)}')
