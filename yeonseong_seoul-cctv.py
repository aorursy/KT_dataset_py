# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CCTV_Seoul = pd.read_csv('../input/01. CCTV_in_Seoul.csv', encoding = 'utf-8')

CCTV_Seoul.head()
#column 출력

CCTV_Seoul.columns
#column 이름바꾸기

CCTV_Seoul.rename(columns={CCTV_Seoul.columns[0] : '구별'},inplace = True)

CCTV_Seoul.head()
pop_Seoul = pd.read_excel('../input/01. population_in_Seoul.xls',  encoding='utf-8')

pop_Seoul.head()
pop_Seoul = pd.read_excel('../input/01. population_in_Seoul.xls',

                          header = 2,

                          parse_col = 'B, D, G, J, N',

                          encoding='utf-8')

pop_Seoul.head()
pop_Seoul.rename(columns={pop_Seoul.columns[0] : '구별', 

                          pop_Seoul.columns[1] : '인구수', 

                          pop_Seoul.columns[2] : '한국인', 

                          pop_Seoul.columns[3] : '외국인', 

                          pop_Seoul.columns[4] : '고령자'}, inplace=True)

pop_Seoul.head()