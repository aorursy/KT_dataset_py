# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
a = 5
print(a)
a = "alguma coisa"
print (a)
#para inserir o comentario em python devo usar jogo da velha (#)
# o (:) é para começar a execução e finalizar a execução
if a == 5:
    print("vale 5")
else:
    print("vale alguma coisa")