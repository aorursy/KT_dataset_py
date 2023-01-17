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
Rectangulos = [6,7,10,5,9]
Rectangulos
Triangulos = [1, 4, 9, 16, 25,4,7]
Triangulos [2] 
Triangulos = [1, 4, 9, 16, 25,4,7]
Triangulos [-2]
Triangulos = [1, 4, 9, 16, 25,4,7]
Triangulos [-2:]
Triangulos = [1, 4, 9, 16, 25,4,7]
Triangulos [:]
Triangulos = [1, 4, 9, 16, 25,4,7]
Triangulos + [36, 49, 64, 81, 100]

4**3

Rectangulos = [1, 8, 27, 65, 125]

Rectangulos [3] = 64 
Rectangulos
Rectangulos = [1, 8, 27, 65, 125]
Rectangulos.append(216) 
Rectangulos.append(5 ** 2) 
Rectangulos
letras = ['Q', 'D', 'P', 'K', 'Ñ', 'L', 'Y']
letras
letras = ['P', 'X', 'B', 'M', 'E', 'R', 'A']
letras[2:5] = ['C', 'D', 'E']
letras
letras[2:5] = []
letras
letras = ['P', 'X', 'B', 'M', 'E', 'R', 'A']

letras[2:5] = ['C', 'D', 'E']

letras[2:5] = []
letras[:] = []
letras
letras = ['P', 'X', 'B', 'M', 'E', 'R', 'A']
len(letras)
letras = ['P', 'X', 'B', 'M', 'E', 'R', 'A']
n = [1, 2, 3]
x

letras = ['P', 'X', 'B', 'M', 'E', 'R', 'A']
n = [1, 2, 3]
x[0]
x[0][1]
