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
import numpy as np

import pandas as pd

import cv2
a=np.zeros((1,200,200,3))

b=np.zeros((1,1,388))

for i in range(1,2330):

    path = '/kaggle/input/annotation/annotation/' + str(i) +'.txt'

    df = pd.read_table(path, delim_whitespace=True, names=('A', 'B','C'))

    

        

    img_name= df.iloc[0,0]

    img_path = '/kaggle/input/helen_1/helen_1/' + img_name +'.jpg'

    img=cv2.imread(img_path)

    shape1 = img.shape

    img = cv2.resize(img,(200,200))

    img = np.reshape(img,[1,200,200,3])







    x_coordinate=df.iloc[1:,0]

    x_coordinate = np.asarray(list(map(float, x_coordinate))).reshape(194,1)

    x_coordinate = (x_coordinate*200)/(shape1[1])

    

    y_coordinate = df.iloc[1:,2]

    y_coordinate = np.array(y_coordinate).reshape(194,1)

    y_coordinate = (y_coordinate*200)/(shape1[0])

       



    out=np.concatenate((x_coordinate,y_coordinate))

    out=(out.T)

    out=np.reshape(out,[1,1,388])



    a=np.concatenate([a,img],axis=0)

    b=np.concatenate([b,out],axis=0)

    print(i,end=" ")





a.shape
np.save('img',a)
np.save('out',b)