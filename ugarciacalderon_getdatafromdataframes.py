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
def getData():

    '''

    Este método crea un dataframe a partir de dos dataframes.

    Obtiene los datos de dataframe2 a partir de la columna 1 del dataframe1, remplazando la primer y ultima

    última columna del dataframe2

    return: dataframe3

    '''

    # carga dataframe1

    data1 = pd.read_csv('../input/dataframe1.csv')

    dataframe1 = pd.DataFrame(data1)

    print(dataframe1)

    

    # carga dataframe2

    data2 = pd.read_csv('../input/dataframe2.csv')

    dataframe2 = pd.DataFrame(data2)

    print(dataframe2)

    

    #crear dataframe3

    dataframe3 = pd.DataFrame(dataframe1,dataframe2)
if __name__ == '__main__':

    getData()