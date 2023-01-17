# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
anime_dict = defaultdict(dict)

columns = []

with open('anime_slash3.txt', 'rU') as csvfile:

        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:

            data_list = row[0].split('///')

            if data_list[0] == 'anime_id':

                columns = data_list[1:]

            if data_list[0] != 'anime_id':

                anime_dict[data_list[0]] = {}

                data = data_list[1:]

                for x in range(len(data)):

                    anime_dict[data_list[0]][columns[x]] = data[x]