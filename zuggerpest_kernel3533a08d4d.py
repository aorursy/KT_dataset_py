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

!pip install ipysankeywidget
from ipysankeywidget import SankeyWidget

from IPython.display import Image



links = [

    {'source': 'start', 'target': 'A', 'value': 2},

    {'source': 'A', 'target': 'B', 'value': 2},

    {'source': 'C', 'target': 'A', 'value': 2},

    {'source': 'A', 'target': 'C', 'value': 2},

]

SankeyWidget(links=links, margins=dict(top=0, bottom=0, left=50, right=100)).auto_save_png('sankey.png')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
