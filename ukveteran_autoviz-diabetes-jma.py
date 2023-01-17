!pip install autoviz
import numpy as np
import pandas as pd 
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/diabetes-data-set/diabetes.csv')
df.head()
df = AV.AutoViz(filename="",sep=',', depVar='Glucose', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=5)
df = AV.AutoViz(filename="",sep=',', depVar='Insulin', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=5)
df = AV.AutoViz(filename="",sep=',', depVar='BMI', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=5)
df = AV.AutoViz(filename="",sep=',', depVar='BloodPressure', dfte=df, header=0, verbose=2, 
                 lowess=False, chart_format='svg', max_rows_analyzed=150000, max_cols_analyzed=5)