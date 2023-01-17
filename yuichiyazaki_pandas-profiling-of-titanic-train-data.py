import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling as pdp

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
pdp.ProfileReport(df)
profile = pdp.ProfileReport(df)
profile.to_file(outputfile="outputfile.html")