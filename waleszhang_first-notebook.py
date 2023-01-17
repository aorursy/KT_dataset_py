import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

Iris.head()
sns.distplot(a=Iris['PetalLengthCm'], kde=False)