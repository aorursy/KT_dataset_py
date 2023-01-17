import pandas as pd

import matplotlib.pyplot as plt



#Overview of data

bite_df = pd.read_csv("../input/Health_AnimalBites.csv")

bite_df.describe().transpose()