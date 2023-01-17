# Here's an example of loading the CSV using Panda's built-in HDF5 support:

import pandas as pd



with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")