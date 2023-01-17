# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
preds_age = np.load('../input/trends-h2o-automl-age/preds_age.npy')

preds_domain1_var1 = np.load('../input/trends-h2o-automl-domain1-var1/preds_domain1_var1.npy')

preds_domain1_var2 = np.load('../input/trends-h2o-automl-domain1-var2/preds_domain1_var2.npy')

preds_domain2_var1 = np.load('../input/trends-h2o-automl-domain2-var1/preds_domain2_var1.npy')

preds_domain2_var2 = np.load('../input/trends-h2o-automl-domain2-var2/preds_domain2_var2.npy')
test = pd.read_csv('../input/trends-train-test-creator/test.csv')

test.head()
preds_age
Id = test.Id.values
Id.shape
sub_df = pd.DataFrame({'Id': Id, 'age': preds_age.flatten(), 

                       'domain1_var1': preds_domain1_var1.flatten(), 'domain1_var2': preds_domain1_var2.flatten(), 

                       'domain2_var1': preds_domain2_var1.flatten(), 'domain2_var2': preds_domain2_var2.flatten()})

sub_df.head()
sub_df = pd.melt(sub_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

sub_df.head()
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")



sub_df.head()
sub_df.to_csv("submission_h2o_automl.csv", index=False)
