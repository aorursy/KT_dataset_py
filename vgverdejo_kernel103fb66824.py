import pandas as pd

# Load your data
file_train = "/content/sample_data/train_set.csv"
file_test = "/content/sample_data/test_set.csv"

pdtrain = pd.read_csv(file_train, sep=',', error_bad_lines=False, encoding="latin-1")
pdtest = pd.read_csv(file_test, sep=',', error_bad_lines=False, encoding="latin-1")
# Here include any preprocessing, at least treat NaN positions

# Extract training and test data from dataframes
X_tr = pdtrain.iloc[:,2:].values
Y_tr = pdtrain.iloc[:,1].values

X_test = pdtest.iloc[:,1:].values
# Include your classification model

# Make your predictions over the test data

# Here, all predictions are set to 1
Y_test = np.ones((X_test.shape[0],))
# Generate submission file
df_test = pd.DataFrame(Y_test.astype('int'), columns=['target']) 
df_test.to_csv("/content/sample_data/predictions1.csv", index_label='ID')  

# Submit  /content/sample_data/predictions1.csv