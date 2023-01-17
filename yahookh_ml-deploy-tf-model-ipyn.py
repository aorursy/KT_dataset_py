import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Upload the trained TF model 
# The model contains the model construct and the hyperparameters.
# We will still need to declare/import the required tensorflow modules.
#
import numpy as np
from tensorflow import keras

my_model = keras.models.load_model("/kaggle/input/mytfmodel.h5")
my_model.summary()

#
# Test the uploaded TF model to predict the result.
#
x = 45
wt = 3
bias = 10

# format the input value and call the model prediction function
xs = np.array([x], dtype=float).reshape((-1, 1))
print("Input =", x, " Calculated =",wt*x + bias," Predicted by Trained Model =", my_model.predict([x]))