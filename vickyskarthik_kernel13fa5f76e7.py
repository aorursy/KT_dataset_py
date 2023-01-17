# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import display # Allows the use of display() for DataFrames



# Pretty display for notebooks

%matplotlib inline



# Load the dataset

in_file = '/kaggle/input/titanic/train.csv'

train_data = pd.read_csv(in_file)



# Print the first few entries of the RMS Titanic data

display(train_data.head())
outcomes = train_data['Survived']

data = train_data.drop('Survived', axis = 1)



# Show the new dataset with 'Survived' removed

display(data.head())
def accuracy_score(truth, pred):

    """ Returns accuracy score for input truth and predictions. """

    

    # Ensure that the number of predictions matches number of outcomes

    if len(truth) == len(pred): 

        

        # Calculate and return the accuracy as a percent

        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)

    

    else:

        return "Number of predictions does not match number of outcomes!"

    

# Test the 'accuracy_score' function

predictions = pd.Series(np.ones(5, dtype = int))

print(accuracy_score(outcomes[:5], predictions))


def predictions_3(data):

    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    

    predictions = []

    for _, passenger in data.iterrows():

        if passenger['Sex'] == 'female':

            if passenger['Age'] > 40 and passenger['Age'] < 60 and passenger['Pclass'] == 3:

                predictions.append(0)

            else:

                predictions.append(1)

        else:

            if passenger['Age'] <= 10:

                predictions.append(1)

            elif passenger['Pclass'] == 1 and passenger['Age'] <= 40:

                predictions.append(1)

            else:

                predictions.append(0)

        

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions_3(data)

print(accuracy_score(outcomes, predictions))