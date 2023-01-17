import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

train_data = pd.read_csv('../input/train.csv', sep=',')

dog_outcomeType = [train_data[(train_data['OutcomeType'] == x) & (train_data['AnimalType'] == 'Dog')].shape[0] for x in train_data.OutcomeType.unique()]
cat_outcomeType = [train_data[(train_data['OutcomeType'] == x) & (train_data['AnimalType'] == 'Cat')].shape[0] for x in train_data.OutcomeType.unique()]

N = 5
ind = np.arange(N)
width = 0.35

p1 = plt.bar(ind, dog_outcomeType, width, color='b')
p2 = plt.bar(ind, cat_outcomeType, width, color='g',bottom=dog_outcomeType)

plt.ylabel('Number')
plt.title('Outcome Type of Dog and Cat')
plt.xticks(ind + width/2., train_data.OutcomeType.unique())
plt.legend((p1[0], p2[0]), ('Dog', 'Cat'))

plt.show()
