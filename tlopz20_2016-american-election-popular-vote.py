import pandas as pd
data = pd.read_csv(r'/kaggle/input/2016-american-election-dataset/america.csv')

data.head()
Ndata = pd.DataFrame(data[['state_code','Trump_vote','Cliton_vote']])

Ndata.head()
Ndata.tail()
import matplotlib.pyplot as plt

%matplotlib inline
Ndata.set_index('state_code').plot(figsize = (20,8),grid= True)
Ndata.plot(x= "state_code", y=["Trump_vote","Cliton_vote"], kind="bar",figsize = (20,8))
Trumpsum = Ndata['Trump_vote'].sum()

Clitonsum = Ndata['Cliton_vote'].sum()

Trumpsum
Clitonsum
labels = 'Trump', 'Cliton'

sizes = [62883925, 66753516]

colors = ['lightcoral', 'lightskyblue']

explode = (0.1, 0)



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
#Please rate me in the comment section