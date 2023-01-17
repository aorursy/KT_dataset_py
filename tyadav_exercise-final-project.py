import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
print("Setup Complete")
# Check for a dataset with a CSV file
step_1.check()
# Fill in the line below: Specify the path of the CSV file to read
my_filepath = "../input/POK2mons.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
sns.scatterplot(x=my_data['condenser_inlet_temp'], y=my_data['condenser_outlet_temp'])
sns.scatterplot(x=my_data['condenser_inlet_temp'], y=my_data['condenser_outlet_temp'], hue=my_data['water_flow_rate'])
sns.regplot(x=my_data['chiller_inlet_temp'], y=my_data['chiller_outlet_temp'])
sns.set_style("dark")

plt.figure(figsize=(10,8))
sns.barplot(data=my_data)