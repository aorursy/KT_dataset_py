import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex5 import *

print("Setup Complete")
# Paths of the files to read

cancer_b_filepath = "../input/cancer_b.csv"

cancer_m_filepath = "../input/cancer_m.csv"



# Fill in the line below to read the (benign) file into a variable cancer_b_data

cancer_b_data = pd.read_csv(cancer_b_filepath, index_col="Id")



# Fill in the line below to read the (malignant) file into a variable cancer_m_data

cancer_m_data = pd.read_csv(cancer_m_filepath, index_col="Id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the (benign) data

cancer_b_data.head() # Your code here
# Print the first five rows of the (malignant) data

cancer_m_data.head() # Your code here
# Fill in the line below: In the first five rows of the data for benign tumors, what is the

# largest value for 'Perimeter (mean)'?

max_perim = cancer_b_data.iloc[0:5, 3].max()



# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517?

mean_radius = cancer_m_data.loc[842517, "Radius (mean)"]



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

step_2.solution()
# Histograms for benign and maligant tumors

sns.distplot(a=cancer_b_data["Area (mean)"], label="Benign", kde=False) # Your code here (benign tumors)

sns.distplot(a=cancer_m_data["Area (mean)"], label="Malignant", kde=False) # Your code here (malignant tumors)

plt.title("Tumor Distributions by Type and Area")

plt.legend()



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

step_3.a.solution_plot()
#step_3.b.hint()
step_3.b.solution()
# KDE plots for benign and malignant tumors

sns.kdeplot(data=cancer_b_data["Radius (worst)"], label="Benign", shade=True) # Your code here (benign tumors)

sns.kdeplot(data=cancer_m_data["Radius (worst)"], label="Malignant", shade=True) # Your code here (malignant tumors)

plt.legend()



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

step_4.a.solution_plot()
#step_4.b.hint()
step_4.b.solution()