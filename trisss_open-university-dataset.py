# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# show all output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Ingest data

# details here https://analyse.kmi.open.ac.uk/open_dataset
assessment = pd.read_csv("../input/assessments.csv")
assessment.head(5)

courses = pd.read_csv("../input/courses.csv")
courses.head(5)

studenta = pd.read_csv("../input/studentAssessment.csv")
studenta.head(5)

studentinfo = pd.read_csv("../input/studentInfo.csv")
studentinfo.head(5)

studentreg = pd.read_csv("../input/studentRegistration.csv")
studentreg.head(5)

vle = pd.read_csv("../input/vle.csv")
vle.head(5)
# only use student info and student assessment

#TASKS
# students graduated
# how many students withdrawn
# graph score of current module assessment - need to link assessment no with course - in student assessment
# how many students pass the module now - student info
# use plotly if hv enough time


# view all headers
studenta.columns
studentinfo.columns


# graph of 

# find out how many courses are there

courses.code_module.unique()
courses.code_presentation.unique()


# Plot of result count

# sns.distplot(studentinfo.final_result)

#plt.figure()
sns.countplot(studentinfo.final_result).set(title="Total number of pass and fail in all courses", 
                                            xlabel='Result' ,ylabel='Count' )


#sns.barplot(data = studentinfo, x='final_result', y='code_module')
# Plot of result count in each courses

AAA_result = studentinfo.loc[studentinfo['code_module'] == 'AAA']
sns.countplot(AAA_result.final_result).set(title="Total number of pass and fail in AAA course", 
                                            xlabel='Result' ,ylabel='Count' )
BBB_result = studentinfo.loc[studentinfo['code_module'] == 'BBB']
sns.countplot(BBB_result.final_result).set(title="Total number of pass and fail in BBB course", 
                                            xlabel='Result' ,ylabel='Count' )
CCC_result = studentinfo.loc[studentinfo['code_module'] == 'CCC']
sns.countplot(CCC_result.final_result).set(title="Total number of pass and fail in CCC course", 
                                            xlabel='Result' ,ylabel='Count' )
DDD_result = studentinfo.loc[studentinfo['code_module'] == 'DDD']
sns.countplot(DDD_result.final_result).set(title="Total number of pass and fail in DDD course", 
                                            xlabel='Result' ,ylabel='Count' )
EEE_result = studentinfo.loc[studentinfo['code_module'] == 'EEE']
sns.countplot(EEE_result.final_result).set(title="Total number of pass and fail in EEE course", 
                                            xlabel='Result' ,ylabel='Count' )
FFF_result = studentinfo.loc[studentinfo['code_module'] == 'FFF']
sns.countplot(FFF_result.final_result).set(title="Total number of pass and fail in FFF course", 
                                            xlabel='Result' ,ylabel='Count' )
GGG_result = studentinfo.loc[studentinfo['code_module'] == 'GGG']
sns.countplot(GGG_result.final_result).set(title="Total number of pass and fail in GGG course", 
                                            xlabel='Result' ,ylabel='Count' )
