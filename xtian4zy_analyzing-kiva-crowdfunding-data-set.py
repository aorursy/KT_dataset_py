#INTRODUCTION
'''
This notebook is my first attempt working with kaggle competition. An attempt is made to analyze the kiva loan dataset in other to find interesting patterns. Feel free to share your suggestions and comments. And if you 
feel i deserve a pat on the back, please leave an up vote.
'''
#Load Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Prepare to Load Data set
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))