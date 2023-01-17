import numpy as np

import pandas as pd

from subprocess import check_output



datasetDirectory = "../input"



for fileName in ( fileName for fileName in 

                  check_output(["ls", datasetDirectory]).decode("utf8").split("\n") 

                  if len(fileName) > 0 and fileName.endswith("csv") ):

    print("-" * 80)

    print("File: {}".format(fileName))

    print("-" * 80)

    print( pd.read_csv(datasetDirectory + "/" + fileName).describe() )