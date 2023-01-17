# Following

# http://mailchi.mp/d0302a0d3d83/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576389



# Check also

#https://google.github.io/styleguide/pyguide.html



# import pandas

import pandas as pd
# read file

data = pd.read_csv('../input/archive.csv')
# sumarise data

data.describe().transpose()