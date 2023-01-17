from pandas import read_csv

from matplotlib import pyplot

series = read_csv('../input/daily-min-temperatures/daily-min-temperatures.csv', header=0, index_col=0)

series.plot()

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf

series = read_csv('../input/daily-min-temperatures/daily-min-temperatures.csv', header=0, index_col=0)

plot_acf(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_pacf

series = read_csv('../input/daily-min-temperatures/daily-min-temperatures.csv', header=0, index_col=0)

plot_pacf(series, lags=50)

pyplot.show()