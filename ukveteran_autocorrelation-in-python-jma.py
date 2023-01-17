from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_air_passengers.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_retail_sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_wp_log_R.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_wp_log_R_outliers1.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_wp_log_R_outliers2.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_wp_log_peyton_manning.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()
from pandas import read_csv

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

series = read_csv('../input/data-for-autocorrelation/example_yosemite_temps.csv', header=0, index_col=0, parse_dates=True, squeeze=True)

autocorrelation_plot(series)

pyplot.show()