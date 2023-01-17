import gc
import feather
import pandas as pd
from timeit import default_timer
class Timer(object):
    """ A timer as a context manager
    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.
    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Adapted from: https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, prefix="", timer=default_timer,
                 output=None, fmt="took {:.2f} seconds"):
        self.timer = timer
        self.output = output
        self.fmt = fmt
        self.prefix = prefix
        self.end = None

    def __call__(self):
        """ Return the current time """
        return self.timer()

    def __enter__(self):
        """ Set the start time """
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:
            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])
            if callable(self.output):
                self.output(output)
            else:
                print(output)
        gc.collect()

    def __str__(self):
        return '%.3f' % (self.elapsed)

    @property
    def elapsed(self):
        """ Return the current elapsed time since start
        If the `elapsed` property is called in the context manager scope,
        the elapsed time bewteen start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time bewteen entering and exiting the scope.
        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.
        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            return self() - self.start
        else:
            # if elapsed is called out of the context manager scope
            return self.end - self.start

with Timer("train_V2.csv load time"):
    train_df = pd.read_csv("../input/train_V2.csv")
with Timer("test_V2.csv load time"):
    test_df = pd.read_csv("../input/test_V2.csv")
ft_train_path = "train_V2.feather"
ft_test_path = "test_V2.feather"
with Timer("train_V2.feather write time"):
    feather.write_dataframe(train_df, ft_train_path)
with Timer("test_V2.feather write time"):
    feather.write_dataframe(test_df, ft_test_path)
del train_df
del test_df
with Timer("train_V2.feather load time"):
    _ = feather.read_dataframe(ft_train_path)
with Timer("test_V2.feather load time"):
    _ = feather.read_dataframe(ft_test_path)
