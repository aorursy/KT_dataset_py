import sys
from math import sqrt
from pyspark.sql import SparkSession, functions, types, Row
import re

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+
def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        row = Row(host = m.group(1), byte = m.group(2)) #(xxxx)=group
        return row
    else:
        return None

def not_none(row):
    return row is not None

def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # return an RDD of Row() objects
    rdd = log_lines.map(line_to_row).filter(not_none)
    return rdd

line_re = re.compile("^(\\S+) - - \\[\\S+ [+-]\\d+\\] \"[A-Z]+ \\S+ HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
logs = spark.createDataFrame(create_row_rdd('../input/NASA_access_log_95'))
logs.cache()
totalByte = logs.groupby('host').agg(functions.sum('byte').alias('sum_byte'))
totalByte.cache()

requestNum = logs.groupby('host').agg(functions.count('host').alias('count'))
joined = requestNum.join(totalByte, 'host')
joined.cache()
a = joined.withColumn('x-squared', joined['count'] ** 2)
b = a.withColumn('y-squared', a['sum_byte'] ** 2)
aggOutput = b.withColumn('xy', b['count'] * b['sum_byte'])
aggOutput.cache().show()
n = aggOutput.groupBy().agg(functions.count('host')).first()[0]
x = aggOutput.groupBy().agg(functions.sum('count')).first()[0]
y = aggOutput.groupBy().agg(functions.sum('sum_byte')).first()[0]
xx = aggOutput.groupBy().agg(functions.sum('x-squared')).first()[0]
yy = aggOutput.groupBy().agg(functions.sum('y-squared')).first()[0]
xy = aggOutput.groupBy().agg(functions.sum('xy')).first()[0]
print(n,x,y,xx,yy,xy)
r = (n*xy - x*y) / (sqrt(n*xx-x**2)*sqrt(n*yy-y**2))
print("r = %g\nr^2 = %g" % (r, r**2))
