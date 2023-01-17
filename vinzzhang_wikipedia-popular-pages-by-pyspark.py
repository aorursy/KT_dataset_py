import sys
import ntpath
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wikipedia_popular').getOrCreate()
assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+
schema = types.StructType([
    types.StructField('language', types.StringType(), False),
    types.StructField('pageName', types.StringType(), False),
    types.StructField('times', types.LongType(), False),
    types.StructField('returnedBytes', types.LongType(), False),
    types.StructField('filename', types.StringType(), False),
])
def getFilename(path):
    head, tail = ntpath.split(path)
    return tail[11:22]


def removeLine(time):
    time = int(time[0:8]+time[9:11])
    return time
popular = spark.read.csv('../input/pagecounts-20160802-140000', schema = schema, sep=' ')
#extract filename from input_file_name() 
path_to_hour = functions.udf(getFilename, returnType = types.StringType())
popular = popular.withColumn('filename', path_to_hour(functions.input_file_name()))
popular.show()
#filter
en = popular.filter(popular['language'] == 'en')
no_main = en.filter(en['pageName'] != 'Main_Page')
no_spe = no_main.filter(no_main['pageName'].startswith('Special:') == False)
no_spe.cache() #cache data for later repeattd use
#find max count
groups = no_spe.groupBy('filename')
max = groups.agg(functions.max('times').alias('max_times'))
#To allow tie
joined = no_spe.join(max, on=((no_spe['times'] == max['max_times']) & (no_spe['filename'] == max['filename']))) 
 #filename to int type to sort
removal = functions.udf(removeLine, returnType = types.FloatType())
new = joined.withColumn('date', removal(no_spe['filename']))
output = new.select(
    no_spe['filename'],
    'pageName',
    'max_times'
    )
# output.sort('date').write.csv(out_directory + '-popular', mode='overwrite')
output.sort('date').show()



