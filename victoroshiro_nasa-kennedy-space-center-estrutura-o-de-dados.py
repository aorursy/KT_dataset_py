!pip install pyspark
import pyspark
from pyspark.sql.types import StructType,StructField
from pyspark.sql.types import StringType, IntegerType, TimestampType
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re
import time
import datetime
from pyspark import SparkFiles

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
df = spark.read.csv("../input/*.txt")

#Função para coletar a coluna Host
def getValueHost(text):
    result = text.split(" ")
    return str(result[0])

#Função para coletar a coluna Timestamp
def getValueBetween(text):
    try:
        s = re.search('\[(.*)\]', text)
        s = str(s.group(0)).replace('[','')
        s = s.replace(']','')
        subStr = datetime.datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
        return subStr
    except:
        return None

#Função para coletar a coluna Bytes
def getValueCodes(text,num):
    try:
        result = text.split(" ")
        return int(result[len(result) - num])
    except:
        return None

#Função para coletar a coluna Status
def getValueStatus(text):
    try:
        result = text.split('"')
        result = result[len(result) - 1].split(' ')
        try:
            result.remove('')
        except ValueError:
            pass
        try:
            result.remove(' ')
        except ValueError:
            pass

        return int(result[0])
    except:
        return None

#Função para coletar a coluna Request
def getValueRequest(text):
    try:
        match = re.search('"([^"]*)"', text)
        return str(match.group(0)).replace('"','')
    except:
        return ""

#Acionando a função da coluna Host
udf_funcHost = udf(lambda x: getValueHost(x),returnType=StringType())
dfFinal = df.withColumn('host', udf_funcHost('_c0'))

#Acionando a função da coluna Timestamp
udf_funcTimeStamp = udf(lambda x: getValueBetween(x),returnType=TimestampType())
dfFinal = dfFinal.withColumn('timestamp', udf_funcTimeStamp('_c0'))

#Acionando a função da coluna Request
udf_funcRequest = udf(lambda x: getValueRequest(x),returnType=StringType())
dfFinal = dfFinal.withColumn('request', udf_funcRequest('_c0'))

#Acionando a função da coluna Status
udf_funcStatus = udf(lambda x: getValueStatus(x),returnType=IntegerType())
dfFinal = dfFinal.withColumn('status', udf_funcStatus('_c0'))

#Acionando a função da coluna bytes
udf_funcBytes = udf(lambda x: getValueCodes(x, 1),returnType=IntegerType())
dfFinal = dfFinal.withColumn('bytes', udf_funcBytes('_c0'))

#Retirando a coluna que guardava todos os dados em uma String 
dfFinal = dfFinal.drop('_c0')
re1 = (dfFinal.groupBy("host").count()).filter("count = 1")
re2 = (dfFinal.groupBy("status").count()).filter("status = 404")
re3 = (dfFinal.groupBy("host","status").count()).filter("status = 404").limit(5).orderBy('count', ascending=False)
dfFinal.createOrReplaceTempView("data")
re4 = spark.sql("select DATE(timestamp),status, count(*) as count from data where status = 404 group by DATE(timestamp),status order by DATE(timestamp)")
re5 = dfFinal.groupBy().sum("bytes")