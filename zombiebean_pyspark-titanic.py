from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.sql import functions as func
from pyspark.sql.functions import col,count,round,mean,regexp_extract,array
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator,Bucketizer,VectorAssembler,MinMaxScaler
from pyspark.sql.functions import pandas_udf, PandasUDFType,udf
from pyspark.sql.types import *
from pyspark.sql.functions import create_map, lit, struct
from itertools import chain

from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

# spark = SparkSession \
#         .builder \
#         .master('local[4]') \
#         .appName("Pyspark Titanic") \
#         .config("spark.some.config.option", "some-value") \
#         .getOrCreate()
spark = SparkSession \
        .builder \
        .master('yarn-client') \
        .appName("Pyspark Titanic") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
#get the benifit of arrow storage method ,which make topandas more effient
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
#build the file path
# train_path='file:///F:/learning/Titanic/Input/train.csv'
# test_path='file:///F:/learning/Titanic/Input/test.csv'
train_path='hdfs:/user/hdfs/data/Titanic/train.csv'
test_path='hdfs:/user/hdfs/data/Titanic/test.csv'
#build the data type schema
# schema = StructType(
#   [StructField("PassengerId", IntegerType()),
#     StructField("Survival", ShortType()),
#     StructField("Pclass", ShortType()),
#     StructField("Name", StringType()),
#     StructField("Sex", StringType()),
#     StructField("Age", ShortType()),
#     StructField("SibSp",ShortType()),
#     StructField("Parch", ShortType()),
#     StructField("Ticket", StringType()),
#     StructField("Fare", FloatType()),
#     StructField("Cabin", StringType()),
#     StructField("Embarked", StringType())
#   ])

#load the data,using the inferSchema
train_df = spark.read.format('csv').option('header', 'true').option("inferSchema", "true").load(train_path)
test_df  = spark.read.format('csv').option('header', 'true').option("inferSchema", "true").load(test_path)
#train_df = spark.read.format('csv').option('header', 'true').schema(schema).load(train_path)
#test_df  = spark.read.format('csv').option('header', 'true').schema(schema).load(test_path)#this is not true ,because test has not Survial feature
train_df.printSchema()
test_df.printSchema()
###look the missing ratio
# for data in full_data:
#     miss_ratio=data.agg(*(round((1-count(c) / count('*')),2) \
#     .alias(c) for c in data.columns))
#     miss_ratio.show()
for data in (train_df,test_df):
    miss_ratio=data.agg(*(count(c) \
    .alias(c) for c in data.columns))
    miss_ratio.show()
#get the average age and fill in missing values
def age_mean_fill(df):
    mean_age=df.agg(mean(col('Age')).cast('int')).first()[0]
    return df.fillna({'Age':mean_age})
train_df=age_mean_fill(train_df)
test_df=age_mean_fill(test_df)
#get the most frequent embarked value and fill in missing values
def embarked_mode_fill(df):
    mode=df.groupby('Embarked').count().orderBy('count',ascending=False).head()[0]
    return df.fillna({"Embarked":mode})
train_df=embarked_mode_fill(train_df)
test_df=embarked_mode_fill(test_df)
train_df=train_df.fillna({"Fare":0.0})
test_df=test_df.fillna({"Fare":0.0})
for data in (train_df,test_df):
    miss_ratio=data.agg(*(count(c) \
    .alias(c) for c in data.columns))
    miss_ratio.show()
train_df=train_df.withColumn('Has_Cabin',col('Cabin').isNotNull().cast('int'))
test_df=test_df.withColumn('Has_Cabin',col('Cabin').isNotNull().cast('int'))
train_df=train_df.withColumn('Title', regexp_extract(col('Name'), ' ([A-Za-z]+)\.', 1))
test_df=test_df.withColumn('Title', regexp_extract(col('Name'), ' ([A-Za-z]+)\.', 1))
# Prefix cleaning
to_replace = {'Capt' : 'Rare',
              'Col' :'Rare',
              'Don' : 'Rare',
              'Dr' : 'Rare',
              'Major' : 'Rare',
              'Rev' : 'Rare',
              'Jonkheer' : 'Rare',
              'Dona' : 'Rare',
              'Countess' : 'Royal',
              'Lady' : 'Royal',
              'Sir' : 'Royal',
              'Mlle' : 'Miss',
              'Ms' : 'Miss',
              'Mme' : 'Mrs'}
train_df=train_df.replace(to_replace, None, 'Title')
test_df=test_df.replace(to_replace, None, 'Title')
# train_df=train_df.na.replace(to_replace, None, 'Prefix')
# test_df=test_df.na.replace(to_replace, None, 'Prefix')
test_df.show(10)
train_df.show(10)
def handleCategorical(catcol):
    indexer=StringIndexer(inputCol=catcol, outputCol=catcol+'Index').setHandleInvalid("keep")
    encoder= OneHotEncoderEstimator(inputCols=[catcol+'Index'], outputCols=[catcol+'_onehot'])
    return [indexer,encoder]
genderStages=handleCategorical('Sex')
embarkedStages = handleCategorical("Embarked")
pClassStages = handleCategorical("Pclass")
hasCabinStages=handleCategorical("Has_Cabin")
titleStage=handleCategorical('Title')
preProcessStages=genderStages+embarkedStages+pClassStages+hasCabinStages+titleStage
splits = [-float("inf"),7.91, 14.454, 31, float("inf")]
bucketizer = Bucketizer(splits=splits,inputCol="Fare", outputCol=" Fare_category")
preProcessStages+=[bucketizer]
sta_features = [ 'Age', 'Fare']
### 转换特征
feat_vector = VectorAssembler(inputCols=sta_features, outputCol='vector_features')
### 标准化
stda = MinMaxScaler(inputCol='vector_features', outputCol='stda_features')
sta_stage=[feat_vector,stda]
preProcessStages+=sta_stage
#add all the column to a feature vector
cols=["Sex_onehot", "Embarked_onehot", "Pclass_onehot", 'Has_Cabin_onehot','Title_onehot',"SibSp", "Parch",'stda_features']
assembler = VectorAssembler(inputCols=cols,outputCol="features")
preProcessStages+=[assembler]
trainDF, testDF = train_df.randomSplit([0.8, 0.2], seed=24)
rf = RandomForestClassifier(labelCol='Survived', featuresCol='features')
wholeStages=preProcessStages+[rf]
pipeline=Pipeline(stages=wholeStages)

paramGrid = ParamGridBuilder()\
           .addGrid(rf.maxDepth, range(3,10,1))\
           .addGrid(rf.numTrees, range(100,150,15))\
           .addGrid(rf.minInstancesPerNode, range(2,10,2))\
           .build()

# Set AUC as evaluation metric for best model selection
evaluator =BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', 
                                         labelCol='Survived', metricName='areaUnderROC')

# Set up 3-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
model=crossval.fit(trainDF)
result = model.transform(testDF)
result.select("PassengerId", "prediction").show(5)
print(evaluator.evaluate(result,
{evaluator.metricName: 'areaUnderROC'}))
print(evaluator.evaluate(result,
{evaluator.metricName: 'areaUnderPR'}))
modelPath = './Survival_oneHotEncoder_RandomForest_PipelineModel'
model.bestModel.save(modelPath)
# model.bestModel.write().overwrite().save(modelPath)
loadedPipelineModel = PipelineModel.load(modelPath)
test_reloadedModel = loadedPipelineModel.transform(test_df)
test_reloadedModel.select("PassengerId", "prediction").show(5)
spark_submitssion=test_reloadedModel.select("PassengerId", "prediction") \
        .withColumn('prediction', test_reloadedModel['prediction'].cast('int'))
spark_submitssion=spark_submitssion.withColumnRenamed('prediction','Survived').toPandas()
spark_submitssion_path='F:/learning/Titanic/Input/spark_submitssion.csv'
spark_submitssion.to_csv(spark_submitssion_path, index=False)