from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('CaliforniaHousing').getOrCreate()

# step 1: read data and peek it
df = spark.read.csv('data/CaliforniaHousing/cal_housing.data', inferSchema=True)
df.printSchema()
df.show()

# 指定列名
df = df.select(df['_c0'].alias('longitude'),
               df['_c1'].alias('latitude'),
               df['_c2'].alias('housingMedianAge'),
               df['_c3'].alias('totalRooms'),
               df['_c4'].alias('totalBedrooms'),
               df['_c5'].alias('population'),
               df['_c6'].alias('households'),
               df['_c7'].alias('medianIncome'),
               df['_c8'].alias('medianHouseValue'))

vecAssembler = VectorAssembler(inputCols=['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                                         'totalBedrooms', 'population', 'households', 'medianIncome'],
                              outputCol='features')
assembledFeatures = vecAssembler.transform(df)

scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
scalerModel = scaler.fit(assembledFeatures)
scalerFeatures = scalerModel.transform(assembledFeatures)
scalerFeatures.printSchema()
scalerFeatures.show()

test_ratio = 0.3
seed = 1234

(training, testing) = scalerFeatures.randomSplit([1-test_ratio, test_ratio], seed)

lr = LinearRegression(featuresCol='scaledFeatures', labelCol='medianHouseValue',
                      maxIter=50, elasticNetParam=0.3)
lrModel = lr.fit(training)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# Summarize the model over the training set and print out some metrics
trainingSummary = lrModel.summary
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

predictions = lrModel.transform(testing)
predictions.show()
