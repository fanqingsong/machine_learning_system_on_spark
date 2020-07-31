from pyspark.sql import Row
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext

spark_conf = SparkConf().setAppName("iris_model")
sc = SparkContext(conf=spark_conf)
sc.setLogLevel("ERROR")

rawData = sc.textFile("file://./iris.txt")
def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
    return rel
 
df = rawData.map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()

kmeansmodel = KMeans().setK(3).setFeaturesCol('features').setPredictionCol('prediction').fit(df)

results = kmeansmodel.transform(df).collect()
for item in results:
    print(str(item[0])+' is predcted as cluster'+ str(item[1]))


results2 = kmeansmodel.clusterCenters()
for item in results2:
    print(item)

kmeansmodel.computeCost(data)


