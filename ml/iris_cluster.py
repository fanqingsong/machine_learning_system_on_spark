# task.py

import time, os
from celery import Celery, Task
import asyncio
from pyspark.sql import Row
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import sql
from numpy import array
from pyspark.sql import Row

# 实例化一个Celery
broker = 'redis://localhost:6379/1'
backend = 'redis://localhost:6379/2'

# 参数1 自动生成任务名的前缀
# 参数2 broker 是我们的redis的消息中间件
# 参数3 backend 用来存储我们的任务结果的
app = Celery('iris_cluster', broker=broker, backend=backend)

class IRIS_Cluster():
    def __init__(self):
        self._init_sparkcontext()

        self._model_path = "./kmeans_model"
        self._kmeans_model = None
        self._init_model()


    def train(self):
        kmeans = self._get_kmeans_instance()
        df = self._get_train_df()

        kmeansmodel = kmeans.fit(df)

        self._kmeans_model = kmeansmodel
        self._save_model()

        self._print_after_train(df)

    def predict(self, one_features):
        print("enter predict")

        if self._kmeans_model == None:
            print("no model, train first.")
            return

        df = self._get_predict_df(one_features)

        transformed = self._kmeans_model.transform(df)
        transformed.show()
        results = transformed.collect()
        print(results)

        predicted_cluster = None

        for item in results:
            print(str(item[0]) + ' is predcted as cluster' + str(item[1]))
            predicted_cluster = item[1]

        return predicted_cluster

    def _init_sparkcontext(self):
        spark_conf = SparkConf().setAppName("iris_cluster")

        sc = SparkContext(conf=spark_conf)
        sc.setLogLevel("ERROR")

        self._sparkcontext = sc

        self._sqlcontext = SQLContext(sc)

    def _get_predict_df(self, one_features):
        Features = Row('features')
        one_feature = Features(Vectors.dense(one_features))
        data = [one_feature]
        df = self._sqlcontext.createDataFrame(data)

        return df

    def _get_train_df(self):
        sc = self._sparkcontext
        rawData = sc.textFile("file:///root/win10/mine/machine_learning_system_on_spark/ml/iris.txt")

        def f(x):
            rel = {}
            rel['features'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
            return rel

        df = rawData.map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()

        return df

    def _get_kmeans_instance(self):
        kmeans = KMeans()

        kmeans.setK(3).setFeaturesCol('features').setPredictionCol('prediction')

        return kmeans

    def _save_model(self):
        model_path = self._model_path
        self._kmeans_model.write().overwrite().save(model_path)

    def _init_model(self):
        model_path = self._model_path
        if os.path.exists(model_path):
            self._kmeans_model = KMeansModel.load(model_path)

    def _print_after_train(self, df):
        print("----------- classified result ---------")
        kmeansmodel = self._kmeans_model
        transformed = kmeansmodel.transform(df)
        results = transformed.collect()
        for item in results:
            print(str(item[0]) + ' is predcted as cluster' + str(item[1]))

        print("----------- cluster centroid ---------")
        results2 = kmeansmodel.clusterCenters()
        for item in results2:
            print(item)


iris_cluster = IRIS_Cluster()

@app.task()
def train():
    print('Enter train function ...')
    iris_cluster.train()

    return 0

@app.task()
def predict(one_features):
    print('Enter predict function ...')
    result = iris_cluster.predict(one_features)

    return result



if __name__ == '__main__':
    # 这里生产的任务不可用，导入的模块不能包含task任务。会报错
    print("Start Task ...")

    r = train.delay()
    print("r=", r.get())

    one_feature = [5.1, 3.5, 1.4, 0.2]
    r = predict.delay(one_feature)
    print(one_feature, ", cluster=", r.get())


    time.sleep(20)
    #
    # # https://docs.celeryproject.org/en/4.0/whatsnew-4.0.html#asyncresult-then-on-success-on-error
    # # https://docs.telethon.dev/en/latest/concepts/asyncio.html
    # loop = asyncio.get_event_loop()  # get the default loop for the main thread
    # try:
    #     # run the event loop forever; ctrl+c to stop it
    #     # we could also run the loop for three seconds:
    #     #     loop.run_until_complete(asyncio.sleep(3))
    #     loop.run_forever()
    # except KeyboardInterrupt:
    #     pass

    print("End Task ...")

