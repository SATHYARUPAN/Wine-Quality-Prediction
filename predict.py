import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.types import FloatType

warnings.filterwarnings('ignore')

# Initialize spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

testpath = dataset/TestDataset.csv
testdata = spark.read.option("delimiter", ";").option("header", "true").option("inferSchema","true").csv(testpath)

features = ['fixed acidity',
 'volatile acidity',
 'citric acid',
 'residual sugar',
 'chlorides',
 'free sulfur dioxide',
 'total sulfur dioxide',
 'density',
 'pH',
 'sulphates',
 'alcohol']

test_assembler = VectorAssembler(inputCols=features,outputCol="corr_features")
df_test = test_assembler.transform(testdata)

# Logistic Regression Model
logistic_reg = LogisticRegressionModel.load("./models/logisticregression")
logistic_reg_predictions = rf.transform(df_test)
logistic_reg_predictions.select('quality', 'rawPrediction', 'prediction', 'probability').show()
logistic_reg_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
logistic_reg_f1 = rf_evaluator.evaluate(logistic_reg_predictions)
print("RandomForest F1 score = %s" % (lr_f1))

# Random Forest Classifier Model
random_forest = RandomForestClassificationModel.load("./models/randomforest")
random_forest_predictions = rf.transform(df_test)
random_forest_predictions.select('quality', 'rawPrediction', 'prediction', 'probability').show()
random_forest_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
random_forest_f1 = rf_evaluator.evaluate(random_forest_predictions)
print("RandomForest F1 score = %s" % (rf_f1))

f1_score_path = "f1_score_test.txt"
with open(f1_score_path, 'w') as f:
    f.write("----------Results----------")
    f.write("\nLogistic Regression F1 Score on test dataset: {}".format(lr_f1))
    f.write("\nRandomForest F1 Score on test dataset: {}\n".format(rf_f1))
