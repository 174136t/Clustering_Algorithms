
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# staring spark session
spark = SparkSession.builder.appName('Clustering Algorithm using K-Means').getOrCreate()

# reading the dataset
wine_clustering_data = spark.read.csv('wine-clustering.csv', header=True, inferSchema=True)
wine_clustering_data.printSchema()

# dropping null values
wine_clustering_data = wine_clustering_data.na.drop()
# wine_clustering_data.columns are the column heads of the dataset
print(wine_clustering_data.columns,'\n')

# defining features column
featuresCol = ['Ash_Alcanity','Alcohol',]
# adding our features column with vector values
wine_assembler = VectorAssembler(inputCols=featuresCol, outputCol='features')
assembled_wine_data = wine_assembler.transform(wine_clustering_data)
assembled_wine_data.show(5)

# converting Spark data to Pandas dataframe
wine_pddf = wine_clustering_data.toPandas()
# defining relevant column names to plot scatter diagram
plt.scatter(wine_pddf.Ash_Alcanity,wine_pddf.Alcohol)
# labeling x and y axis
plt.xlabel('Ash_Alcanity')
plt.ylabel('Alcohol')

wine_silhouette_score=[]
# defining clustering evaluator with prediction column
wine_evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', \
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
# training K-means model from range 2 to 7 
for i in range(2,7):
#  k is the number of clusters specified by the user
    wine_KMeans_algo = KMeans(featuresCol='features', k=i) 
    wine_KMeans_fit = wine_KMeans_algo.fit(assembled_wine_data)
    wine_output = wine_KMeans_fit.transform(assembled_wine_data)
#  evaluating silhoutte scores
    wine_score = wine_evaluator.evaluate(wine_output)
    wine_silhouette_score.append(wine_score)
    print("Silhouette Score is:",wine_score)

wine_output.show(5)

#  cluster center calculation
centers = wine_KMeans_fit.clusterCenters()
new_centers=[]
print("\n\nCluster Centers: ")
# printing centroids and pass them to new array
for center in centers:
    print(center)
x = np.vstack(centers)

# visualizing the silhouette scores in a plot
plt.plot(range(2,7),wine_silhouette_score,marker="o")
plt.xlabel('Number of Clusters')
plt.ylabel('WCCS')
plt.title('Silhouette Scores for k-means clustering')

# converting Spark data to Pandas dataframe
wine_pddf_pred = wine_output.toPandas()
# defining dataframes according to prediction values 
df1 = wine_pddf_pred[wine_pddf_pred.prediction==0]
df2 = wine_pddf_pred[wine_pddf_pred.prediction==1]
df3 = wine_pddf_pred[wine_pddf_pred.prediction==2]
df4 = wine_pddf_pred[wine_pddf_pred.prediction==3]
df5 = wine_pddf_pred[wine_pddf_pred.prediction==4]
df6 = wine_pddf_pred[wine_pddf_pred.prediction==5]
# defining colors for clusters
plt.scatter(df1.Ash_Alcanity,df1.Alcohol,color='blue',label='Cluster 01')
plt.scatter(df2.Ash_Alcanity,df2.Alcohol,color='green',label='Cluster 02')
plt.scatter(df3.Ash_Alcanity,df3.Alcohol,color='brown',label='Cluster 03')
plt.scatter(df4.Ash_Alcanity,df4.Alcohol,color='black',label='Cluster 04')
plt.scatter(df5.Ash_Alcanity,df5.Alcohol,color='violet',label='Cluster 05')
plt.scatter(df6.Ash_Alcanity,df6.Alcohol,color='orange',label='Cluster 06')
# plotting centroids
plt.scatter(x[:,0],x[:,1],color='red',marker='*',label='centroids')
plt.xlabel('Ash_Alcanity')
plt.ylabel('Alcohol')
plt.title('Scatter diagram of clusters')
plt.legend()
