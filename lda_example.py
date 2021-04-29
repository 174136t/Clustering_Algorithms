# importing libraries
from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
# starting spark session
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LDAExample") \
        .getOrCreate()


# loading data.
    wine_dataset = spark.read.csv('wine-clustering.csv', header=True, inferSchema=True)
# dropping null values
    wine_dataset = wine_dataset.na.drop()
    
 # wine_dataset.columns are the column heads of the dataset   
    print(wine_dataset.columns,'\n')
 # defining features column
    featuresCol = ['Ash_Alcanity','Alcohol',]
# adding our features column with vector values
    wine_assembler = VectorAssembler(inputCols=featuresCol, outputCol='features')

    assembled_wine_data = wine_assembler.transform(wine_dataset)
    assembled_wine_data.show(5)

# trains a LDA model.
    lda = LDA(k=10, maxIter=10)
    model = lda.fit(assembled_wine_data)

    ll = model.logLikelihood(assembled_wine_data)
    lp = model.logPerplexity(assembled_wine_data)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))

# describing topics.
    topics = model.describeTopics(3)
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)

# showing the result
    transformed = model.transform(assembled_wine_data)
    transformed.show(truncate=True)

    
# stop spark session
    spark.stop()