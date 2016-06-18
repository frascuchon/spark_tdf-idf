package com.ai.recogn.classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, StringIndexer, Tokenizer}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by francisco.aranda@gft.com on 18/06/16.
  */
object TdfIdfClassification {

  private val TEXT_COLUMN = "text"
  private val TOKENS_COLUMN = "tokens"
  private val TF_COLUMN = "tf"
  private val RAW_LABEL_COLUMN = "rawLabel"
  private val LABEL_COLUMN = "label"
  private val FEATURES_COLUMN = "features"

  def main(args: Array[String]) {

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Text classification")

    val sc: SparkContext = new SparkContext(conf)
    implicit val sqlContext = new SQLContext(sc)

    val parseTrain: DataFrame = readDataset("dbpedia_csv/train.csv")

    val tokenizer = new Tokenizer()
      .setInputCol(TEXT_COLUMN)
      .setOutputCol(TOKENS_COLUMN)

    val featuresSize = 10000

    val tf = new HashingTF()
      .setInputCol(TOKENS_COLUMN)
      .setOutputCol(TF_COLUMN)
      .setNumFeatures(featuresSize) // TODO: from dataframe

    val labelIndexer = new StringIndexer()
      .setInputCol(RAW_LABEL_COLUMN)
      .setOutputCol(LABEL_COLUMN)

    val idf = new IDF()
      .setInputCol(TF_COLUMN)
      .setOutputCol(FEATURES_COLUMN)

    val classifier = new OneVsRest().setClassifier(
      new LogisticRegression()
        .setMaxIter(1)
        .setRegParam(0.01)
        .setLabelCol(LABEL_COLUMN)
    )

    //    new MultilayerPerceptronClassifier()
    //      .setMaxIter(1)
    //      .setFeaturesCol("features")
    //      .setLabelCol("label")
    //      .setLayers(Array(featuresSize, 200, 15))

    val idfPipe = new Pipeline().setStages(Array(labelIndexer, tokenizer, tf, idf))
    val idfModel = idfPipe.fit(parseTrain)


    val model = classifier.fit(idfModel.transform(parseTrain))

    //model.save("model-tfIdf")

    val testDF = readDataset("dbpedia_csv/test.csv")
    val fitTestDF = idfModel.transform(testDF)

    val result = model.transform(fitTestDF)

    val predictionsAndLabels = result.select("prediction", LABEL_COLUMN)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("precision")

    println("Precision: " + evaluator.evaluate(predictionsAndLabels))
  }

  def readDataset(path: String)(implicit sqlContext: SQLContext): DataFrame = {
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .schema(StructType(Seq(
        RAW_LABEL_COLUMN,
        "title",
        "abstract").map(StructField(_, StringType, nullable = false))))
      .load(path)

    df.withColumn(TEXT_COLUMN, df("abstract"))
  }
}
