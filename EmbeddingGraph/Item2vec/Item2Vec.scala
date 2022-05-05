package Easyml.GraphEmbedding

import org.apache.log4j.{Level, Logger}
import org.apache.parquet.filter2.predicate.Operators.UserDefined
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array_join, col, collect_list, concat_ws, struct, udf}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.sql.types.DoubleType
import scopt.OptionParser

import java.io.{BufferedWriter, File, FileWriter}

object Item2Vec {
  /** create by berwin 2021/04/15
   *
   * @param input_pt     hdfs数据路径
   * @param timeName     时间戳列名
   * @param itemName     item列名
   * @param userName     user列名
   * @param filter       过滤条件  没有则为""   按照spark sql规范  如 "a >= 0.3"
   * @param embLength    Word2vec中的生成embedding隐向量维度 默认值为8
   * @param windowsSise  Word2vec中的滑动窗口的长度 默认值为5
   * @param output_pt    输出路径
   
   * function:
   *          对行为序列做item2vec，生成序列中各个物品的embedding
   *
   *          说明:以MovieLens数据为例,并且截取序列中评分较高的,为什么?
   *
   *          因为item2vec的本质是希望学习到物品之间的相似性,
   *          故希望评分好的电影靠近一些,评分差的电影和评分好的电影不要在序列中成对出现.
   *
   *          故先输入前需手动过滤序列中评分低的item,保证输入均为处理好序列.
   *
   */

  def main(args: Array[String]) {
//    if (args.length < 2) {
//      System.err.println("error")
//      System.exit(1)
//    }
//    val default_params = Params()
//    val parser = new OptionParser[Params]("Accuracy") {
//      head("Accuracy: Compute the Accuracy")
//      opt[String]("input_pt")
//        .required()
//        .text("Input document file path")
//        .action((x, c) => c.copy(input_pt = x))
//      opt[String]("output_pt")
//        .required()
//        .text("Output document file path")
//        .action((x, c) => c.copy(output_pt = x))
//      opt[Double]("threshold")
//        .required()
//        .text("Output dimension")
//        .action((x, c) => c.copy(threshold = x))
//    }
//    parser.parse(args, default_params).map { params =>
//      run(params)
//    } getOrElse {
//      System.exit(1)
//    }
    Logger.getLogger("org").setLevel(Level.ERROR)

    val input_pt = "xxxxxx/ratings.csv"
    val timeName = "timestamp"
    val itemName = "movieId"
    val userName = "userId"
    val filter = "rating>3.5"
    val embLength = 16
    val windowsSise = 5
    val output_pt = "xxxx/GraphEmbedding/"
//


    val spark = SparkSession.builder().master("local").appName("Demo").getOrCreate()//.enableHiveSupport()
    val res = generateSequence(spark, input_pt,timeName, itemName, userName, filter)
    println(res)
    val model = trainItems2vec(spark, res, embLength, windowsSise)

    model.save(spark.sparkContext,output_pt)
//    val synonyms = model.findSynonyms("158", 20)
//    for ((synonym, cosineSimilarity) <- synonyms) {
//      println(s"$synonym $cosineSimilarity")
//    }

    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(output_pt + "item2vec.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    for (movieId <- model.getVectors.keys) {
      bw.write(movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
    }
    bw.close()

    spark.close()
  }
  def generateSequence(spark:SparkSession, input_pt:String , timeName:String, itemName:String, userName:String, filter:String):RDD[Seq[String]]={

    var data = spark.read.format("csv").option("header","true").load(input_pt)

    val sortUdf:UserDefinedFunction = udf((rows:Seq[Row]) =>{
      rows.map{case Row(movieId:String, timestamp:String) => (movieId,timestamp)}
        .sortBy {case(_,timestamp) => timestamp}
        .map{case(movieId, _) => movieId}
    })

    var tuserSeq = data
    if(filter!= ""){
      data.createOrReplaceTempView("data")
      tuserSeq = spark.sql(s"select * from data where $filter ")
    }

    val userSeq = tuserSeq
      .groupBy(userName)
      .agg(sortUdf(collect_list(struct(itemName, timeName))) as "Ids")
      .withColumn("Ids", concat_ws(" ",col("Ids")))


    userSeq.select("Ids").rdd.map(r => r.getAs[String]("Ids").split(" ").toSeq)
  }
  def trainItems2vec(sparkSession: SparkSession, samples : RDD[Seq[String]], embLength:Int, windowsSise:Int): Word2VecModel = {
//    使用方法官方API： https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec
//    setVectorSize用于设定生成的 Embedding向量的维度
//    setWindowSize用于设定在序列数据上采样的滑动窗口大小
//    setNumlterations用于设定训练时的迭代次数
    val word2vec = new Word2Vec()
      .setVectorSize(embLength)
      .setWindowSize(windowsSise)
      .setNumPartitions(10)

    val model: Word2VecModel = word2vec.fit(samples)
    model

  }

}
