import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import scala.io.Source
import scala.collection.mutable.HashMap
import java.io.File
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.collection.mutable.ListBuffer
import org.apache.spark.util.IntParam
import org.apache.spark.util.StatCounter
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd._
import org.apache.spark.sql.functions.{udf => udF}

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.clustering.LDA

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vectors


/* Read Data */
// read original data
val sqlContext = new SQLContext(sc)
val rawData = sqlContext.read.format("csv").option("header","true").load("/home/zluxyoe/cleaned_questions.csv")

// transfer data to a wanted schema
def getYearUdf = udF[Int, String] { date => date.split("-")(0).toInt }
def getMonthUdf = udF[Int, String] { date => date.split("-")(1).toInt }
def getHourUdf = udF[Int, String]{ date => date.split("T")(1).split(":")(0).toInt}
def toDouble = udF[Double, String](_.toDouble)
val data = rawData.withColumn("id", $"Id").withColumn("title", $"Title").withColumn("ownerUserId", $"OwnerUserId").withColumn("score", toDouble($"Score")).withColumn("year", getYearUdf($"CreationDate")).withColumn("month", getMonthUdf($"CreationDate")).withColumn("Hour", getHourUdf($"CreationDate")).drop("CreationDate").drop("ClosedDate")

// same line, for copy purpose
//val data = rawData.withColumn("id", $"Id").withColumn("title", $"Title").withColumn("ownerUserId", $"OwnerUserId").withColumn("score", toDouble($"Score")).withColumn("year", getYearUdf($"CreationDate")).withColumn("month", getMonthUdf($"CreationDate")).drop("CreationDate").drop("ClosedDate")

// Define a tokenizing stage and tokenize title into word sequence
val tokenizer = new Tokenizer().setInputCol("title").setOutputCol("words")
val wordsData = tokenizer.transform(data)
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
val filteredWords = remover.transform(wordsData).drop("words")


/* Basic data analysis */
// (1) data stats or each column
data.count // 1264216
data.select($"id").distinct.count // 1264216
data.select($"ownerUserId").distinct.count // 630910
data.select($"score").describe().show
// +-------+-----------------+
// |summary|            score|
// +-------+-----------------+
// |  count|          1264216|
// |   mean|1.781537332228037|
// | stddev|13.66388572556547|
// |    min|            -73.0|
// |    max|           5190.0|
// +-------+-----------------+
data.select($"year").distinct.show // 2008 -2016
data.groupBy($"year").count.sort($"count".desc).show
// +----+------+                                                                   
// |year| count|
// +----+------+
// |2015|230038|
// |2014|217672|
// |2016|211049|
// |2013|206914|
// |2012|166442|
// |2011|121117|
// |2010| 70643|
// |2009| 34515|
// |2008|  5826|
// +----+------+


/* Top 10 scored questions */
data.sort($"score".desc).select($"id", $"score", $"title", $"year").show(false)
// +--------+------+-----------------------------------------------------------------------------------+----+
// |id      |score |title                                                                              |year|
// +--------+------+-----------------------------------------------------------------------------------+----+
// |348170  |5190.0|How to undo 'git add' before commit?                                               |2008|
// |40480   |3613.0|Is Java "pass-by-reference" or "pass-by-value"?                                    |2008|
// |406230  |2537.0|Regular expression to match line that doesn't contain a word?                      |2009|
// |520650  |2399.0|Make an existing Git branch track a remote branch?                                 |2009|
// |2669690 |2363.0|Why does Google prepend while(1); to their JSON responses?                         |2010|
// |4366730 |1760.0|How to check if a string contains a specific word in PHP?                          |2010|
// |1232040 |1759.0|How do I empty an array in JavaScript?                                             |2009|
// |16956810|1716.0|How to find all files containing specific text on Linux?                           |2013|
// |3010840 |1620.0|Loop through an array in JavaScript                                                |2010|
// |2530    |1614.0|How do you disable browser Autocomplete on web form field / input tag?             |2008|
// |806000  |1613.0|How do I give text or an image a transparent background using CSS?                 |2009|
// |4089430 |1567.0|How can I determine the URL that a local Git repository was originally cloned from?|2010|
// |728360  |1520.0|How do I correctly clone a JavaScript object?                                      |2009|
// |1098040 |1513.0|Checking if a key exists in a JavaScript object?                                   |2009|
// |332030  |1473.0|When should static_cast dynamic_cast const_cast and reinterpret_cast be used?      |2008|
// |894860  |1460.0|Set a default parameter value for a JavaScript function                            |2009|
// |2793150 |1412.0|Using java.net.URLConnection to fire and handle HTTP requests                      |2010|
// |489340  |1327.0|Vertically align text next to an image?                                            |2009|
// |5189560 |1147.0|Squash my last X commits together using Git                                        |2011|
// |1218390 |1129.0|What is your most productive shortcut with Vim?                                    |2009|
// +--------+------+-----------------------------------------------------------------------------------+----+


// highest scored questions for each year
data.createOrReplaceTempView("datasql")

val scoresQ = spark.sql("select score,year, title from datasql where score in (select max(score) from datasql group by year) order by year")
scoresQ.show(false)
// /+------+----+-----------------------------------------------------------------------------+
// |score |year|title                                                                        |
// +------+----+-----------------------------------------------------------------------------+
// |5190.0|2008|How to undo 'git add' before commit?                                         |
// |2537.0|2009|Regular expression to match line that doesn't contain a word?                |
// |2363.0|2010|Why does Google prepend while(1); to their JSON responses?                   |
// |1147.0|2011|Squash my last X commits together using Git                                  |
// |318.0 |2011|iphone dismiss keyboard when touching outside of UITextField                 |
// |930.0 |2012|How to exit the VIM editor?                                                  |
// |1716.0|2013|How to find all files containing specific text on Linux?                     |
// |605.0 |2014|What is the difference between the `COPY` and `ADD` commands in a Dockerfile?|
// |472.0 |2015|Why is [] faster than list()?                                                |
// |318.0 |2016|Huge number of files generated for every AngularJS 2 project                 |
// +------+----+-----------------------------------------------------------------------------+

/+------+----+-----------------------------------------------------------------------------+
|score |year|title                                                                        |
+------+----+-----------------------------------------------------------------------------+
|5190.0|2008|How to undo 'git add' before commit?                                         |
|2537.0|2009|Regular expression to match line that doesn't contain a word?                |
|2363.0|2010|Why does Google prepend while(1); to their JSON responses?                   |
|1147.0|2011|Squash my last X commits together using Git                                  |
|318.0 |2011|iphone dismiss keyboard when touching outside of UITextField                 |
|930.0 |2012|How to exit the VIM editor?                                                  |
|1716.0|2013|How to find all files containing specific text on Linux?                     |
|605.0 |2014|What is the difference between the `COPY` and `ADD` commands in a Dockerfile?|
|472.0 |2015|Why is [] faster than list()?                                                |
|318.0 |2016|Huge number of files generated for every AngularJS 2 project                 |
+------+----+-----------------------------------------------------------------------------+


/* Hour distribution */
data.filter($"year"===2013).groupBy($"hour").count.sort($"count").show(24)

// /+----+-----+                                                                    
// |hour|count|
// +----+-----+
// |   2| 4804|
// |   1| 4816|
// |   3| 4968|
// |   0| 5144|
// |   4| 5181|
// |  23| 5943|
// |   5| 6261|
// |  22| 7285|
// |   6| 7586|
// |   7| 8509|
// |  21| 8587|
// |   8| 8913|
// |  20| 9400|
// |  19| 9870|
// |  18|10066|
// |  17|10370|
// |   9|10609|
// |  11|10635|
// |  12|10707|
// |  10|11015|
// |  16|11041|
// |  13|11325|
// |  14|11871|
// |  15|12008|
// +----+-----+
data.filter($"year"===2014).groupBy($"hour").count.sort($"count").show(24)

// +----+-----+                                                                    
// |hour|count|
// +----+-----+
// |   2| 4869|
// |   3| 4996|
// |   1| 5039|
// |   0| 5369|
// |   4| 5542|
// |  23| 6417|
// |   5| 6557|
// |  22| 7577|
// |   6| 8037|
// |   7| 8896|
// |  21| 8950|
// |   8| 9477|
// |  20| 9776|
// |  19|10018|
// |  18|10611|
// |  17|10640|
// |  11|11410|
// |   9|11479|
// |  12|11487|
// |  10|11656|
// |  16|11658|
// |  13|11996|
// |  14|12584|
// |  15|12631|
// +----+-----+

data.filter($"year"===2015).groupBy($"hour").count.sort($"count").show(24)

// +----+-----+                                                                    
// |hour|count|
// +----+-----+
// |   1| 5128|
// |   2| 5187|
// |   3| 5294|
// |   0| 5496|
// |   4| 5774|
// |  23| 6560|
// |   5| 6907|
// |  22| 7778|
// |   6| 8620|
// |  21| 9034|
// |   7| 9604|
// |  20|10011|
// |   8|10245|
// |  19|10703|
// |  18|11035|
// |  17|11441|
// |  11|12132|
// |  12|12258|
// |   9|12301|
// |  16|12372|
// |  10|12382|
// |  13|12913|
// |  15|13337|
// |  14|13526|
// +----+-----+

data.filter($"year"===2016).groupBy($"hour").count.sort($"count").show(24)

// +----+-----+                                                                    
// |hour|count|
// +----+-----+
// |   1| 4507|
// |   0| 4654|
// |   2| 4669|
// |   3| 4939|
// |   4| 5229|
// |  23| 5423|
// |   5| 6350|
// |  22| 6616|
// |  21| 7958|
// |   6| 8253|
// |  20| 8881|
// |  19| 9473|
// |   7| 9507|
// |  18| 9740|
// |   8| 9894|
// |  17|10223|
// |  16|11090|
// |  11|11523|
// |  12|11614|
// |   9|11770|
// |  10|11886|
// |  13|12212|
// |  15|12237|
// |  14|12401|
// +----+-----+

/* Trend analysis */
val explodedWords = filteredWords.select($"year", $"filtered").withColumn("word", explode($"filtered")).drop("filtered")

// (1) total word frequency accross all years
explodedWords.groupBy($"word").count.sort($"count".desc).show(false)
// +----------+-----+                                                              
// |word      |count|
// +----------+-----+
// |using     |91595|
// |-         |68617|
// |file      |49421|
// |get       |44010|
// |data      |43144|
// |error     |42794|
// |android   |37789|
// |php       |32795|
// |use       |32111|
// |jquery    |31244|
// |value     |30361|
// |function  |28586|
// |java      |28220|
// |array     |27119|
// |multiple  |27025|
// |string    |25375|
// |working   |24213|
// |javascript|24210|
// |python    |23790|
// |sql       |23382|
// +----------+-----+

// (2) get word/topic trend/count over year
def getCountOverYear(df: DataFrame, year: Int, word: String): Long = { df.filter($"year" === year && $"word" === word).count }
val languageList: List[String] = List("javascript", "html", "css", "sql", "java", "bash", "shell", "python", "c#", "php", "c++", "ruby", "swift", "go", "objective-c", "vb.net", "r", "matlab", "vba", "scala", "spark", "angularjs", "nodejs")
val editorList: List[String] = List("visual", "notepad++", "sublime", "vim", "intellij", "android", "eclipse", "atom", "pycharm", "xcode", "phpstorm", "netbeans", "ipython", "emacs", "rstudio", "rubymine")
def getCountOverYear(df: DataFrame, year: Int, word: String): Long = { df.filter($"year" === year && $"word" === word).count }
def getListCountForYear(df: DataFrame, year: Int, itemList: List[String]) = {
	val counts = itemList.map{ item => getCountOverYear(df, year, item) };
	itemList.zip(counts)
}

// each year data
val year = 2008
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2)
// 31923
// (sql,229), (java,155), (c#,155), (php,104), (javascript,99), (c++,92), (python,85), (html,78), (ruby,43), (css,41), (vb.net,22), (bash,9), (shell,6), (go,6), (objective-c,6), (vba,5), (matlab,3), (r,1), (scala,1), (swift,0), (spark,0), (angularjs,0), (nodejs,0)
// (visual,96), (eclipse,41), (emacs,11), (vim,7), (xcode,5), (netbeans,5), (intellij,3), (android,2), (atom,1), (notepad++,0), (sublime,0), (pycharm,0), (phpstorm,0), (ipython,0), (rstudio,0), (rubymine,0)

val year = 2009
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2)
// 188351
// (sql,1109), (c#,1026), (java,833), (php,831), (javascript,632), (python,527), (c++,499), (html,406), (ruby,264), (css,232), (vb.net,104), (shell,63), (objective-c,58), (vba,56), (bash,53), (go,49), (r,31), (matlab,29), (scala,21), (spark,5), (swift,1), (angularjs,0), (nodejs,0)
// (visual,462), (eclipse,161), (android,120), (xcode,82), (emacs,53), (vim,45), (netbeans,41), (intellij,15), (notepad++,6), (atom,6), (ipython,5), (sublime,0), (pycharm,0), (phpstorm,0), (rstudio,0), (rubymine,0)

val year = 2010
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 393209
// (php,1965), (java,1714), (sql,1698), (c#,1630), (javascript,1339), (python,1060), (c++,950), (html,914), (ruby,563), (css,521), (vb.net,187), (objective-c,154), (shell,134), (bash,119), (matlab,119), (scala,107), (r,104), (go,97), (vba,97), (spark,27), (nodejs,6), (swift,3), (angularjs,0)
// (android,1252), (visual,660), (eclipse,381), (xcode,181), (netbeans,114), (vim,86), (emacs,81), (intellij,27), (notepad++,14), (atom,8), (ipython,3), (rubymine,3), (phpstorm,2), (pycharm,1), (sublime,0), (rstudio,0)
// During the 2010s there have been increased efforts towards standardisation and code sharing in PHP applications

val year = 2011
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 676774
// (php,3582), (java,2705), (javascript,2537), (sql,2343), (c#,2255), (html,1635), (python,1614), (c++,1251), (css,1043), (ruby,758), (r,242), (vb.net,229), (objective-c,220), (matlab,209), (bash,205), (shell,202), (scala,195), (go,185), (vba,143), (nodejs,51), (spark,23), (swift,1), (angularjs,0)
// (android,3833), (visual,759), (eclipse,647), (xcode,537), (netbeans,138), (vim,134), (emacs,118), (intellij,59), (notepad++,19), (ipython,13), (atom,5), (pycharm,4), (rubymine,4), (sublime,3), (phpstorm,3), (rstudio,2)

val year = 2012
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 943245
// (php,4888), (java,3849), (javascript,3552), (sql,3000), (python,2464), (c#,2391), (html,2121), (c++,1712), (css,1488), (ruby,997), (r,512), (matlab,388), (vb.net,351), (shell,337), (bash,331), (scala,248), (objective-c,242), (vba,242), (go,199), (nodejs,121), (angularjs,65), (spark,16), (swift,3)
// (android,5651), (eclipse,982), (visual,905), (xcode,744), (vim,185), (netbeans,173), (emacs,151), (intellij,95), (sublime,74), (notepad++,37), (ipython,32), (atom,12), (rubymine,8), (pycharm,7), (phpstorm,5), (rstudio,4)

val year = 2013
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 1184023
// (php,5737), (java,4679), (javascript,4150), (python,3811), (sql,3667), (c#,2875), (html,2800), (c++,2074), (css,1953), (r,986), (ruby,965), (angularjs,674), (matlab,596), (vba,521), (shell,443), (bash,441), (vb.net,407), (go,341), (scala,325), (objective-c,212), (nodejs,204), (spark,20), (swift,5)
// (android,6398), (visual,1056), (eclipse,980), (xcode,557), (netbeans,207), (vim,192), (intellij,155), (emacs,148), (sublime,137), (ipython,72), (notepad++,39), (phpstorm,32), (pycharm,22), (rstudio,17), (rubymine,14), (atom,11)

val year = 2014
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 1258823
// (php,5827), (java,4970), (javascript,4231), (python,4142), (sql,3797), (html,2907), (c#,2641), (css,2104), (c++,2030), (angularjs,1523), (r,1268), (ruby,970), (swift,720), (matlab,698), (vba,621), (bash,515), (shell,478), (scala,370), (go,368), (vb.net,351), (nodejs,314), (objective-c,179), (spark,103)
// (android,6779), (visual,1044), (eclipse,997), (xcode,738), (intellij,237), (netbeans,199), (vim,157), (sublime,152), (emacs,129), (ipython,67), (phpstorm,59), (notepad++,53), (pycharm,44), (atom,21), (rstudio,18), (rubymine,7)
// apache took over spark since 2012, aws spark release on 2013, and spark 1.0.0 release on 2014

val year = 2015
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 1352407
// (php,5318), (java,5224), (python,5151), (javascript,4078), (sql,4049), (c#,2954), (html,2942), (c++,1936), (css,1921), (angularjs,1868), (swift,1805), (r,1755), (ruby,924), (vba,880), (matlab,684), (bash,578), (shell,500), (spark,458), (nodejs,434), (scala,429), (go,426), (vb.net,395), (objective-c,178)
// (android,7246), (visual,1281), (eclipse,849), (xcode,815), (intellij,262), (netbeans,162), (vim,161), (sublime,138), (ipython,105), (emacs,94), (phpstorm,69), (pycharm,65), (notepad++,63), (rstudio,48), (atom,37), (rubymine,14)

val year = 2016
explodedWords.filter($"year" === year).count 
getListCountForYear(explodedWords, year, languageList).sortBy(-_._2)
getListCountForYear(explodedWords, year, editorList).sortBy(-_._2) 
// 1261017
// (python,4936), (php,4543), (java,4091), (javascript,3592), (sql,3490), (c#,2877), (html,2536), (swift,1763), (r,1705), (c++,1577), (css,1549), (angularjs,1501), (vba,898), (spark,733), (ruby,727), (matlab,565), (nodejs,496), (bash,473), (scala,457), (shell,441), (go,404), (vb.net,279), (objective-c,124)
// (android,6508), (visual,1270), (eclipse,650), (xcode,571), (intellij,267), (netbeans,156), (sublime,119), (vim,110), (pycharm,105), (ipython,67), (phpstorm,60), (emacs,52), (rstudio,52), (atom,40), (notepad++,31), (rubymine,6)

/* Word embedding */
val word2vecData = filteredWords.select($"filtered".as("text"))
val word2vecRdd = word2vecData.rdd.map(_.getAs[Seq[String]](0))
val word2vecModel = new Word2Vec()
val learnedWord2vecModel = word2vecModel.fit(word2vecRdd)
val interestingWord = "Javascript" // word to be checked
val numSynonyms = 10 // interesting synonyms
val synonyms = learnedWord2vecModel.findSynonyms(interestingWord, numSynonyms)
for((synonym, cosineSimilarity) <- synonyms) {
  println(s"$synonym $cosineSimilarity")
}

// hadoop
// fjajs, (cluster,0.6837189197540283), (cassandra,0.6705034375190735), (cloudera,0.6604753732681274), (hdfs,0.6499502658843994), (mahout,0.6408772468566895)

// spark
// (pyspark,0.7449303269386292), (rdd,0.7172156572341919), (flink,0.713650643825531), (hadoop,0.7105602025985718), (mapreduce,0.7035561800003052), (dstream,0.7002890110015869), (hivecontext,0.690696120262146), (spark?,0.6793029308319092), (rdds,0.6766228079795837), (pig,0.6725078225135803)

// function
// (function?,0.8123030662536621), (functions,0.7977750301361084), (funcion,0.641543447971344), (prepare(),0.634732723236084), (connection(),0.6249853372573853), (fetch_assoc(),0.6142003536224365), (method,0.6106415390968323), (funtion,0.6097581386566162), (bind_param(),0.6077978014945984), (subroutine,0.59543377161026)

// ipython
// (jupyter-notebook,0.8724756836891174), (anaconda,0.8624472618103027), (jupyter,0.8393844962120056), (pycharm,0.8026067018508911), (ipython-notebook,0.8015119433403015), (pypy,0.8000865578651428), (pip,0.7962930202484131), (qtconsole,0.7933409214019775), (python-2.6,0.7872650623321533), (python-3.4,0.7860600352287292)

// rstudio
// (xtable,0.9140763282775879), (rmarkdown,0.9080970883369446), (shinyapps,0.8909071683883667), (knitr,0.8892173767089844), (sweave,0.8850757479667664), (shiny-server,0.8790168762207031), (shinydashboard,0.8787872791290283), (ggvis,0.8709255456924438), (hmisc,0.855720579624176), (texmaker,0.8556791543960571)

/*** Tag analysis***/

/* load tag data*/
val rawTags = sqlContext.read.format("csv").option("header","true").load("/home/zluxyoe/Tags.csv")
val tags = rawTags.withColumn("id", $"Id").withColumn("tag", $"Tag")

/* Most popular tag */
tags.groupBy($"tag").count.sort($"count".desc).show
// +-------------+------+                                                          
// |          tag| count|
// +-------------+------+
// |   javascript|124155|
// |         java|115212|
// |           c#|101186|
// |          php| 98808|
// |      android| 90659|
// |       jquery| 78542|
// |       python| 64601|
// |         html| 58976|
// |          c++| 47591|
// |          ios| 47009|
// |        mysql| 42464|
// |          css| 42308|
// |          sql| 35782|
// |      asp.net| 29970|
// |  objective-c| 26922|
// |ruby-on-rails| 25789|
// |         .net| 24059|
// |            c| 23238|
// |       iphone| 21539|
// |    angularjs| 20345|
// +-------------+------+



/* grouped tags and tags word2vec */
val groupedTags = tags.groupBy($"id").agg(collect_list($"tag").as("text"))
val tagWord2VecRdd = groupedTags.select("text").rdd.map(_.getAs[Seq[String]](0))
val tagWord2VecModel = new Word2Vec()
val learnedTagWord2vecModel = tagWord2VecModel.fit(tagWord2VecRdd)
val tagSynonyms = learnedTagWord2vecModel.findSynonyms("android", 10)

// android
// android-intent,0.715316116809845), (android-fragments,0.7032397985458374), (android-listview,0.6958454251289368), (android-activity,0.690249502658844), (android-viewpager,0.6767613291740417), (fragment,0.6744683384895325), (android-imageview,0.6636172533035278), (android-manifest,0.6500732898712158), (android-layout,0.6418580412864685), (android-widget,0.6348060369491577)

for((synonym, cosineSimilarity) <- tagSynonyms) { println(s"$synonym $cosineSimilarity") }
















