package ABUTrial.AbuTrial;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

public class RandomForestRegression {
	
		public static void main(String args[]) {
			rf();
		}
	 
		public static void rf() {
		  String master = "local[*]";

		    SparkConf conf = new SparkConf()
		        .setAppName(Deepllearning4JExample.class.getName())
		        .setMaster(master);
		    JavaSparkContext context = new JavaSparkContext(conf);

		    SQLContext sqlContext = new SQLContext(context);
		    
		    Dataset<Row> df = sqlContext.read().format("com.databricks.spark.csv").option("header", true).option("inferSchema",true).load("100day.csv").toDF();
		    df = df.na().fill("0");
		    Map<String,String> replacement = new HashMap<String,String>();
		    replacement.put("Null", "0");
		    String[] y = df.columns();
		    for(String x: y) {
		    	df = df.na().replace(x, replacement);   	
		    }
		        
		    
		    for(String x: df.columns()) {
		    	df = df.withColumn(x,
		    			df.col(x).cast(DataTypes.DoubleType)); 
		    }
		    
		       
		    Dataset<Row>[] splits = df.randomSplit(new double[] {0.7, 0.3});
		    
		    Dataset<Row> trainingData = splits[0];
		    Dataset<Row> testData = splits[1];
		    
		    
		    String[] featureArray = Arrays.copyOfRange(y, 1, y.length);
		    String[] labelArray = Arrays.copyOfRange(y, 0, 1);
		    VectorAssembler assembler = new VectorAssembler()
		    	      .setInputCols(featureArray)
		    	      .setOutputCol("features");
		    
		    RandomForestRegressor rf = new RandomForestRegressor()
		    		  .setLabelCol("Test_Tag (0-OK, 1-Fail)")
		    		  .setFeaturesCol("features");

		    		// Chain indexer and forest in a Pipeline
		    		Pipeline pipeline = new Pipeline()
		    		  .setStages(new PipelineStage[] {assembler, rf});

		    		// Train model. This also runs the indexer.
		    		PipelineModel model = pipeline.fit(trainingData);
		    		
		    		// Make predictions.
		    		Dataset<Row> predictions = model.transform(testData);

		    		// Select example rows to display.
		    		predictions.select("prediction", "Test_Tag (0-OK, 1-Fail)", "features").show(5);

		    		// Select (prediction, true label) and compute test error
		    		RegressionEvaluator evaluator = new RegressionEvaluator()
		    		  .setLabelCol("Test_Tag (0-OK, 1-Fail)")
		    		  .setPredictionCol("prediction")
		    		  .setMetricName("rmse");
		    		double rmse = evaluator.evaluate(predictions);
		    		System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);
		    		
	  }
}

