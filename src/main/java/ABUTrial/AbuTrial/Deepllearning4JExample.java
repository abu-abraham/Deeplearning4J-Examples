package ABUTrial.AbuTrial;

import org.datavec.api.split.FileSplit;
import org.apache.commons.lang.ArrayUtils;
/**
 * Hello world!
 *
 */
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.deeplearning4j.eval.Evaluation;

public class Deepllearning4JExample {
  private static final Logger LOGGER = LoggerFactory.getLogger(Deepllearning4JExample.class);
private static final Row CHAR_TO_INT = null;

  public static void main(String[] args) {
    Deepllearning4JExample a = new Deepllearning4JExample();
    try {
		a.deepL();
	} catch (Exception e) {
		e.printStackTrace();
	} 
  }

  public void deepL() throws IOException, InterruptedException {
      	String master = "local[*]";

	    SparkConf conf1 = new SparkConf()
	        .setAppName(Deepllearning4JExample.class.getName())
	        .setMaster(master);
	    JavaSparkContext context = new JavaSparkContext(conf1);
	    
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
	    VectorAssembler assembler = new VectorAssembler()
	    	      .setInputCols(featureArray)
	    	      .setOutputCol("features");
	    Dataset<Row> transformedTrainingData = assembler.transform(trainingData);
	    Dataset<Row> transformedTestingData = assembler.transform(testData);
	   
	   
	    int outputNum = 1; // Number of possible outcomes (e.g. labels 0 through 9).
	    int rngSeed = 6; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later.
	    int numEpochs = 15;
	   
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.06) //specify the learning rate
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(featureArray.length)
                        .nOut(10)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(10).nOut(6).build())
                .layer(2, new OutputLayer.Builder(LossFunction.MEAN_ABSOLUTE_ERROR) //create hidden layer
                        .nIn(6)
                        .nOut(outputNum)
                        .activation(Activation.TANH)
                        .build())
                .backprop(true).pretrain(false)
                .build();

	    TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(10)    
	            .averagingFrequency(5)
	            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
	            .batchSizePerWorker(10)
	            .build();
	    
	    DataSet trainingDataSet = getDataAsDataSet(transformedTrainingData,trainingData,featureArray);
	    List<DataSet> trainingDataSetList =trainingDataSet .asList();
	    DataSet testingDataSet = getDataAsDataSet(transformedTestingData, testData, featureArray);
	    List<DataSet> testingDataSetList = testingDataSet.asList();
	    
	    
	    
	    SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(context, conf, tm);
        //Execute training:
	    JavaRDD<DataSet> trainData = context.parallelize(trainingDataSetList);
	    JavaRDD<DataSet> testtData = context.parallelize(testingDataSetList);
	    DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingDataSet);           
        normalizer.transform(trainingDataSet); 
        normalizer.transform(testingDataSet);
	    System.out.println("Fitting to model");
	    MultiLayerNetwork n = null;
        for (int i = 0; i < numEpochs; i++) {
           n = sparkNet.fit(trainData);
        }
        Evaluation evaluation = sparkNet.doEvaluation(testtData, 64, new Evaluation(2))[0];
        System.out.println(evaluation.stats());
        System.out.println(testtData.count());
        tm.deleteTempFiles(context);
  }
  

public DataSet getDataAsDataSet(Dataset<Row> sparkSqlModifiedDataset, Dataset<Row> sparkSqlDataset, String[] featureArray) {
		int rows = (int) sparkSqlModifiedDataset.count();
	    double[][] data = new double[rows][featureArray.length];
	    ArrayList<Double[]> tempList = new ArrayList<Double[]>();
	    int j = 0;
	    List<String> featureLists = sparkSqlModifiedDataset.select("features").map(row -> row.toString(), Encoders.STRING()).collectAsList();
	    for(String u:featureLists) {
	    	u = u.replace("[", "");
	    	u = u.replace("]", "");
	    	System.out.println(u);
	    	List<String> elephantList = Arrays.asList(u.split(","));
	    	Double[] elephantNumbers = new Double[featureArray.length];
	    	int i = 0;
	    	for (String el : elephantList) {	    		
	    		elephantNumbers[i] = Double.parseDouble(el);
	    		i+=1;
	    	}
	    	tempList.add(elephantNumbers);
	    }
	    System.out.println(tempList.size());
	    for (Double[] ds : tempList) {
			data[j] = ArrayUtils.toPrimitive(ds);
			j+=1;
		}
	    INDArray featuresINDArray = Nd4j.create(data);
	    
	   	    
	    data = new double[rows][1];
	    j = 0;
	    tempList = new ArrayList<Double[]>();
	    featureLists = sparkSqlDataset.select("Test_Tag (0-OK, 1-Fail)").map(row -> row.toString(), Encoders.STRING()).collectAsList();
	    for(String u:featureLists) {
	    	u = u.replace("[", "");
	    	u = u.replace("]", "");
	    	List<String> elephantList = Arrays.asList(u.split(","));
	    	Double[] elephantNumbers = new Double[1];
	    	int i = 0;
	    	for (String el : elephantList) {	    		
	    		elephantNumbers[i] = Double.parseDouble(el);
	    		i+=1;
	    	}
	    	tempList.add(elephantNumbers);
	    }
	    for (Double[] ds : tempList) {
			data[j] = ArrayUtils.toPrimitive(ds);
			j+=1;
		}
	    
	    INDArray labelsINDArray =  Nd4j.create(data);
	    return new DataSet(featuresINDArray,labelsINDArray);
  } 
  
}