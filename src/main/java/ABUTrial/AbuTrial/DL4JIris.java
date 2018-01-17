package ABUTrial.AbuTrial;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class DL4JIris {
	
	 public static void main(String[] args) {
		    DL4JIris a = new DL4JIris();
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
	    
	    Dataset<Row> df = sqlContext.read().format("com.databricks.spark.csv").option("header", true).option("inferSchema",true).load("iris.csv").toDF();
	    
	    df = df.na().fill("0");
	    String[] y = df.columns();
	    
	    
	    String[] featureArray = Arrays.copyOfRange(y, 1, y.length); 

	    StringIndexer indexer = new StringIndexer().setInputCol("species").setOutputCol("class");
	    df = indexer.fit(df).transform(df);
	    

	    for(String x: df.columns()) {
	    	df = df.withColumn(x,
	    			df.col(x).cast(DataTypes.DoubleType)); 
	    }
	    
	    Dataset<Row>[] splits = df.randomSplit(new double[] {0.7, 0.3});
	    
	    Dataset<Row> trainingData = splits[0];
	    Dataset<Row> testData = splits[1];
	    
	    VectorAssembler assembler = new VectorAssembler()
	    	      .setInputCols(featureArray)
	    	      .setOutputCol("features");
	    Dataset<Row> td = assembler.transform(trainingData);
	    Dataset<Row> ts = assembler.transform(testData);
	   
	   
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
	            .batchSizePerWorker(2)
	            .build();
	    
	    DataSet scet = getDataAsDataSet(td,trainingData,featureArray);
	    List<DataSet> traind =scet .asList();
	    DataSet scetest = getDataAsDataSet(ts, testData, featureArray);
	    List<DataSet> testdd =scetest.asList();
	    
	    
	    
	    SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(context, conf, tm);
        //Execute training:
	    JavaRDD<DataSet> trainData = context.parallelize(traind);
	    JavaRDD<DataSet> testtData = context.parallelize(testdd);
	    DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(scet);           
        normalizer.transform(scet); 
        normalizer.transform(scetest);
	    System.out.println("Fitting to model");
	    MultiLayerNetwork n = null;
        for (int i = 0; i < numEpochs; i++) {
           n = sparkNet.fit(trainData);
        }
        Evaluation evaluation = sparkNet.doEvaluation(trainData, 64, new Evaluation(3))[0];
        System.out.println(evaluation.stats());
        System.out.println(testtData.count());
        tm.deleteTempFiles(context);
  }
  

public DataSet getDataAsDataSet(Dataset<Row> td, Dataset<Row> trainingData, String[] featureArray) {
	  int rows = (int) td.count();
	    double[][] data = new double[rows][featureArray.length];
	    ArrayList<Double[]> tempList = new ArrayList<Double[]>();
	    int j = 0;
	    List<String> listTwo = td.select("features").map(row -> row.toString(), Encoders.STRING()).collectAsList();
	    for(String u:listTwo) {
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
	    INDArray first = Nd4j.create(data);
	    
	   	    
	    double[][] dataa = new double[rows][1];
	    j = 0;
	    tempList = new ArrayList<Double[]>();
	    listTwo = trainingData.select("species").map(row -> row.toString(), Encoders.STRING()).collectAsList();
	    for(String u:listTwo) {
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
			dataa[j] = ArrayUtils.toPrimitive(ds);
			j+=1;
		}
	    
	    INDArray second =  Nd4j.create(dataa);
	    return new DataSet(first,second);
  } 
  
}
