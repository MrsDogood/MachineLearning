package com.github.mrsdogood.neural;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.zip.GZIPInputStream;
import java.util.Random;

import org.ejml.data.RowD1Matrix64F;

import static com.github.mrsdogood.hessianfree.Utils.copy;

public class MNISTErrorFunction extends FeedForwardNeuralNetErrorFunction {
    private static final String TRAINING_FILE = "extern/dat/train_data.gz";
    private static final String TESTING_FILE = "extern/dat/test_data.gz";
    private String filename;
    private int digit;
    double[][][] trainingData;
    double negativeSampleRate;
    public MNISTErrorFunction(FeedForwardNeuralNet nn, int digit, boolean testing,
        double negativeSampleRate, Random r){
        super(nn);
        assert(nn.getInputSize()==28*28);
        assert(nn.getOutputSize()==1);
        assert(0 <= digit);
        assert(digit <= 9);
        this.digit = digit;
        this.negativeSampleRate = negativeSampleRate;
        this.filename = testing ? TESTING_FILE : TRAINING_FILE;
        init(r);
    }

    private void init(Random r){
        // read training data
        try {
            FileInputStream fis = new FileInputStream(filename);
            GZIPInputStream zis = new GZIPInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(zis);
            trainingData = (double[][][])ois.readObject();
            ois.close();
        } catch (Throwable t) {
            System.out.println("default path: "+new java.io.File(".").getAbsolutePath());
            throw new RuntimeException(t);
        }
        // save training data
        assert(trainingData.length == 10);
        for(int i = 0; i < 10; i++){
            double[] output = new double[]{i==digit?1:0};
            //System.out.println(trainingData[i].length+" "+i+"'s");
            for(int j = 0; j < trainingData[i].length; j++){
                if(i==digit || r.nextDouble()<=negativeSampleRate)
                    addTrainingSet(trainingData[i][j], output);
            }
        }
    }

    public String getErrorReport(RowD1Matrix64F x){
        nn.setWeights(x);
        double[] avgAct = new double[10];
        for(int d = 0; d < 10; d++){
            for(int i = 0; i < trainingData[d].length; i++){
                double act = nn.evaluate(trainingData[d][i])[0];
                avgAct[d]+=act;
            }
            avgAct[d]/=trainingData[d].length;
        }

        StringBuilder out = new StringBuilder();
        out.append("Average activations by digit:\n");
        for(int d = 0; d<10; d++)
            out.append("\t"+d+") "+avgAct[d]+"\n");
        return out.toString();
    }
}
