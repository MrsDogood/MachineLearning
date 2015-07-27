package com.github.mrsdogood.example;

import com.github.mrsdogood.hessianfree.*;
import com.github.mrsdogood.neural.*;

import org.ejml.data.*;

import java.util.*;
import java.io.*;

public class MNISTTrainer{
    static Random r = new Random();
    static final int DEFAULT_TRAINING_ITERATIONS = 100;
    static final double DEFAULT_LEARNING_RATE = 1.0;
    static final double DEFAULT_NEGATIVE_SAMPLE_RATE = 0.2;
    
    private static void printHelpAndDie(){
        System.out.println(
            "Usage: <program> <nn_file> <digit> \\\n"+
            "\t(<training_iterations>) (<learning_rate>) (<negative_sample_rate>) \\\n"+
            "\t(<rand_seed>)");
        System.out.println("nn_file is the file to store the neural net weights in");
        System.out.println("digit is an integer 0-9");
        System.out.println("training_iterations default is "+DEFAULT_TRAINING_ITERATIONS);
        System.out.println("learning_rate default is "+DEFAULT_LEARNING_RATE);
        System.out.println("negative_sample_rate default is "+DEFAULT_NEGATIVE_SAMPLE_RATE);
        System.out.println("rand_seed will default to a randomly generated value");
        System.exit(1);
    }
    public static void main(String[] args){
        //parse args
        String filename = null;
        int digit = -1;
        int trainingIterations = DEFAULT_TRAINING_ITERATIONS;
        double learningRate = DEFAULT_LEARNING_RATE;
        double negativeSampleRate = DEFAULT_NEGATIVE_SAMPLE_RATE;
        long randSeed = r.nextLong();
        try{
            int arg = 0;
            filename = args[arg++];
            digit = Integer.parseInt(args[arg++]);
            if(arg < args.length)
                trainingIterations = Integer.parseInt(args[arg++]);
            if(arg < args.length)
                learningRate = Double.parseDouble(args[arg++]);
            if(arg < args.length)
                negativeSampleRate = Double.parseDouble(args[arg++]);
            if(arg < args.length)
                randSeed = Long.parseLong(args[arg++]);
            if(arg != args.length)
                throw new RuntimeException("Too many arguments.");
        } catch (Throwable t){
            System.out.println(t);
            printHelpAndDie();
        }

        StringBuilder trainingReport = new StringBuilder();
        trainingReport.append("digit = "+digit+"\n");
        trainingReport.append("training_iterations = "+trainingIterations+"\n");
        trainingReport.append("learning_rate = "+learningRate+"\n");
        trainingReport.append("negative_sample_rate = "+negativeSampleRate+"\n");
        trainingReport.append("rand_seed = "+randSeed+"\n");

        // Read or Create a neural net.
        r = new Random(randSeed);
        File file = new File(filename);
        FeedForwardNeuralNet nn = null;
        Vector<String> trainingRecords = null;
        if(file.exists()){
            System.out.println("Reading neural net...");
            try{
                FileInputStream fis = new FileInputStream(file);
                ObjectInputStream ois = new ObjectInputStream(fis);
                trainingRecords = (Vector<String>)ois.readObject();
                nn = (FeedForwardNeuralNet)ois.readObject();
                ois.close();
            } catch (Throwable t) {
                t.printStackTrace();
                System.exit(1);
            }
        } else {
            System.out.println("File not found. Creating neural net...");
            nn = new FeedForwardNeuralNet(r, 28*28, 28, 1);
            trainingRecords = new Vector<String>();
        }
        DenseMatrix64F initialWeights = nn.getWeights();

        // Print training history
        System.out.println("\n#### Training History ####\n");
        for(String record : trainingRecords)
            System.out.println(record);

        // Setup an error function to measure how good/bad the neural net is
        // based on the training data.
        System.out.println("Loading training data...");
        MNISTErrorFunction errorFunction = new MNISTErrorFunction(
            nn, digit, false, negativeSampleRate, r);

        double preTrainingError = errorFunction.evaluate(nn.getWeights());
        System.out.println("Pre-training error: "+preTrainingError);
        errorFunction.enableStochastic(100, r);

        // Create and use an optimizer to minimize the error function
        // i.e. optimize the neural net
        StochasticGradientDescentOptimizer optimizer = 
            new StochasticGradientDescentOptimizer(
            errorFunction, initialWeights, learningRate);
        System.out.println("Training...");
        int eachPercent = (int)Math.ceil(trainingIterations/100.0);
        int remaining = trainingIterations;
        for(double i = 1; i<=100; i++){
            int iterations = Math.min(eachPercent, remaining); 
            if(iterations > 0){
                remaining-=iterations;
                optimizer.optimize(iterations);
            }
            System.out.print("\r"+i+"%");
        }
        System.out.println();
        nn.setWeights(optimizer.getCurrentBest());

        errorFunction.disableStochastic();
        double postTrainingError = errorFunction.evaluate(nn.getWeights());
        System.out.println("Post-training error: "+postTrainingError);
        String errorReport = errorFunction.getErrorReport(nn.getWeights());
        trainingReport.append(errorReport);
        trainingRecords.add(trainingReport.toString());
        System.out.println(trainingReport);

        // Save if training was better.
        if(postTrainingError < preTrainingError || !file.exists()){
            System.out.println("Saving...");
            try{
                FileOutputStream fos = new FileOutputStream(file);
                ObjectOutputStream oos = new ObjectOutputStream(fos);
                oos.writeObject(trainingRecords);
                oos.writeObject(nn);
                oos.close();
            } catch (Throwable t){
                throw new RuntimeException(t);
            }
        } else {
            System.out.println("Training didn't improve the neural net so will "+
                "not overwrite existing file.");
        }
    }
}
