package com.github.mrsdogood.example;

import com.github.mrsdogood.hessianfree.*;
import com.github.mrsdogood.neural.*;

import org.ejml.data.*;

import java.util.*;


public class SimpleExample{
    // Training data
    // The neural net will be given this data and should learn to reproduce the
    // outputs given the inputs.
    public static final double[][] INPUTS = {
        {0, 0, 0},
        {1, 1, 0},
        {0, 1, 1},
        {0, 0, 1}
    };
    public static final double[][] OUTPUTS = {
        {0},
        {1},
        {1},
        {0}
    };

    public static void main(String[] args){
        // Create a neural net.
        Random r = new Random();
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 3, 3, 1);
        DenseMatrix64F initialWeights = nn.getWeights();

        // Setup an error function to measure how good/bad the neural net is
        // based on the training data.
        FeedForwardNeuralNetErrorFunction errorFunction =
            new FeedForwardNeuralNetErrorFunction(nn);
        for(int i = 0; i < INPUTS.length; i++)
            errorFunction.addTrainingSet(INPUTS[i], OUTPUTS[i]);

        System.out.println("Pre-training error: "+
            errorFunction.evaluate(nn.getWeights()));

        // Create and use an optimizer to minimize the error function
        // i.e. optimize the neural net
        final double learningRate = 5;
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer(
            errorFunction, initialWeights, learningRate);
        final int optimizeTimes = 50;
        optimizer.optimize(optimizeTimes);
        nn.setWeights(optimizer.getCurrentBest());

        System.out.println("Post-training error: "+
            errorFunction.evaluate(nn.getWeights()));

        // Manual Tests
        Scanner kb = new Scanner(System.in);
        while(true){
            double[] in = new double[3];
            System.out.print("Input: ");
            for(int i = 0; i < in.length; i++)
                in[i] = kb.nextDouble();
            System.out.println("Output: "+nn.evaluate(in)[0]);
        }
    }
}
