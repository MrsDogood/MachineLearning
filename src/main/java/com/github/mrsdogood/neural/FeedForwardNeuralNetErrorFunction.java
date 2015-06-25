package com.github.mrsdogood.neural;

import com.github.mrsdogood.hessianfree.Gradientable;

import java.util.Vector;

import org.ejml.data.RowD1Matrix64F;

public class FeedForwardNeuralNetErrorFunction implements Gradientable{
    private FeedForwardNeuralNet nn;
    private Vector<double[]> trainingInputs, trainingOutputs;
    public FeedForwardNeuralNetErrorFunction(FeedForwardNeuralNet nn){
        this.nn = nn;
        trainingInputs = new Vector<double[]>();
        trainingOutputs = new Vector<double[]>();
    }

    public void addTrainingSet(double[] in, double[] out){
        assert(nn.getInputSize()==in.length);
        assert(nn.getOutputSize()==out.length);
        trainingInputs.add(in);
        trainingOutputs.add(out);
    }

    public int dim(){
        return nn.getNumWeights();
    }

    public double evaluate(RowD1Matrix64F x){
        nn.setWeights(x);
        double totalError = 0;
        int trainingSets = trainingInputs.size();
        for(int i = 0; i < trainingSets; i++){
            double[] expOutput = trainingOutputs.get(i);
            nn.getInputLayer().setData(trainingInputs.get(i));
            nn.propagate();
            RowD1Matrix64F actualOutput = nn.getOutputLayer();
            double error = 0;
            for(int j = 0; j < expOutput.length; j++){
                double e = actualOutput.get(j)-expOutput[j];
                error += e*e;
            }
            totalError += error;
        }
        totalError *= 0.5; // makes derivative easier
        return totalError;
    }

    public void gradient(RowD1Matrix64F x, RowD1Matrix64F out){
        nn.setWeights(x);
        // tare the output
        for(int i = 0; i < out.getNumElements(); i++)
            out.set(i,0);
        int trainingSets = trainingInputs.size();
        for(int i = 0; i < trainingSets; i++){
            nn.getInputLayer().setData(trainingInputs.get(i));
            nn.propagate();
            nn.initBackprop();
            double[] expOutput = trainingOutputs.get(i);
            RowD1Matrix64F actualOutput = nn.getOutputLayer();
            for(int o = 0; o < expOutput.length; o++){
                for(int w = 0; w < nn.getNumWeights(); w++){
                    out.set(w, out.get(w)+
                        (actualOutput.get(o)-expOutput[o])*nn.backprop(w,o)
                    );
                }
            }
        }
    }
}
