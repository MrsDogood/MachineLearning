package com.github.mrsdogood.neural;

import com.github.mrsdogood.hessianfree.StochasticGradientable;
import com.github.mrsdogood.math.LCG;

import java.util.Vector;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

import org.ejml.data.RowD1Matrix64F;

import static com.github.mrsdogood.hessianfree.Utils.copy;

public class FeedForwardNeuralNetErrorFunction implements StochasticGradientable, Iterable<Integer>{
    protected FeedForwardNeuralNet nn;
    protected Vector<double[]> trainingInputs, trainingOutputs;
    private boolean stochastic = false;
    private int stochasticBatchSize = 1;
    private LCG stochasticGenerator;
    public FeedForwardNeuralNetErrorFunction(FeedForwardNeuralNet nn){
        this.nn = nn;
        trainingInputs = new Vector<double[]>();
        trainingOutputs = new Vector<double[]>();
    }

    public void enableStochastic(int batchSize, Random r){
        if(stochastic)
            throw new RuntimeException("Stochastic already enabled!");
        stochastic = true;
        stochasticBatchSize = batchSize;
        stochasticGenerator = LCG.makeLCG(trainingInputs.size(), r);
    }
    public void disableStochastic(){
        stochastic = false;
    }
    public boolean stochasticEnabled(){
        return stochastic;
    }

    public void step(){
        if(!stochastic)
            throw new UnsupportedOperationException("Stochastic updates were not enabled!");
        stochasticGenerator.next(stochasticBatchSize);
    }
    public Iterator<Integer> iterator(){
        if(stochastic)
            return new StochasticIterator(
                stochasticGenerator, stochasticBatchSize, trainingInputs.size());
        else
            return new NonStochasticIterator(trainingInputs.size());
    }

    public void addTrainingSet(double[] in, double[] out){
        if(stochastic)
            throw new RuntimeException("Cannot add to training set after stochastic is enabled!");
        assert(nn.getInputSize()==in.length);
        assert(nn.getOutputSize()==out.length);
        trainingInputs.add(in);
        trainingOutputs.add(out);
    }

    public int dim(){
        return nn.getNumWeights();
    }

    public double evaluate(RowD1Matrix64F x){
        try{
            nn.setWeights(x);
            double totalError = 0;
            int trainingSets = trainingInputs.size();
            for(int i : this){
                if(i<0){
                    System.out.println(stochasticGenerator);
                    System.out.println(i);
                }
                double[] expOutput = trainingOutputs.get(i);
                copy(trainingInputs.get(i), nn.getInputLayer());
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
        } catch(Utils.SigNaNException e){
            System.err.println("Weights in error:\n"+x);
            throw e;
        }
    }

    public void gradient(RowD1Matrix64F x, RowD1Matrix64F out){
        nn.setWeights(x);
        // tare the output
        for(int i = 0; i < out.getNumElements(); i++)
            out.set(i,0);
        int trainingSets = trainingInputs.size();
        for(int i : this){
            copy(trainingInputs.get(i), nn.getInputLayer());
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

class NonStochasticIterator implements Iterator<Integer>{
    private int i, bound;
    public NonStochasticIterator(int bound){   
        i = 0;
        this.bound = bound;
    }
    public boolean hasNext(){
        return i<bound;
    }
    public Integer next(){
        if(hasNext())
            return i++;
        throw new NoSuchElementException();
    }
    public void remove(){
        throw new UnsupportedOperationException();
    }
}

class StochasticIterator implements Iterator<Integer>{
    private int i, batchSize, mod;
    private LCG generator;
    public StochasticIterator(LCG generator, int batchSize, int mod){
        i = 0;
        this.mod = mod;
        this.batchSize = batchSize;
        this.generator = generator;
    }
    public boolean hasNext(){
        return i<batchSize;
    }
    public Integer next(){
        if(hasNext())
            return generator.poll(i++)%mod;
        throw new NoSuchElementException();
    }
    public void remove(){
        throw new UnsupportedOperationException();
    }
}
