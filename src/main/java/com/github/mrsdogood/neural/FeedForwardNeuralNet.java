package com.github.mrsdogood.neural;

import java.util.Arrays;
import java.util.Random;
import java.util.HashMap;
import java.util.List;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.mult;
import static com.github.mrsdogood.neural.Utils.sig;
import static com.github.mrsdogood.neural.Utils.dsig;

public class FeedForwardNeuralNet {
    private DenseMatrix64F[] weights;
    private int numWeights;
    private DenseMatrix64F[] layers;
    // Memoized calculations for backpropigation
    private int[][] weightIndexToAddress;
    // presigLayers[i] = mult(weights[i], layers[i])
    // layers[i+1] = sig(presigLayers[i])
    private DenseMatrix64F[] presigLayers; 
    // dsigLayers[i] = dsig(presigLayers[i])
    private DenseMatrix64F[] dsigLayers;

    public FeedForwardNeuralNet(Random rand, int... layerSizes) {
        weights = new DenseMatrix64F[layerSizes.length-1];
        numWeights = 0;
        presigLayers = new DenseMatrix64F[layerSizes.length-1];
        dsigLayers = new DenseMatrix64F[layerSizes.length-1];
        layers = new DenseMatrix64F[layerSizes.length];
        // init presigLayers, dsigLayers, layers, weights, and numWeights
        for(int i = 0; i < layerSizes.length; i++){
            if(i < weights.length){
                weights[i] = RandomMatrices.createGaussian(
                    layerSizes[i+1], layerSizes[i], 0, 1, rand);
                numWeights+= weights[i].getNumElements();
                presigLayers[i] = new DenseMatrix64F(layerSizes[i+1], 1);
                dsigLayers[i] = new DenseMatrix64F(layerSizes[i+1], 1);
            }
            layers[i] = new DenseMatrix64F(layerSizes[i], 1);
        }
        //init weight addresses
        weightIndexToAddress = new int[numWeights][3];
        int i = 0;
        for(int wl = 0; wl < weights.length; wl++){
            for(int wr = 0; wr < weights[wl].getNumRows(); wr++){
                for(int wc = 0; wc < weights[wl].getNumCols(); wc++){
                    weightIndexToAddress[i][0] = wl;
                    weightIndexToAddress[i][1] = wr;
                    weightIndexToAddress[i][2] = wc;
                    i++;
                }
            }
        }
        assert(i==numWeights);
    }

    public double[] evaluate(double... inputs){
        getInputLayer().setData(inputs);
        propagate();
        return getOutputLayer().getData();
    }
    
    public int getInputSize(){
        return getInputLayer().getNumElements();
    }

    public int getOutputSize(){
        return getOutputLayer().getNumElements();
    }

    public DenseMatrix64F getInputLayer(){
        return layers[0];
    }

    public DenseMatrix64F getOutputLayer(){
        return layers[layers.length-1];
    }

    public void propagate(){
        for(int i = 1; i < layers.length; i++){
            mult(weights[i-1], layers[i-1], presigLayers[i-1]);
            try{
                sig(presigLayers[i-1], layers[i]);
            }catch(Utils.SigNaNException e){
                System.err.println("Presig Layer: \n"+ presigLayers[i-1]);
                throw e;
            }
        }
    }

    public double getWeight(int w){
        int[] addr = weightIndexToAddress[w];
        return weights[addr[0]].get(addr[1], addr[2]);
    }

    public void setWeight(int w, double weight){
        int[] addr = weightIndexToAddress[w];
        weights[addr[0]].set(addr[1], addr[2], weight);
    }

    public void setWeights(RowD1Matrix64F w){
        assert(w.getNumElements()==numWeights);
        for(int i = 0; i < numWeights; i++){
            setWeight(i, w.get(i));
        }
    }

    public DenseMatrix64F getWeights(){
        DenseMatrix64F matrix = new DenseMatrix64F(numWeights, 1);
        for(int i = 0; i < numWeights; i++)
            matrix.set(i, getWeight(i));
        return matrix;
    }

    public void initBackprop(){
        for(int i = 0; i < presigLayers.length; i++)
            dsig(presigLayers[i], dsigLayers[i]);
    }

    /**
    * Calculates the change in the output neuron o with respect to the weight w 
    * Requires a single forward propagation with the relevant input
    * layer and initBackprop();
    **/
    public double backprop(int w, int o){
        int[] addr = weightIndexToAddress[w];
        return deval(addr[0], addr[1], addr[2], layers.length-1, o);
    }

    public int getNumWeights(){
        return numWeights;
    }

    private double deval(int wl, int wr, int wc, int l, int n){
        assert(wl < l);
        double solution = 0;
        if(wl==l-1){
            if(wr==n){
                solution = dsigLayers[l-1].get(n,0)*layers[l-1].get(wc,0);
            }
            // else solution = 0;
        }else{
            int height = layers[l-1].getNumRows();
            for(int i = 0; i < height; i++){
                solution += deval(wl, wr, wc, l-1, i) * weights[l-1].get(n,i);
            }
            solution *= dsigLayers[l-1].get(n,0);
        }
        return solution;
    }
}
