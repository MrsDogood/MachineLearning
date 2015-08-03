package com.github.mrsdogood.neural;

import java.io.Serializable;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;

public abstract class NeuralLayer implements Serializable {
    private static final long serialVersionUID = -19939757720L;
    DenseMatrix64F preactivations, activations;
    ActivationFunction actFunc;
    NeuralLayer prev, next;
    int totalWeights = 0;
    int layerId = 0;

    // only the input layer's activation layer may be null
    public NeuralLayer(int size, ActivationFunction actFunc){
        this.activations = new DenseMatrix64F(size, 1);
        this.preactivations = new DenseMatrix64F(size, 1);
        this.actFunc = actFunc;
    }

    public int size(){
        return activations.getNumElements();
    }

    // Get the change in the o'th neuron of this layer
    // with respect to the w'th weight.
    double backprop(int w, int o){
        return prev.deval(w,o);
    }

    // to be called from the input layer
    abstract void propagate();
    abstract void push(NeuralLayer next, RowD1Matrix64F weights);
    // The change in the next layer's n'th neuron per unit of change
    // of some previous layer's  weight, with identifier w
    protected abstract double deval(int w, int n);
}
