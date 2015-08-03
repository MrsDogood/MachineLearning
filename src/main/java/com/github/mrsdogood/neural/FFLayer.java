package com.github.mrsdogood.neural;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;

import static org.ejml.ops.CommonOps.mult;
import static com.github.mrsdogood.neural.Utils.activate;
import static com.github.mrsdogood.hessianfree.Utils.copy;

public class FFLayer extends NeuralLayer {
    private static final long serialVersionUID = -19939757721L;
    // weights are the connection weights to the next layer
    DenseMatrix64F weights;

    public FFLayer(int size, ActivationFunction actFunc){
        super(size, actFunc);
    }

    void push(NeuralLayer next, RowD1Matrix64F weights){
        assert(next.prev==null);
        assert(this.next==null);
        assert(this.weights==null);
        next.prev = this;
        this.next = next;
        this.weights = new DenseMatrix64F(next.size(), size());
        copy(weights, this.weights);
        next.layerId = layerId+1;
        totalWeights += weights.getNumElements();
        next.totalWeights = totalWeights;
    }

    // to be called from the input layer
    void propagate(){
        if(next==null)
            return;
        // n.pa(i) = sum(k: w(i,k)*a(k))
        mult(weights, activations, next.preactivations);
        // n.a(i) = n.f( n.pa(i) )
        activate(next.actFunc, next.activations, next.activations);
        next.propagate();
    }

    // The change in the next layer's n'th neuron per unit of change
    // of some previous layer's  weight, with identifier w
    protected double deval(int w, int n){
        assert(w<totalWeights);
        int nWeights = weights.getNumElements();
        if(w<nWeights){ 
            // if w is in this layer
            int wr = w%weights.getNumRows();
            int wc = w/weights.getNumRows();
            return deval(wr, wc, n);
        }else{ 
            // else w is in a previous layer
            return backprop_deval(w-nWeights, n);
        }
    }

    // The change in the next layer's n'th neuron per unit of change
    // of this layer's weight at row wr and column wc in it's
    // weight matrix.
    private double deval(int wr, int wc, int n){
        if(wr!=n)
            return 0;
        // n.a(wr) == n.f( sum(k: w(wr,k)*a(k)) )
        // n.a'(wr) == n.df( sum(k: w(wr,k)*a(k)) ) * w(wr,wc)*a(wc)
        //          == n.df( n.pa(wr) ) * w(wr,wc)*a(wc)
        return next.actFunc.df(next.preactivations.get(wr))*
            weights.get(wr,wc)*activations.get(wc);
    }

    // The change in the next layer's n'th neuron per unit of change
    // of some previous layer's  weight, with identifier w
    private double backprop_deval(int w, int n){
        assert(prev!=null);
        //n.a(n) == n.f( sum(k: w(n,k)*a(k)) )
        //n.a'(n) == n.df( sum(k: w(n,k)*a(k)) ) * sum(k: w(n,k)*a'(k))
        //        == n.df( n.pa(n) ) * sum(k: w(n,k)*a'(k))
        double sum = 0;
        int maxK = next.size();
        for(int k = 0; k < maxK; k++)
            sum+= weights.get(n,k)*prev.deval(w,k);
        return next.actFunc.df(next.preactivations.get(n))*sum;
    }
}
