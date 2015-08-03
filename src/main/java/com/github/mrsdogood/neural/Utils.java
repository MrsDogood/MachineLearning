package com.github.mrsdogood.neural;

import org.ejml.data.RowD1Matrix64F;

public final class Utils {

    public static class SigNaNException extends RuntimeException{};

    public static void activate(ActivationFunction func, RowD1Matrix64F in, 
        RowD1Matrix64F out){
        assert(in.getNumRows()==out.getNumRows());
        assert(in.getNumCols()==out.getNumCols());
        int size = in.getNumRows()*out.getNumCols();
        for(int i = 0; i < size; i++){
            out.set(i, func.f(in.get(i)));
        }
    }

    /** the sigmoid function **/
    public static final double sig(double x) {
        if(Double.isNaN(x)){
            System.err.println("sig not a number.");
            throw new SigNaNException();
        }
        double ret = 1.0/(1.0+Math.exp(-x));
        if(Double.isNaN(ret)){
            System.err.println("sig returning not a number.");
            throw new SigNaNException();
        }
        return ret;
    }

    public static final void sig(RowD1Matrix64F in, RowD1Matrix64F out) {
        assert(in.getNumRows()==out.getNumRows());
        assert(in.getNumCols()==out.getNumCols());
        int size = in.getNumRows()*out.getNumCols();
        for(int i = 0; i < size; i++){
            out.set(i, sig(in.get(i)));
        }
    }

    /** the derivative of the sigmoid function **/
    public static final double dsig(double x) {
        if(Double.isNaN(x)){
            System.err.println("dsig not a number.");
            throw new SigNaNException();
        }
        double sig = sig(x);
        double ret = sig*(1-sig);
        if(Double.isNaN(ret)){
            System.err.println("dsig returning not a number from:"+x);
            throw new SigNaNException();
        }
        return ret;
    }

    public static final void dsig(RowD1Matrix64F in, RowD1Matrix64F out) {
        assert(in.getNumRows()==out.getNumRows());
        assert(in.getNumCols()==out.getNumCols());
        int size = in.getNumElements();
        for(int i = 0; i < size; i++){
            out.set(i, dsig(in.get(i)));
        }
    }
}
