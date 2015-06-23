package com.github.mrsdogood.neural;

import org.ejml.data.D1Matrix64F;

public class Utils {

    /** the sigmoid function **/
    public static final double sig(double x) {
        return 1.0/(1.0+Math.exp(-x));
    }

    public static final void sig(D1Matrix64F in, D1Matrix64F out) {
        assert(in.getNumRows()==out.getNumRows());
        assert(in.getNumCols()==out.getNumCols());
        int size = in.getNumRows()*out.getNumCols();
        for(int i = 0; i < size; i++){
            out.set(i, sig(in.get(i)));
        }
    }

    /** the derivative of the sigmoid function **/
    public static final double dsig(double x) {
        double exp = Math.exp(-x);
        double div = 1+exp;
        return exp/(div*div);
    }

    public static final void dsig(D1Matrix64F in, D1Matrix64F out) {
        assert(in.getNumRows()==out.getNumRows());
        assert(in.getNumCols()==out.getNumCols());
        int size = in.getNumElements();
        for(int i = 0; i < size; i++){
            out.set(i, dsig(in.get(i)));
        }
    }
}
