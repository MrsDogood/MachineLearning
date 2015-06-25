package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.mult;
import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.multTransA;

public class Quadratic implements ConjugateGradientOptimizable, Function{
    private int dim;
    private RowD1Matrix64F a;
    private D1Matrix64F b;
    private double c;
    public Quadratic(RowD1Matrix64F a, D1Matrix64F b, double c){
        this.a = a;
        this.b = b;
        this.c = c;
        dim = a.getNumRows();
        assert(a.getNumCols()==dim);
        assert(b.getNumRows()==dim);
    }
    public int dim(){
        return dim;
    }
    public double evaluate(RowD1Matrix64F x){
        return 0; //TODO
    }
    public void getATimes(RowD1Matrix64F v, RowD1Matrix64F out){
        mult(a,v,out);
    }
    public void getB(D1Matrix64F out){
        out.setData(b.getData());
    }
}
