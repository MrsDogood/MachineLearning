package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.mult;
import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.multTransA;

public class Quadratic implements ConjugateGradientOptimizable, Function{
    private int dim;
    private RowD1Matrix64F a;
    private RowD1Matrix64F b;
    private DenseMatrix64F ax, tmp1x1;
    private double c;
    public Quadratic(RowD1Matrix64F a, RowD1Matrix64F b, double c){
        this.a = a;
        this.b = b;
        this.c = c;
        dim = a.getNumRows();
        assert(a.getNumCols()==dim);
        assert(b.getNumRows()==dim);
        ax = new DenseMatrix64F(dim,1);
        tmp1x1 = new DenseMatrix64F(1,1);
    }
    public int dim(){
        return dim;
    }
    public double evaluate(RowD1Matrix64F x){
        double result = c;
        getATimes(x, ax); // ax = a * x
        multTransA(x, ax, tmp1x1); // tmp1x1 = x^T * a * x
        result += 0.5*tmp1x1.get(0);
        multTransA(b, x, tmp1x1); // tmp1x1 = b^T * x
        result += tmp1x1.get(0);
        return result; // (0.5 * x^T * a * x) + (b^T * x) + c
    }
    public void getATimes(RowD1Matrix64F v, RowD1Matrix64F out){
        mult(a,v,out);
    }
    public void getB(RowD1Matrix64F out){
        out.setData(b.getData());
    }
}
