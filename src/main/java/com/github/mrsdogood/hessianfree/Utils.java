package com.github.mrsdogood.hessianfree;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;

public class Utils{
    public static DenseMatrix64F getEmptyGradient(Hessianable f){
        return new DenseMatrix64F(f.dim(),1);
    }
    public static DenseMatrix64F getEmptyHessian(Hessianable f){
        return new DenseMatrix64F(f.dim(),f.dim());
    }

    public static void copy(double[] from, RowD1Matrix64F to){
        System.arraycopy(from, 0, to.getData(), 0, from.length);
    }

    public static void copy(RowD1Matrix64F from, RowD1Matrix64F to){
        System.arraycopy(from.getData(), 0, to.getData(), 0, from.getData().length);
    }
}
