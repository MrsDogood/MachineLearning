package com.github.mrsdogood.hessianfree;

import org.ejml.data.DenseMatrix64F;

public class Utils{
    public static DenseMatrix64F getEmptyGradient(Hessianable f){
        return new DenseMatrix64F(f.dim(),1);
    }
    public static DenseMatrix64F getEmptyHessian(Hessianable f){
        return new DenseMatrix64F(f.dim(),f.dim());
    }
}
