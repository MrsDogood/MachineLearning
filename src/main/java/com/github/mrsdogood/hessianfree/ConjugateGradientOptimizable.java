package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;

public interface ConjugateGradientOptimizable{
    public int dim();
    public void getATimes(RowD1Matrix64F v, RowD1Matrix64F out);
    public void getB(RowD1Matrix64F out);
}
