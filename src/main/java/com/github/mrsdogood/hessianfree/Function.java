package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;

public interface Function{
    public int dim();
    public double evaluate(RowD1Matrix64F x);
}
