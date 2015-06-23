package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;
public interface Function{
    public int dim();
    public double evaluate(D1Matrix64F x);
}
