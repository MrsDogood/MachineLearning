package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;

public interface Hessianable{
    public int dim();
    public double evaluate(D1Matrix64F x);
    public void gradiant(D1Matrix64F x, D1Matrix64F out);
    public void hessian(D1Matrix64F x, D1Matrix64F out);
}
