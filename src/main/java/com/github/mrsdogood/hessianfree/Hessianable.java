package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;

public interface Hessianable{
    public int dim();
    public double evaluate(D1Matrix64F x);
    public D1Matrix64F gradiant(D1Matrix64F x);
    public D1Matrix64F hessian(D1Matrix64F x);
}
