package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;

public interface Hessianable extends Gradiantable{
    public void hessian(D1Matrix64F x, D1Matrix64F out);
}
