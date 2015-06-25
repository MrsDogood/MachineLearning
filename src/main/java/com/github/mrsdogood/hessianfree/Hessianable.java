package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;

public interface Hessianable extends Gradientable{
    public void hessian(RowD1Matrix64F x, RowD1Matrix64F out);
}
