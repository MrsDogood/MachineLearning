package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;

public interface Gradientable extends Function{
    public void gradient(D1Matrix64F x, D1Matrix64F out);
}
