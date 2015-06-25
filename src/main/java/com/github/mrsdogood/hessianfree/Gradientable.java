package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;

public interface Gradientable extends Function{
    public void gradient(RowD1Matrix64F x, RowD1Matrix64F out);
}
