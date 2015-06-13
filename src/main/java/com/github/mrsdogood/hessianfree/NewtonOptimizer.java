package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class NewtonOptimizer extends HessianableOptimizer {
    private DenseMatrix64F grad, hess, prod;
    public NewtonOptimizer(Hessianable function, D1Matrix64F initialConditions){
        super(function, initialConditions);
        this.grad = Utils.getEmptyGradiant(function);
        this.hess = Utils.getEmptyHessian(function);
        this.prod = new DenseMatrix64F(function.dim(), 1);
    }

    protected void step(D1Matrix64F x, D1Matrix64F out){
        getFunction().gradiant(x,grad);   // grad = f'(x)
        getFunction().hessian(x,hess);    // hess = f''(x)
        CommonOps.invert(hess);           // hess' = hess^-1
        CommonOps.mult(hess, grad, prod); // prod = hess^-1 * grad
        CommonOps.subtract(x, prod, out); // out = x - (hess^-1 * grad)
    }
}
