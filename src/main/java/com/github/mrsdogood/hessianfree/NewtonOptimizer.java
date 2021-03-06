package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class NewtonOptimizer extends Optimizer<Hessianable> {
    private DenseMatrix64F grad, hess, prod;
    public NewtonOptimizer(Hessianable function, RowD1Matrix64F initialConditions){
        super(function, initialConditions);
        this.grad = Utils.getEmptyGradient(function);
        this.hess = Utils.getEmptyHessian(function);
        this.prod = new DenseMatrix64F(function.dim(), 1);
    }

    protected void step(RowD1Matrix64F x, RowD1Matrix64F out){
        getFunction().gradient(x,grad);   // grad = f'(x)
        getFunction().hessian(x,hess);    // hess = f''(x)
        CommonOps.invert(hess);           // hess' = hess^-1
        CommonOps.mult(hess, grad, prod); // prod = hess^-1 * grad
        CommonOps.subtract(x, prod, out); // out = x - (hess^-1 * grad)
    }
}
