package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.multTransA;
import static org.ejml.ops.CommonOps.changeSign;

public class ConjugateGradientOptimizer extends Optimizer<ConjugateGradientOptimizable>{
    private ConjugateGradientOptimizable optimizable;
    private int i;
    private DenseMatrix64F b, d;
    private DenseMatrix64F tmp1x1, tmpNx1;
    public ConjugateGradientOptimizer(ConjugateGradientOptimizable optimizable,
        RowD1Matrix64F initialConditions){
        super(optimizable, initialConditions);
        i = 0;
        b = new DenseMatrix64F(optimizable.dim(), 1);
        optimizable.getB(b);
        d = new DenseMatrix64F(optimizable.dim(), 1);
        tmp1x1 = new DenseMatrix64F(1, 1);
        tmpNx1 = new DenseMatrix64F(optimizable.dim(), 1);
    }

    public void step(RowD1Matrix64F x, RowD1Matrix64F out){
        if(i==0)
            firstStep(x,out);
        else
            ithStep(x,out);
        i++;
    }

    private void firstStep(RowD1Matrix64F x, RowD1Matrix64F out){
        getFunction().getATimes(x, d);      // d = A * x
        add(d,b,d);                         // d = d + b
        changeSign(d);
        multTransA(d,d,tmp1x1);             // tmp1x1 = d^T * d
        double alphaNum = tmp1x1.get(0);
        getFunction().getATimes(d, tmpNx1); // tmpNx1 = A * d
        multTransA(d,tmpNx1,tmp1x1);        // tmp1x1 = d^T * A * d
        // alpha = (d^T * d) / (d^T * A * d)
        double alpha = alphaNum/tmp1x1.get(0);
        add(x,alpha,d,out);                // out = x + alpha * d
    }

    private void ithStep(RowD1Matrix64F x, RowD1Matrix64F out){
        throw new RuntimeException("Not Implemented.");
    }
}
