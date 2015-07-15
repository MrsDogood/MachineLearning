package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.multTransA;
import static org.ejml.ops.CommonOps.changeSign;
import static org.ejml.ops.CommonOps.insert;

public class ConjugateGradientOptimizer extends Optimizer<ConjugateGradientOptimizable>{
    private ConjugateGradientOptimizable optimizable;
    private int i;
    private DenseMatrix64F b, d;
    private DenseMatrix64F tmp1x1, tmpNx1, grad;
    private Double defaultAlpha, maxAlpha;
    private boolean gradHasBeenZero = false;
    public ConjugateGradientOptimizer(ConjugateGradientOptimizable optimizable,
        RowD1Matrix64F initialConditions){
        super(optimizable, initialConditions);
        i = 0;
        b = new DenseMatrix64F(optimizable.dim(), 1);
        optimizable.getB(b);
        d = new DenseMatrix64F(optimizable.dim(), 1);
        grad = new DenseMatrix64F(optimizable.dim(), 1);
        tmp1x1 = new DenseMatrix64F(1, 1);
        tmpNx1 = new DenseMatrix64F(optimizable.dim(), 1);
    }

    public boolean gradHasBeenZero() {
        return gradHasBeenZero;
    }

    // set a default alpha for when the actual alpha is negative
    // (i.e. when the quadratic is concave)
    public void setDefaultAlpha(Double defaultAlpha){
        this.defaultAlpha = defaultAlpha;
    }

    // set a max alpha for when the actual alpha is too large 
    public void setMaxAlpha(Double maxAlpha){
        this.maxAlpha = maxAlpha;
    }

    @Override
    public void optimize(int iterations){
        for(int i = 0; i < iterations; i++){
            if(gradHasBeenZero)
                break;
            super.optimize(1);
        }
    }

    public void step(RowD1Matrix64F x, RowD1Matrix64F out){
        try{
            if(!gradHasBeenZero){
                if(i==0)
                    firstStep(x,out);
                else
                    ithStep(x,out);
            }
            i++;
        } catch (com.github.mrsdogood.neural.Utils.SigNaNException e){
            System.err.println("iteration: "+i);
            System.err.println("grad has been zero: "+gradHasBeenZero);
            throw e;
        }
    }

    private void firstStep(RowD1Matrix64F x, RowD1Matrix64F out){
        calcGrad(x);
        if(gradIsZero()){
            insert(x,out,0,0); // out = x
            return;
        }
        changeSign(grad, d);        // d = -grad
        calcAlphaAndUpdateX(x, out);
    }

    private void calcGrad(RowD1Matrix64F x){
        try{
            getFunction().getATimes(x, grad); // grad = A * x
            add(grad,b,grad);                 // grad = A * x + b
        } catch (com.github.mrsdogood.neural.Utils.SigNaNException e){
            System.err.println("x:\n"+x);
            throw e;
        }
    }

    private boolean gradIsZero(){
        final double EPSILON = 1e-3;
        int dim = grad.getNumRows();
        for(int i = 0; i < dim; i++){
            if(Math.abs(grad.get(i))>EPSILON){
                return false;
            }
        }
        gradHasBeenZero = true;
        return true;
    }

    // requires an updated d and grad
    private void calcAlphaAndUpdateX(RowD1Matrix64F x, RowD1Matrix64F out){
        multTransA(d,grad,tmp1x1);          // tmp1x1 = d^T * grad
        double alphaNum = -tmp1x1.get(0);
        getFunction().getATimes(d, tmpNx1); // tmpNx1 = A * d
        multTransA(d,tmpNx1,tmp1x1);        // tmp1x1 = d^T * A * d
        // alpha = (d^T * d) / (d^T * A * d)
        double alpha = alphaNum/tmp1x1.get(0);
        if((alpha<0 || tmp1x1.get(0)==0.0) && defaultAlpha!=null){
            alpha = defaultAlpha;
        } else if(maxAlpha!=null && alpha>maxAlpha){
            alpha = maxAlpha;
        }
        add(x,alpha,d,out); // out = x + alpha * d
    }

    private void ithStep(RowD1Matrix64F x, RowD1Matrix64F out){
        getFunction().getATimes(d, tmpNx1);   // tmpNx1 = A * d
        calcGrad(x);
        if(gradIsZero()){
            insert(x,out,0,0); // out = x
            return;
        }
        multTransA(grad, tmpNx1, tmp1x1); // tmp1x1 = grad^T * A * d
        double betaNum = tmp1x1.get(0);
        multTransA(d, tmpNx1, tmp1x1);        // tmp1x1 = d^T * A * d
        // beta = [grad^T * A * d] / (d^T * A * d)
        if(tmp1x1.get(0)==0){
            insert(x,out,0,0); // out = x
            return;
        }
        double beta = betaNum/tmp1x1.get(0);
        try{
            add(-1.0, grad, beta, d, d);      // d' = -grad + beta * d
            calcAlphaAndUpdateX(x, out);
        } catch (com.github.mrsdogood.neural.Utils.SigNaNException e){
            System.err.println("grad:\n"+grad);
            System.err.println("beta: "+beta);
            throw e;
        }
    }
}
