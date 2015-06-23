package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;

import org.ejml.data.DenseMatrix64F;

public class ParabloidTest extends TestCase{
    public void testParabloid(){
        Parabloid f = new Parabloid(3,4,5);
        assertEquals(f.dim(), 2);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{7},{11}});
        //calc gradiant
        DenseMatrix64F grad = Utils.getEmptyGradiant(f);
        f.gradiant(x, grad);
        //calc hessian
        DenseMatrix64F hess = Utils.getEmptyHessian(f);
        f.hessian(x, hess);
    }

    public void testGradiant(){
        long randseed = 23984787029L;
        Parabloid f = new Parabloid(3,4,5);
        for(int i = 0; i < 100; i++){
            GradiantCheckUtil.check(this, f, randseed+i);
        }
    }
        
}
