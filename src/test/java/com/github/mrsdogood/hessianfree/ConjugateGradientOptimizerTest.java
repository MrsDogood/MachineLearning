package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;


public class NewtonOptimizerTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-10;
    public void testOptimizer(){
        DenseMatrix64F a = new DenseMatrix64F(new double[][]{{5, 4},{4,5}});
        DenseMatrix64F b = new DenseMatrix64F(new double[][]{{-1},{2}});
        Parabloid f = new Parabloid(3,4,5);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{7},{11}});
        NewtonOptimizer optimizer = new NewtonOptimizer(f,x);
        TestUtils.checkOptimizer(this, optimizer, 1, 0.001, double[]{0.53172, -1.13956});
    }
}
