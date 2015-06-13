package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;

public class NewtonOptimizerTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-10;
    public void testOptimizer(){
        Parabloid f = new Parabloid(3,4,5);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{7},{11}});
        NewtonOptimizer optimizer = new NewtonOptimizer(f,x);
        optimizer.optimize(1);
        D1Matrix64F best = optimizer.getCurrentBest();
        assertEquals(0, best.get(0,0), MAX_ERROR);
        assertEquals(0, best.get(1,0), MAX_ERROR);
    }
}
