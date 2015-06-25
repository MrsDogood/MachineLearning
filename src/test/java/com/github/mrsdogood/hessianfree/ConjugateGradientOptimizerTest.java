package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

public class ConjugateGradientOptimizerTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-3;
    public void testOptimizer(){
        DenseMatrix64F a = new DenseMatrix64F(new double[][]{{5, 4},{4,5}});
        DenseMatrix64F b = new DenseMatrix64F(new double[][]{{-1},{2}});
        DenseMatrix64F x0 = new DenseMatrix64F(new double[][]{{4},{3}});
        Quadratic f = new Quadratic(a,b,0);
        assertEquals("f(x0):", 112.5, f.evaluate(x0), MAX_ERROR);
        ConjugateGradientOptimizer optimizer = new ConjugateGradientOptimizer(f, x0);
        TestUtils.checkOptimizer(this, optimizer, 1, MAX_ERROR, 
            new double[]{0.55257, -0.66985});
        assertEquals("f(x1):", -1.48774, 
            f.evaluate(optimizer.getCurrentBest()), MAX_ERROR);
    }
}
