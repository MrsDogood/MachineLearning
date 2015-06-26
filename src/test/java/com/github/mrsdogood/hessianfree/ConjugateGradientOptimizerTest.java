package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.multTransA;

public class ConjugateGradientOptimizerTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-3;
    public static final long RAND_SEED = -2309487539010L;

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

    public void testOptimizerMore(){
        DenseMatrix64F a = new DenseMatrix64F(new double[][]{{5, 4},{4,5}});
        DenseMatrix64F b = new DenseMatrix64F(new double[][]{{-1},{2}});
        DenseMatrix64F x0 = new DenseMatrix64F(new double[][]{{4},{3}});
        Quadratic f = new Quadratic(a,b,0);
        ConjugateGradientOptimizer optimizer = new ConjugateGradientOptimizer(f, x0);
        optimizer.optimize(1);
        double curBest = f.evaluate(optimizer.getCurrentBest());
        for(int i = 2; i < 10; i++){
            optimizer.optimize(1);
            double fxi = f.evaluate(optimizer.getCurrentBest());
            if(fxi > curBest+MAX_ERROR)
                fail(i+"th iteration failed; larger than previous opt.");
            curBest = fxi;
        }
    }

    public void testOptimizer9x9(){
        Random r = new Random(RAND_SEED);
        DenseMatrix64F aPart = RandomMatrices.createGaussian(9, 9, 0, 100, r);
        DenseMatrix64F a = new DenseMatrix64F(9,9);
        multTransA(aPart,aPart,a);
        DenseMatrix64F b = RandomMatrices.createGaussian(9, 1, 0, 100, r);
        DenseMatrix64F x0 = RandomMatrices.createGaussian(9, 1, 0, 100, r);
        Quadratic f = new Quadratic(a,b,0);
        ConjugateGradientOptimizer optimizer = new ConjugateGradientOptimizer(f, x0);
        optimizer.optimize(1);
        double curBest = f.evaluate(optimizer.getCurrentBest());
        for(int i = 2; i < 20; i++){
            optimizer.optimize(1);
            double fxi = f.evaluate(optimizer.getCurrentBest());
            if(fxi > curBest+MAX_ERROR)
                fail(i+"th iteration failed; larger than previous opt.");
            curBest = fxi;
        }
    }
}
