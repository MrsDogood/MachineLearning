package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.dot;
import static org.ejml.ops.CommonOps.scale;

public class TestUtils {
    public static final double MAX_ERROR = 1.0e-3;
    public static final double EPSILON = 1.0e-10;

    public static void checkGradient(TestCase t, Gradientable g, Random r, double epsilon, double maxError){
        int dim = g.dim();
        // test from random position x
        RowD1Matrix64F x = RandomMatrices.createGaussian(dim, 1, 0, 1, r);
        // calculate gradient
        RowD1Matrix64F grad = new DenseMatrix64F(dim, 1);
        g.gradient(x,grad);
        // create random unit vector direction
        RowD1Matrix64F randDir = RandomMatrices.createGaussian(dim, 1, 0, 1, r);
        scale(Math.sqrt(dot(randDir, randDir)), randDir);
        // get the expected rate of change in randDir
        double expChange = dot(grad, randDir);
        // get the actual approximate rate of change
        double y1 = g.evaluate(x);
        add(x, epsilon, randDir, x); // x = x+epsilon*randDir
        double y2 = g.evaluate(x);
        double actChange = (y2-y1)/epsilon;
        t.assertEquals(expChange, actChange, maxError);
    }
    
    public static void checkGradient(TestCase t, Gradientable g, long randseed){
        Random r = new Random(randseed);
        checkGradient(t,g,r,EPSILON,MAX_ERROR);
    }

    public static void checkOptimizer(TestCase t, Optimizer optimizer, int iterations, double maxError, double[] expected){
        optimizer.optimize(iterations);
        RowD1Matrix64F best = optimizer.getCurrentBest();
        assert(best.getNumElements()==expected.length);
        for(int i = 0; i < expected.length; i++){
            t.assertEquals("Optimizer expected output ["+i+"]:", expected[i], best.get(i), maxError);
        }
    }

    public static void checkOptimizer(TestCase t, Optimizer optimizer, int iterations, double[] expected){
        checkOptimizer(t, optimizer, iterations, MAX_ERROR, expected);
    }

    public static double[] getRandVect(Random r, int len){
        double[] vect = new double[len];
        for(int i = 0; i < vect.length; i++)
            vect[i] = r.nextGaussian();
        return vect;
    }
}
