package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.D1Matrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.dot;
import static org.ejml.ops.CommonOps.scale;

public class GradiantCheckUtil {
    public static final double MAX_ERROR = 1.0e-3;
    public static final double EPSILON = 1.0e-10;

    public static void check(TestCase t, Gradiantable g, Random r, double epsilon, double max_error){
        int dim = g.dim();
        // test from random position x
        D1Matrix64F x = RandomMatrices.createGaussian(dim, 1, 0, 1, r);
        // calculate gradiant
        D1Matrix64F grad = new DenseMatrix64F(dim, 1);
        g.gradiant(x,grad);
        // create random unit vector direction
        D1Matrix64F randDir = RandomMatrices.createGaussian(dim, 1, 0, 1, r);
        scale(Math.sqrt(dot(randDir, randDir)), randDir);
        // get the expected rate of change in randDir
        double expChange = dot(grad, randDir);
        // get the actual approximate rate of change
        double y1 = g.evaluate(x);
        add(x, epsilon, randDir, x); // x = x+epsilon*randDir
        double y2 = g.evaluate(x);
        double actChange = (y2-y1)/epsilon;
        t.assertEquals(expChange, actChange, max_error);
    }
    
    public static void check(TestCase t, Gradiantable g, long randseed){
        Random r = new Random(randseed);
        check(t,g,r,EPSILON,MAX_ERROR);
    }
}
