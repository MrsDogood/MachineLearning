package com.github.mrsdogood.hessianfree;

import junit.framework.TestCase;
import java.util.Random;
import java.util.Arrays;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.mult;
import static org.ejml.ops.CommonOps.scale;

public class GradientableToCGOAdapterTest extends TestCase {
    public static final double MAX_ERROR = 1e-5;
    public static final long RAND_SEED = 7239094844L;
    public void testParabloid(){
        Random r = new Random(RAND_SEED);
        Parabloid f = new Parabloid(3, 4, 5);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{4},{3}});
        DenseMatrix64F v = RandomMatrices.createGaussian(2, 1, 0, 1, r);
        DenseMatrix64F A = new DenseMatrix64F(2,2);
        f.hessian(x, A);
        scale(2, A);
        GradientableToCGOAdapter adapter = new GradientableToCGOAdapter(f, x);
        DenseMatrix64F estAv = new DenseMatrix64F(2,1);
        DenseMatrix64F actAv = new DenseMatrix64F(2,1);
        adapter.getATimes(v, estAv);
        mult(A, v, actAv);
        assertEquals(estAv.get(0), actAv.get(0), MAX_ERROR);
        assertEquals(estAv.get(1), actAv.get(1), MAX_ERROR);
    }
}
