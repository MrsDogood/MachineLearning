package com.github.mrsdogood.hessianfree;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import org.ejml.data.DenseMatrix64F;

public class ParabloidTest extends TestCase {
    public ParabloidTest(String testName){
        super(testName);
    }
    public static Test suite(){
        return new TestSuite(ParabloidTest.class);
    }

    public void testParabloid()
    {
        Parabloid f = new Parabloid(3,4,5);
        assertEquals(f.dim(), 2);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{7},{11}});
        System.out.println(f.gradiant(x));
        System.out.println(f.hessian(x));
    }
}
