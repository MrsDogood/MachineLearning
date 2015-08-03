package com.github.mrsdogood.neural;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.data.RowD1Matrix64F;
import org.ejml.ops.RandomMatrices;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.dot;
import static org.ejml.ops.CommonOps.scale;

public class ActivationFunctionsTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-3;
    public static final double EPSILON = 1.0e-10;
    
    public void _testDerivative(ActivationFunction f, double x){
        double fx = f.f(x);
        double fxh = f.f(x+EPSILON);
        double approxDerivative = (fxh-fx)/EPSILON;
        double actualDerivative = f.df(x);
        assertEquals(
            "Derivative of "+f.getClass().getName()+" at "+x+":",
            approxDerivative, actualDerivative, MAX_ERROR);
    }

    public void _testDerivative(ActivationFunction f, 
        double minSqrtX, double maxSqrtX, double delta){
        for(double xSqrt = minSqrtX; xSqrt <= maxSqrtX; xSqrt+=delta)
            _testDerivative(f,xSqrt*Math.abs(xSqrt));
    }

    public void testLinearDerivative(){
        _testDerivative(
            ActivationFunctions.LINEAR, -10, 10, 0.1);
    }

    public void testSigmoidDerivative(){
        _testDerivative(
            ActivationFunctions.SIGMOID, -10, 10, 0.1);
    }

    public void testTanhDerivative(){
        _testDerivative(
            ActivationFunctions.TANH, -10, 10, 0.1);
    }
}
