package com.github.mrsdogood.neural;

import com.github.mrsdogood.hessianfree.GradiantCheckUtil;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;

public class FeedForwardNeuralNetErrorFunctionTest extends TestCase{
    public static final long RAND_SEED = -2384761899L;
    public static final double EPSILON = 1e-10;
    public static final double MAX_ERROR = 1e-8;
    public static final double MAX_ERROR_EST = 1e-3;

    private static double[] getRandVect(Random r, int len){
        double[] vect = new double[len];
        for(int i = 0; i < vect.length; i++)
            vect[i] = r.nextGaussian();
        return vect;
    }

    public void testEvalManual11(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1);
        double in = r.nextGaussian();
        double expOut = r.nextGaussian();
        double actOut = nn.evaluate(in)[0];
        double expError = 0.5*(expOut-actOut)*(expOut-actOut);
        double w = nn.getWeight(0);
        FeedForwardNeuralNetErrorFunction f =
            new FeedForwardNeuralNetErrorFunction(nn);
        f.addTrainingSet(new double[]{in}, new double[]{expOut});
        D1Matrix64F x = new DenseMatrix64F(1,1);
        x.set(0,0,w);
        double actError = f.evaluate(x);
        assertEquals(expError, actError, MAX_ERROR);
    }

    public void testGradManual11(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1);
        double in = r.nextGaussian();
        double expOut = r.nextGaussian();
        double actOut = nn.evaluate(in)[0];
        nn.initBackprop();
        double expDer = (actOut-expOut)*nn.backprop(0,0);
        double w = nn.getWeight(0);
        FeedForwardNeuralNetErrorFunction f =
            new FeedForwardNeuralNetErrorFunction(nn);
        f.addTrainingSet(new double[]{in}, new double[]{expOut});
        D1Matrix64F x = new DenseMatrix64F(1,1);
        x.set(0,0,w);
        D1Matrix64F grad = new DenseMatrix64F(1,1);
        f.gradiant(x, grad);
        double actDer = grad.get(0);
        assertEquals(expDer, actDer, MAX_ERROR);
    }

    public void testGradManual11_2(){
        Random r = new Random(RAND_SEED+5);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1);
        double in = r.nextGaussian();
        double expOut = r.nextGaussian();
        double actOut = nn.evaluate(in)[0];
        nn.initBackprop();
        double expDer = (actOut-expOut)*nn.backprop(0,0);
        double w = nn.getWeight(0);

        FeedForwardNeuralNetErrorFunction f =
            new FeedForwardNeuralNetErrorFunction(nn);
        f.addTrainingSet(new double[]{in}, new double[]{expOut});

        D1Matrix64F x = new DenseMatrix64F(1,1);
        x.set(0,0,w);
        double y1 = f.evaluate(x);
        x.set(0,0,w+EPSILON);
        double y2 = f.evaluate(x);
        double actDer = (y2-y1)/EPSILON;

        assertEquals(expDer, actDer, MAX_ERROR_EST);
    }

    public void _testGradX(int... sizes){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, sizes);
        FeedForwardNeuralNetErrorFunction f = 
            new FeedForwardNeuralNetErrorFunction(nn);
        for(int i = 0; i < 10; i++){
            f.addTrainingSet(
                getRandVect(r, nn.getInputSize()),
                getRandVect(r, nn.getOutputSize())
            );
        }
        for(int i = 0; i < Math.max(1000/nn.getNumWeights(), 5); i++){
            GradiantCheckUtil.check(this, f, RAND_SEED+1+i);
        }
    }

    public void testGrad11(){
        _testGradX(1, 1);
    }

    public void testGrad12(){
        _testGradX(1, 2);
    }

    public void testGrad111(){
        _testGradX(1, 1, 1);
    }

    public void testGrad22(){
        _testGradX(2, 2);
    }

    public void testGrad333(){
        _testGradX(3, 3, 3);
    }

    public void testGrad535(){
        _testGradX(5, 3, 5);
    }
}
