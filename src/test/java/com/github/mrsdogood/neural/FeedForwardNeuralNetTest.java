package com.github.mrsdogood.neural;

import junit.framework.TestCase;
import java.util.Random;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;

public class FeedForwardNeuralNetTest extends TestCase {
    public static final double MAX_ERROR = 1.0e-10;
    public static final double MAX_ERROR_EST = 1.0e-3;
    public static final double EPSILON = 1.0e-12;
    public static final long RAND_SEED = 10934859303L;

    public void testEval(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 3, 2, 1);
        assertEquals(8, nn.getNumWeights());
        nn.evaluate(1, -1, 0);
    }

    public void testEvalManual11(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1);
        assertEquals(1, nn.getNumWeights());
        double x = r.nextGaussian();
        double w = nn.getWeight(0);
        double exp_f_x = Utils.sig(w*x); //f(x) = sig(w*x)
        double act_f_x = nn.evaluate(x)[0];
        assertEquals(exp_f_x, act_f_x, MAX_ERROR);
    }

    public void testEvalManual111(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1, 1);
        assertEquals(2, nn.getNumWeights());
        double x = r.nextGaussian();
        double w1 = nn.getWeight(0);
        double w2 = nn.getWeight(1);
        double exp_f_x = Utils.sig(w2*Utils.sig(w1*x)); // f(w1,w2) = sig(w2*sig(w1*x))
        double act_f_x = nn.evaluate(x)[0];
        assertEquals(exp_f_x, act_f_x, MAX_ERROR);
    }

    private static double[] getRandVect(Random r, int len){
        double[] vect = new double[len];
        for(int i = 0; i < vect.length; i++)
            vect[i] = r.nextGaussian();
        return vect;
    }

    private void unitize(double[] vect){
        double mag = 0;
        for(int i = 0; i < vect.length; i++){
            mag+=vect[i]*vect[i];
        }
        mag = Math.sqrt(mag);
        for(int i = 0; i < vect.length; i++){
            vect[i]/=mag;
        }
    }

    public void testDsig(){
        Random r = new Random(RAND_SEED);
        double x = r.nextGaussian();
        double y = Utils.sig(x);
        double dy = Utils.dsig(x);
        double est_dy = (Utils.sig(x+EPSILON)-Utils.sig(x))/EPSILON;
        assertEquals(dy, est_dy, MAX_ERROR_EST);
    }

    public void runBackpropTest(Random r, int... sizes){
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, sizes);
        double[] in = getRandVect(r, sizes[0]);
        double initOut = nn.evaluate(in)[0];
        // test weight mod direction
        double[] testDir = getRandVect(r, nn.getNumWeights());
        unitize(testDir);
        // get gradiant vector for output neuron
        nn.initBackprop();
        double[] grad = new double[nn.getNumWeights()];
        for(int w = 0; w < grad.length; w++){
            grad[w] = nn.backprop(w, 0);
        }
        // predict change to output neuron
        double expChange = 0; // grad * testDir
        for(int i = 0; i < grad.length; i++)
            expChange+= grad[i]*testDir[i];
        // apply change to weights
        for(int i = 0; i < grad.length; i++){
            double w = nn.getWeight(i);
            w += testDir[i]*EPSILON;
            nn.setWeight(i,w);
        }
        // check that the output changed as expected
        double endOut = nn.evaluate(in)[0];
        double actualChange = (endOut-initOut)/EPSILON;
        assertEquals(expChange, actualChange, MAX_ERROR_EST);
    }

    public void testBackPropManual11(){
        Random r = new Random(RAND_SEED+1);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1);
        double x = r.nextGaussian();
        double w = nn.getWeight(0);
        nn.evaluate(x); // f(w) = nn(x) = sig(w*x)
        nn.initBackprop();
        double exp_fp_w = Utils.dsig(w*x)*x; // f'(w) = sig'(w*x) * x
        double fp_w = nn.backprop(0,0);
        assertEquals(exp_fp_w, fp_w, MAX_ERROR);
    }

    public void testBackPropManual21(){
        Random r = new Random(RAND_SEED+2);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 2, 1);
        double x1 = r.nextGaussian();
        double x2 = r.nextGaussian();
        double w1 = nn.getWeight(0);
        double w2 = nn.getWeight(1);
        nn.evaluate(x1,x2); // f(w1,w2) = nn(x1,x2) = sig(w1*x1+w2*x2)
        nn.initBackprop();
        double exp_fp_w1 = Utils.dsig(w1*x1+w2*x2)*x1; // f'(w1) = sig'(w1*x1+w2*x2) * x1
        double fp_w1 = nn.backprop(0,0);
        assertEquals("f'(w1):", exp_fp_w1, fp_w1, MAX_ERROR);
        double exp_fp_w2 = Utils.dsig(w1*x1+w2*x2)*x2; // f'(w2) = sig'(w1*x1+w2*x2) * x2
        double fp_w2 = nn.backprop(1,0);
        assertEquals("f'(w2):", exp_fp_w2, fp_w2, MAX_ERROR);
    }

    public void testBackPropManual111(){
        Random r = new Random(RAND_SEED+3);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 1, 1, 1);
        double x = r.nextGaussian();
        double w1 = nn.getWeight(0);
        double w2 = nn.getWeight(1);
        nn.evaluate(x); // f(w1,w2) = nn(x1,x2) = sig(w2*sig(w1*x))
        nn.initBackprop();
        double exp_fp_w1 = Utils.dsig(w2*Utils.sig(w1*x)) * w2*Utils.dsig(w1*x) * x; // f'(w1) = sig'(w2*sig(w1*x)) * w2*sig'(w1*x) * w1
        double fp_w1 = nn.backprop(0,0);
        assertEquals("f'(w1):", exp_fp_w1, fp_w1, MAX_ERROR);
        double exp_fp_w2 = Utils.dsig(w2*Utils.sig(w1*x)) * Utils.sig(w1*x); // f'(w2) = sig'(w2*sig(w1*x)) * sig(w1*x)
        double fp_w2 = nn.backprop(0,0);
        assertEquals("f'(w2):", exp_fp_w1, fp_w1, MAX_ERROR);
    }

    public void testBackProp11(){
        for(int i = 0; i < 10; i++){
            Random r = new Random(RAND_SEED+i);
            runBackpropTest(r, 1, 1);
        }
    }

    public void testBackProp21(){
        for(int i = 0; i < 10; i++){
            Random r = new Random(RAND_SEED+i);
            runBackpropTest(r, 2, 1);
        }
    }

    public void testBackProp111(){
        for(int i = 0; i < 10; i++){
            Random r = new Random(RAND_SEED+i);
            runBackpropTest(r, 1, 1, 1);
        }
    }

    public void testBackProp9991(){
        for(int i = 0; i < 3; i++){
            Random r = new Random(RAND_SEED+i);
            runBackpropTest(r, 9, 9, 9, 1);
        }
    }
}
