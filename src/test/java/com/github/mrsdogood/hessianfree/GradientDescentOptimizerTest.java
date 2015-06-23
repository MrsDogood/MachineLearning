package com.github.mrsdogood.hessianfree;

import com.github.mrsdogood.neural.FeedForwardNeuralNet;
import com.github.mrsdogood.neural.FeedForwardNeuralNetErrorFunction;

import junit.framework.TestCase;
import java.util.Random;
import java.util.Arrays;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.RandomMatrices;

public class GradientDescentOptimizerTest extends TestCase {
    public static final long RAND_SEED = 93716349023L;

    public void testParabloid(){
        Parabloid f = new Parabloid(3,4,5);
        DenseMatrix64F x = new DenseMatrix64F(new double[][]{{7},{11}});
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer(f,x,.5);
        TestUtils.checkOptimizer(this, optimizer, 25, new double[]{0,0});
    }

    private double[] getInput(int i){
        return new double[]{
            (i>>0)%2==0?0:1,
            (i>>1)%2==0?0:1,
            (i>>2)%2==0?0:1
        };
    }

    private double[] getExpectedOutput(double[] inputs){
        return new double[]{
            inputs[0]==1 || inputs[2]==0 ? 1:0,
            inputs[1]==1 && inputs[2]==1 ? 1:0
        };
    }

    public void testNeuralNet(){
        Random r = new Random(RAND_SEED);
        FeedForwardNeuralNet nn = new FeedForwardNeuralNet(r, 3, 3, 2);
        DenseMatrix64F x = nn.getWeights();
        //DenseMatrix64F x = RandomMatrices.createGaussian(nn.getNumWeights(), 1, 0, 5, r);
        FeedForwardNeuralNetErrorFunction f =
            new FeedForwardNeuralNetErrorFunction(nn);
        for(int i = 0; i < 8; i++){
            double[] input = getInput(i);
            double[] output = getExpectedOutput(input);
            //System.out.println(Arrays.toString(input)+" -> "+Arrays.toString(output));
            f.addTrainingSet(input, output);
        }
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer(f,x,5);
        optimizer.optimize(500);
        nn.setWeights(optimizer.getCurrentBest());
        for(int i = 0; i < 8; i++){
            double[] input = getInput(i);
            double[] expOutput = getExpectedOutput(input);
            double[] actOutput = nn.evaluate(input);
            for(int o = 0; o < expOutput.length; o++){
                assertEquals("nn output "+o+" on set "+i+":", 
                    expOutput[o], actOutput[o], 0.1);
            }
        }
    }
}
