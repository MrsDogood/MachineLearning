import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.add;

public class GradientDescentOptimizer extends Optimizer<Gradientable> {
    private double learningRate;
    private DenseMatrix64F gradient;
    public GradientDescentOptimizer(Gradientable function, RowD1Matrix64F initialConditions, 
        double learningRate){
        super(function, initialConditions);
        this.learningRate = learningRate;
        this.gradient = new DenseMatrix64F(function.dim(), 1);
    }
    protected void step(RowD1Matrix64F x, RowD1Matrix64F out){
        Gradientable g = getFunction();
        g.gradient(x, gradient);
        add(x, -learningRate, gradient, out); // out = x-learningRate*gradient
    }
}
