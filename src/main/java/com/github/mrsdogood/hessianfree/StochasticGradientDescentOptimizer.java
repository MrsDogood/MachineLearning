package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;

public class StochasticGradientDescentOptimizer
  extends GradientDescentOptimizer<StochasticGradientable> {

    public StochasticGradientDescentOptimizer(StochasticGradientable function,
      RowD1Matrix64F initialConditions, double learningRate){
        super(function, initialConditions, learningRate);
    }

    @Override
    public void step(RowD1Matrix64F x, RowD1Matrix64F out){
        super.step(x, out);
        getFunction().step();
    }
}
