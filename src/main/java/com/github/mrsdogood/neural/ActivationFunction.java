package com.github.mrsdogood.neural;

public interface ActivationFunction {
    public double f(double x);
    // the derivative of f with respect to x at x
    public double df(double x);
}
