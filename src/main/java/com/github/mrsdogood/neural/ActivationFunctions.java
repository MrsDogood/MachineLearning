package com.github.mrsdogood.neural;

public final class ActivationFunctions {
    // LINEAR: domain=[-inf,+inf] range=[-inf,+inf]
    public static final ActivationFunction LINEAR = new Linear();
    // SIGMOID: domain=[-inf,+inf] range=[0,1]
    public static final ActivationFunction SIGMOID = new Sigmoid();
    // TANH: domain=[-inf,+inf] range=[-1,1]
    public static final ActivationFunction TANH = new Tanh();
}

class Linear implements ActivationFunction {
    public double f(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("linear input not a number");
        return x;
    }
    public double df(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("dlinear input not a number");
        return 1.0;
    }
}

class Sigmoid implements ActivationFunction {
    public double f(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("sig input not a number");
        double denom = 1.0+Math.exp(-x);
        if(denom==Double.POSITIVE_INFINITY)
            return 0;
        double ret = 1.0/denom;
        assert(0.0<=ret);
        assert(ret<=1.0);
        return ret;
    }
    public double df(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("dsig input not a number");
        double sig = f(x);
        return sig*(1-sig);
    }
}

class Tanh implements ActivationFunction {
    public double f(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("tanh input not a number");
        if(x==Double.POSITIVE_INFINITY)
            return 1;
        else if(x==Double.NEGATIVE_INFINITY)
            return -1;
        double exp = Math.exp(x);
        double invExp = Math.exp(-x); // more accuracy than 1/exp
        double ret = (exp-invExp)/(exp+invExp);
        assert(-1 <= ret);
        assert(ret <= 1);
        return ret;
    }
    public double df(double x){
        if(Double.isNaN(x))
            throw new ActNaNException("dtanh input not a number");
        double tanh = f(x);
        return 1-(tanh*tanh);
    }
}
