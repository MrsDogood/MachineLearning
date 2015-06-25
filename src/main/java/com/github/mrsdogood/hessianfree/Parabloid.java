package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

/**
* Just an example of a Hessianable function.
*/
public class Parabloid implements Hessianable{
    private double a,b,c;
    public Parabloid(double a, double b, double c){
        this.a = a;
        this.b = b;
        this.c = c;
    }
    public int dim(){
        return 2;
    }
    public double evaluate(RowD1Matrix64F point){
        double x = point.get(0,0);
        double y = point.get(1,0);
        return (x*x/(a*a)+y*y/(b*b))*c;
    }
    public void gradient(RowD1Matrix64F point, RowD1Matrix64F grad){
        double x = point.get(0,0);
        double y = point.get(1,0);
        // df/dx
        grad.set(0,0, 2*c*x/(a*a));
        // df/dy
        grad.set(1,0, 2*c*y/(b*b));
    }
    public void hessian(RowD1Matrix64F point, RowD1Matrix64F hess){
        double x = point.get(0,0);
        double y = point.get(1,0);
        // d2f/(dx*dx)
        hess.set(0,0, 2*c/(a*a));
        // d2f/(dy*dx)
        hess.set(1,0, 0);
        // d2f/(dx*dy)
        hess.set(0,1, 0);
        // d2f/(dy*dy)
        hess.set(1,1, 2*c/(b*b));
    }
}
