package com.github.mrsdogood.hessianfree;

import org.ejml.data.D1Matrix64F;
import org.ejml.data.DenseMatrix64F;

public abstract class Optimizer<F extends Function>{
    private F function;
    private D1Matrix64F cur, next;
    private int steps;

    public Optimizer(F function, D1Matrix64F initialConditions){
        this.function = function;
        this.cur = new DenseMatrix64F(initialConditions);
        this.next = new DenseMatrix64F(cur.getNumRows(), cur.getNumCols());
        this.steps = 0;
    }

    public F getFunction(){
        return function;
    }
    public D1Matrix64F getCurrentBest(){
        return cur;
    }
    public int getStepsTaken(){
        return steps;
    }
    public void optimize(int numSteps){
        D1Matrix64F tmp;
        for(int i = 0; i < numSteps; i++){
            step(cur, next);
            //swap cur and next
            tmp = cur;
            cur = next;
            next = tmp;
            steps++;
        }
    }

    protected abstract void step(D1Matrix64F x, D1Matrix64F out);
}
