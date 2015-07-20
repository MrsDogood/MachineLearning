package com.github.mrsdogood.hessianfree;

import java.util.Random;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static com.github.mrsdogood.hessianfree.Utils.copy;

public class HessianFreeOptimizer extends Optimizer<Gradientable>{
    private int cgoIterations;
    private Double defaultAlpha, maxAlpha;
    private boolean gradientDescentMode;
    private double curBestValue;
    public HessianFreeOptimizer(Gradientable f, RowD1Matrix64F initialConditions, 
            int cgoIterations, Double defaultAlpha, Double maxAlpha){
        super(f, initialConditions);
        this.cgoIterations = cgoIterations;
        this.defaultAlpha = defaultAlpha;
        this.maxAlpha = maxAlpha;
        gradientDescentMode = false;
        curBestValue = f.evaluate(initialConditions);
    }

    public HessianFreeOptimizer(Gradientable f, RowD1Matrix64F initialConditions,
            int cgoIterations){
        this(f, initialConditions, cgoIterations, null, null);
    }

    public boolean debug = false;

    public void step(RowD1Matrix64F x, RowD1Matrix64F out){
        Gradientable f = getFunction();
        RowD1Matrix64F best;
        if(gradientDescentMode){
            GradientDescentOptimizer gradOptimizer = new GradientDescentOptimizer(
                f, x, defaultAlpha);
            gradOptimizer.optimize(cgoIterations);
            best = gradOptimizer.getCurrentBest();
        } else {
            ConjugateGradientOptimizer cgOptimizer = new ConjugateGradientOptimizer(
                new GradientableToCGOAdapter(f, x), x);
            cgOptimizer.setDefaultAlpha(defaultAlpha);
            cgOptimizer.setMaxAlpha(maxAlpha);
            cgOptimizer.optimize(cgoIterations);
            if(defaultAlpha != null && cgOptimizer.gradHasBeenZero()){
                gradientDescentMode = true;
            }
            best = cgOptimizer.getCurrentBest();
        }
        if(f.evaluate(best) <= curBestValue){
            // Motherfuckin mutable structures, passin around pointers and shit.
            // This is where the bug is.
            copy(best, out);
        }
        else 
            copy(x, out);
        if(debug)
            System.out.println(curBestValue);
    }
}
