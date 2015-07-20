package com.github.mrsdogood.hessianfree;

import org.ejml.data.RowD1Matrix64F;
import org.ejml.data.DenseMatrix64F;

import static org.ejml.ops.CommonOps.add;
import static org.ejml.ops.CommonOps.insert;
import static com.github.mrsdogood.hessianfree.Utils.copy;

public class GradientableToCGOAdapter implements ConjugateGradientOptimizable{
    public static final double EPSILON = 1.0e-5;
    private Gradientable plugin;
    private RowD1Matrix64F center, gradAtCenter;
    private RowD1Matrix64F tmpNx1; 
    private double evalAtCenter;
    public GradientableToCGOAdapter(Gradientable plugin, RowD1Matrix64F center){
        this.plugin = plugin;
        this.center = new DenseMatrix64F(plugin.dim(), 1);
        gradAtCenter = new DenseMatrix64F(plugin.dim(), 1);
        tmpNx1 = new DenseMatrix64F(plugin.dim(), 1);
        updateCenter(center);
    }

    public void updateCenter(RowD1Matrix64F c){
        insert(c, center, 0, 0);
        evalAtCenter = plugin.evaluate(center);
        plugin.gradient(center, gradAtCenter);
    }

    public int dim(){
        return plugin.dim();
    }

    // 0.5A = H, Hv = 0.5Av , 2Hv = Av
    public void getATimes(RowD1Matrix64F v, RowD1Matrix64F out){
        try{
            // tmpNx1 = center + EPSILON*v
            add(center, EPSILON, v, tmpNx1);
            plugin.evaluate(tmpNx1);
        } catch (com.github.mrsdogood.neural.Utils.SigNaNException e){
            System.err.println("Directional vector:\n"+v);
            throw e;
        }
        plugin.gradient(tmpNx1, out);
        double multiplier = 2/EPSILON;
        add(multiplier, out, -multiplier, gradAtCenter, out);
    }

    public void getB(RowD1Matrix64F out){
        copy(gradAtCenter, out);
    }
}
