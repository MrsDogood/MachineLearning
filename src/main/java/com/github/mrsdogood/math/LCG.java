package com.github.mrsdogood.math;

import java.util.Vector;
import java.util.Random;

/**
 * An implementation fo a linear congruential generator.
 * See: https://en.wikipedia.org/wiki/Linear_congruential_generator
 */
public class LCG {
    public static int[] primeFactors(int n){
        Vector<Integer> factors = new Vector<Integer>();
        int max = (int)Math.sqrt(n);
        assert((max+1)*(max+1)>n);
        for(int i = 2; i <= max; i++){
            if(n%i==0){
                factors.add(i);
                while(n%i==0)
                    n/=i;
                max = (int)Math.sqrt(n);
                assert((max+1)*(max+1)>n);
            }
        }
        if(n!=0)
            factors.add(n);
        int[] ret = new int[factors.size()];
        for(int i = 0; i < ret.length; i++)
            ret[i] = factors.get(i);
        return ret;
    }

    public static int gcd(int a, int b){
        while(b!=0){
            int tmp = b;
            b = a%b;
            a = tmp;
        }
        return a;
    }

    private int m, a, c, x;
    public LCG(int m, int a, int c, int x0){
        this.m = m;
        this.a = a;
        this.c = c;
        this.x = x0;
    }
    public int getM(){ return m; }
    public int getA(){ return a; }
    public int getC(){ return c; }

    private int getNextX(long x){
        //x = (a*x+c)%m;
        x*=a; 
        x%=m;
        x+=c;
        x%=m;
        return (int)x;
    }

    public int next(){
        return next(1);
    }
    public int next(int n){
        for(int i = 0; i < n; i++)
            x = getNextX(x);
        return x;
    }

    public int poll(){
        return poll(0);
    }
    public int poll(int ahead){
        int xp = x;
        for(int i = 0; i < ahead; i++)
            xp = getNextX(xp);
        return xp;
    }

    public static LCG makeLCG(int period, Random r){
        return makeLCG(period, r, true);
    }
    public static LCG makeLCG(int period, Random r, boolean forceMultiplier){
        // init a
        int[] primes = primeFactors(period);
        int pp = 1;
        for(int i = 0; i < primes.length; i++)
            pp*=primes[i];
        if(period%4==0){
            pp*=2;
            assert(pp%4==0);
        }
        assert(period%pp==0);
        if(pp==period && forceMultiplier){
            if(period%4==2){
                period*=4;
                pp*=2;
            }else if(period%4==0){
                period*=2;
            }else if(primes[0]<8){
                assert(primes[0]!=2);
                period*=primes[0];
            }else{
                period*=8;
                pp*=4;
            }
        }
        int multBound = period/pp;
        int mult = 0;
        if(multBound > 1)
            mult = r.nextInt(multBound-1)+1;
        int a = (pp*mult)+1;

        // init m
        int m = period;

        // init c
        int c = r.nextInt(m);
        while(gcd(m,c)!=1)
            c = (c+1)%m;

        // init x0
        int x0 = r.nextInt(m);

        // check full-period requirements
        assert(gcd(c,m)==1);
        for(int p : primeFactors(m)) assert((a-1)%p==0);
        assert(m%4!=0 || (a-1)%4==0);
        return new LCG(m, a, c, x0);
    }

    @Override
    public String toString(){
        return "LCG("+m+", "+a+", "+c+", "+x+")";
    }
}
