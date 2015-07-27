package com.github.mrsdogood.math;

import junit.framework.TestCase;
import java.util.Random;

public class LCGTest extends TestCase {
    public static final long RAND_SEED = 93716349023L;

    public void testPrimeFactors(){
        int m = 62;
        // 62 = 2*31
        int[] factors = LCG.primeFactors(m);
        assertEquals(2, factors.length);
        assertEquals(2, factors[0]);
        assertEquals(31, factors[1]);
    }

    public void testLargeIntegers(){
        LCG lcg = new LCG(60000, 58801, 42761, 17126);
        for(int i = 0; i < 1000; i++)
            assertTrue(lcg.next()>0);
    }
    
    public void _testParticular(int period, long seed){
        Random r = new Random(seed);
        LCG lcg = LCG.makeLCG(period, r);
        int start = lcg.poll();
        int actualPeriod = 1;
        while(lcg.next()!=start)
            actualPeriod++;
        String ident = lcg+" with desired period "+period+" from rand seed "+seed;
        assertEquals(ident+" did not have an actual period with a multiple"+
            " of the desired period.", actualPeriod%period, 0);
        assertTrue(ident+" had an actual period more than 8 times the size of the "+
            "desired period.", actualPeriod/period<=8);
    }

    public void testMany(){
        Random r = new Random(RAND_SEED);
        for(int i = 0; i < 1000; i++){
            int period = r.nextInt(1000)+2;
            long seed = r.nextLong();
            _testParticular(period, seed);
        }
    }
}
