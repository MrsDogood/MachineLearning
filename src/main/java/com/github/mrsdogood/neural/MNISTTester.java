package com.github.mrsdogood.neural;

public class MNISTTester {
    // TODO
    /*
    private static int getDigitsOff(RowD1Matrix64F actualOut, int expectedDigit){
        double correctDigit = actualOut.get(expectedDigit);
        int digitsOff = 0;
        for(int i = 0; i < 10; i++){
            if(i!=expectedDigit && actualOut.get(i)>=correctDigit)
                digitsOff++;
        }
        return digitsOff;
    }
    private static int getDigit(double[] v){
        int d = 0;
        for(int i = 1; i < v.length; i++){
            if(v[d] < v[i])
                d = i;
        }
        return d;
    }
    public String getErrorReport(RowD1Matrix64F x){
        nn.setWeights(x);
        int[] expectedDigits = new int[10];
        int[] errorsByDigit = new int[10];
        int[] totalDigitsOff = new int[10];
        int totalErrors = 0;
        for(int i = 0; i < trainingInputs.size(); i++){
            int expectedDigit = getDigit(trainingOutputs.get(i));
            expectedDigits[expectedDigit]++;
            copy(trainingInputs.get(i), nn.getInputLayer());
            nn.propagate();
            RowD1Matrix64F actualOutput = nn.getOutputLayer();
            int actualDigit = getDigit(actualOutput.getData());
            if(actualDigit==expectedDigit){
                totalDigitsOff[0]++;
            } else {
                errorsByDigit[expectedDigit]++;
                totalErrors++;
                int digitsOff = getDigitsOff(actualOutput, expectedDigit);
                totalDigitsOff[digitsOff]++;
            }
        }
        double errorRate = totalErrors/(double)trainingInputs.size();

        StringBuilder out = new StringBuilder();
        out.append("Total errors: "+totalErrors+"\n");
        out.append("Error rate: "+errorRate+"\n");
        out.append("Errors by expected digit:\n");
        for(int i = 0; i<10; i++){
            out.append("\t"+i+") "+errorsByDigit[i]);
            out.append(" of "+expectedDigits[i]);
            double ratio = errorsByDigit[i]/(double)expectedDigits[i];
            out.append("\t"+ratio+"\n");
        }
        out.append("Total guessed digits off:\n");
        for(int i = 0; i<10; i++){
            out.append("\t"+i+") "+totalDigitsOff[i]+"\n");
        }
        return out.toString();
    }
    */
}
