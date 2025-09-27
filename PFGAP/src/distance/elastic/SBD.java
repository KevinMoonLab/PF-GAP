package distance.elastic;

import java.io.Serializable;

public class SBD implements Serializable {

    public double distance(Object Series1, Object Series2) {

        double[] series1 = (double[]) Series1;
        double[] series2 = (double[]) Series2;

        double[] norm1 = zNormalize(series1);
        double[] norm2 = zNormalize(series2);

        double maxCorr = -Double.MAX_VALUE;
        for (int shift = -norm1.length + 1; shift < norm1.length; shift++) {
            double corr = crossCorrelation(norm1, norm2, shift);
            if (corr > maxCorr) {
                maxCorr = corr;
            }
        }

        return 1.0 - maxCorr;
    }

    private double[] zNormalize(double[] series) {
        double mean = 0.0, std = 0.0;
        for (double v : series) mean += v;
        mean /= series.length;
        for (double v : series) std += (v - mean) * (v - mean);
        std = Math.sqrt(std / series.length);

        double[] normalized = new double[series.length];
        for (int i = 0; i < series.length; i++) {
            normalized[i] = (series[i] - mean) / std;
        }
        return normalized;
    }

    private double crossCorrelation(double[] x, double[] y, int shift) {
        int len = x.length;
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            int j = i + shift;
            if (j >= 0 && j < len) {
                sum += x[i] * y[j];
            }
        }
        return sum;
    }
}
