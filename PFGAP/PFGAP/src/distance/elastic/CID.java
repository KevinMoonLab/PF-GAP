package distance.elastic;

import java.io.Serializable;

public class CID implements Serializable {

    public double distance(Object Series1, Object Series2, double bsf) {

        double[] series1 = (double[]) Series1;
        double[] series2 = (double[]) Series2;

        if (series1.length != series2.length) {
            throw new IllegalArgumentException("Series must be of equal length.");
        }

        double ed = 0.0;
        for (int i = 0; i < series1.length; i++) {
            double diff = series1[i] - series2[i];
            ed += diff * diff;
        }
        ed = Math.sqrt(ed);

        double c1 = complexity(series1);
        double c2 = complexity(series2);
        double correction = Math.max(c1, c2) / Math.min(c1, c2);

        return ed * correction;
    }

    private double complexity(double[] series) {
        double sum = 0.0;
        for (int i = 1; i < series.length; i++) {
            double diff = series[i] - series[i - 1];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}