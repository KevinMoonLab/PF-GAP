package purity;

import java.util.List;

public class Variance {

    public static double compute(List<Object> labels) {
        double mean = 0;
        int n = 0;
        for (Object label : labels) {
            if (label instanceof Number) {
                mean += ((Number) label).doubleValue();
                n++;
            }
        }
        mean /= n;

        double variance = 0;
        for (Object label : labels) {
            if (label instanceof Number) {
                double val = ((Number) label).doubleValue();
                variance += (val - mean) * (val - mean);
            }
        }
        return variance / n;
    }

}
