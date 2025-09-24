package purity;

import java.util.List;

public class MAD {

    public static double compute(List<Object> labels) {
        if (labels == null || labels.isEmpty()) return 0.0;

        double sum = 0.0;
        for (Object label : labels) {
            sum += ((Number) label).doubleValue();
        }
        double mean = sum / labels.size();

        double mad = 0.0;
        for (Object label : labels) {
            mad += Math.abs(((Number) label).doubleValue() - mean);
        }

        return mad / labels.size();
    }

}
