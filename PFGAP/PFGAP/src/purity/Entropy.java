package purity;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Entropy {

    public static double compute(List<Object> labels) {
        if (labels == null || labels.isEmpty()) return 0.0;

        Map<Object, Integer> counts = new HashMap<>();
        for (Object label : labels) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }

        double entropy = 0.0;
        int total = labels.size();

        for (int count : counts.values()) {
            double p = (double) count / total;
            entropy -= p * Math.log(p) / Math.log(2); // log base 2
        }

        return entropy;
    }

}
