package purity;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Gini {

    public static double compute(List<Object> labels) {
        Map<Object, Integer> counts = new HashMap<>();
        for (Object label : labels) {
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        double sum = 0;
        int total = labels.size();
        for (int count : counts.values()) {
            double p = (double) count / total;
            sum += p * p;
        }
        return 1 - sum;
    }



}
