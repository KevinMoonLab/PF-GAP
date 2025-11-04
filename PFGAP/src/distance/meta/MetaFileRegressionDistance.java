package distance.meta;

import java.io.*;
import java.util.*;

public class MetaFileRegressionDistance implements Serializable {
    private final double[] predictions;

    public MetaFileRegressionDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) throw new IllegalArgumentException("Use meta_file_regression:path/to/predictions.csv");

        String filePath = parts[1].trim();
        this.predictions = loadPredictions(filePath);
    }

    private double[] loadPredictions(String filePath) throws IOException {
        List<Double> values = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                values.add(Double.parseDouble(line.trim()));
            }
        }
        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) result[i] = values.get(i);
        return result;
    }

    public double distance(Object T1, Object T2) {
        int i = (Integer) T1;
        int j = (Integer) T2;
        return Math.abs(predictions[i] - predictions[j]);
    }
}