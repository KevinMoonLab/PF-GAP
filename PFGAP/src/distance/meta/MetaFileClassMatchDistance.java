package distance.meta;

import java.io.*;
import java.util.*;

public class MetaFileClassMatchDistance implements Serializable {
    private final String filePath;
    private final String method; // "class" or "prob"
    private final Map<Integer, Object> predictions = new HashMap<>();

    public MetaFileClassMatchDistance(String descriptor) throws IOException {
        String[] parts = descriptor.split(":");
        if (parts.length < 2) {
            throw new IllegalArgumentException("Invalid descriptor format. Use meta_file_classmatch:path/to/file[:method]");
        }

        this.filePath = parts[1].trim();
        this.method = (parts.length >= 3) ? parts[2].trim().toLowerCase() : "class";

        loadPredictions();
    }

    private void loadPredictions() throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (method.equals("class")) {
                    predictions.put(index++, line);
                } else {
                    String[] tokens = line.replace("[", "").replace("]", "").split(",");
                    double[] probs = new double[tokens.length];
                    for (int i = 0; i < tokens.length; i++) {
                        probs[i] = Double.parseDouble(tokens[i].trim());
                    }
                    predictions.put(index++, probs);
                }
            }
        }
    }

    public double distance(Object T1, Object T2) {
        int i1 = (Integer) T1;
        int i2 = (Integer) T2;

        Object pred1 = predictions.get(i1);
        Object pred2 = predictions.get(i2);

        if (pred1 == null || pred2 == null) {
            throw new IllegalArgumentException("Missing prediction for index " + i1 + " or " + i2);
        }

        if (method.equals("class")) {
            return pred1.equals(pred2) ? 0.0 : 1.0;
        } else {
            double[] p1 = (double[]) pred1;
            double[] p2 = (double[]) pred2;
            return cosine(p1, p2);
        }
    }

    private double cosine(double[] a, double[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return 1.0 - (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}