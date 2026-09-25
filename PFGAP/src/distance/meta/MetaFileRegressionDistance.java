package distance.meta;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Regression meta-distance backed by scalar predictions loaded from a file.
 *
 * <p>The distance is unchanged: the absolute difference between predictions at
 * the two supplied observation indices.</p>
 *
 * <p>For compatibility with the classification file-backed meta-distance and
 * current primitive dataset storage, an index may be supplied as an
 * {@link Integer}, a nonempty {@code double[]}, or a nonempty {@code float[]}.
 * Primitive arrays use their first value as the prediction-row index.</p>
 */
public final class MetaFileRegressionDistance implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final double[] predictions;

    public MetaFileRegressionDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        predictions = loadPredictions(parts[1].trim());
    }

    public double distance(Object first, Object second) {
        int firstIndex = predictionIndex(first);
        int secondIndex = predictionIndex(second);
        requirePredictionIndex(firstIndex);
        requirePredictionIndex(secondIndex);
        return Math.abs(
                predictions[firstIndex] - predictions[secondIndex]
        );
    }

    private static double[] loadPredictions(String filePath)
            throws IOException {
        List<Double> values = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(Path.of(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                values.add(Double.parseDouble(line.trim()));
            }
        }

        double[] result = new double[values.size()];
        for (int index = 0; index < values.size(); index++) {
            result[index] = values.get(index);
        }
        return result;
    }

    private static int predictionIndex(Object input) {
        if (input instanceof Integer index) {
            return index;
        }
        if (input instanceof double[] values) {
            requireIdentifier(values.length, "double[]");
            return (int) values[0];
        }
        if (input instanceof float[] values) {
            requireIdentifier(values.length, "float[]");
            return (int) values[0];
        }
        throw new IllegalArgumentException(
                "MetaFileRegressionDistance requires Integer, double[], or "
                        + "float[] input. Received "
                        + (input == null
                        ? "null"
                        : input.getClass().getTypeName())
                        + "."
        );
    }

    private static void requireIdentifier(int length, String type) {
        if (length == 0) {
            throw new IllegalArgumentException(
                    "Meta-file regression input " + type
                            + " must contain a prediction index at position 0."
            );
        }
    }

    private void requirePredictionIndex(int index) {
        if (index < 0 || index >= predictions.length) {
            throw new IllegalArgumentException(
                    "Prediction index " + index
                            + " is outside [0, "
                            + predictions.length + ")."
            );
        }
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Meta-file regression descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 2);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use meta_file_regression:path/to/predictions.csv"
            );
        }
        return parts;
    }
}
