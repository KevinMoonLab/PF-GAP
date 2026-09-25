package distance.meta;

import core.AppContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Classification meta-distance backed by predictions loaded from a text file.
 *
 * <p>The behavioral contract is unchanged:</p>
 * <ul>
 *     <li>{@code class}: zero for equal prediction labels, one otherwise;</li>
 *     <li>{@code prob}: cosine distance between prediction vectors.</li>
 * </ul>
 *
 * <p>Input observations identify prediction rows through their first value.
 * Matching {@code double[]} and {@code float[]} observations are supported.</p>
 */
public final class MetaFileClassMatchDistance implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final String CLASS_METHOD = "class";
    private static final String PROBABILITY_METHOD = "prob";

    private final String filePath;
    private final String method;

    public MetaFileClassMatchDistance(String descriptor) throws IOException {
        String[] parts = parseDescriptor(descriptor);
        filePath = parts[1].trim();
        method = parts.length >= 3 && !parts[2].isBlank()
                ? parts[2].trim().toLowerCase(Locale.ROOT)
                : CLASS_METHOD;
        validateMethod(method);

        synchronized (AppContext.class) {
            if (AppContext.meta_predictions == null) {
                AppContext.meta_predictions = new HashMap<>();
                loadPredictions(AppContext.meta_predictions);
            }
        }
    }

    public double distance(Object first, Object second) {
        int firstIndex = predictionIndex(first);
        int secondIndex = predictionIndex(second);

        Object firstPrediction = AppContext.meta_predictions.get(firstIndex);
        Object secondPrediction = AppContext.meta_predictions.get(secondIndex);
        if (firstPrediction == null || secondPrediction == null) {
            throw new IllegalArgumentException(
                    "Missing prediction for index "
                            + firstIndex + " or " + secondIndex + "."
            );
        }

        if (CLASS_METHOD.equals(method)) {
            return firstPrediction.equals(secondPrediction) ? 0.0 : 1.0;
        }

        return cosine(
                (double[]) firstPrediction,
                (double[]) secondPrediction
        );
    }

    private void loadPredictions(Map<Integer, Object> predictions)
            throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Path.of(filePath))) {
            String line;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                predictions.put(
                        index++,
                        CLASS_METHOD.equals(method)
                                ? trimmed
                                : parseProbabilityVector(trimmed)
                );
            }
        }
    }

    private static int predictionIndex(Object input) {
        if (input instanceof double[] values) {
            requireIdentifier(values.length, "double[]");
            return (int) values[0];
        }
        if (input instanceof float[] values) {
            requireIdentifier(values.length, "float[]");
            return (int) values[0];
        }
        throw new IllegalArgumentException(
                "MetaFileClassMatchDistance requires double[] or float[] "
                        + "input. Received "
                        + (input == null
                        ? "null"
                        : input.getClass().getTypeName())
                        + "."
        );
    }

    private static void requireIdentifier(int length, String type) {
        if (length == 0) {
            throw new IllegalArgumentException(
                    "Meta-file input " + type
                            + " must contain a prediction index at position 0."
            );
        }
    }

    private static double[] parseProbabilityVector(String value) {
        String cleaned = value.trim();
        if (cleaned.startsWith("[") && cleaned.endsWith("]")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1).trim();
        }
        if (cleaned.isEmpty()) {
            return new double[0];
        }

        String[] tokens = cleaned.split(",");
        double[] probabilities = new double[tokens.length];
        for (int index = 0; index < tokens.length; index++) {
            probabilities[index] = Double.parseDouble(tokens[index].trim());
        }
        return probabilities;
    }

    private static double cosine(double[] first, double[] second) {
        if (first.length != second.length) {
            throw new IllegalArgumentException(
                    "Probability vectors must have equal lengths. Received "
                            + first.length + " and " + second.length + "."
            );
        }

        double dotProduct = 0.0;
        double firstNorm = 0.0;
        double secondNorm = 0.0;
        for (int index = 0; index < first.length; index++) {
            double firstValue = first[index];
            double secondValue = second[index];
            dotProduct += firstValue * secondValue;
            firstNorm += firstValue * firstValue;
            secondNorm += secondValue * secondValue;
        }
        return 1.0 - dotProduct
                / (Math.sqrt(firstNorm) * Math.sqrt(secondNorm));
    }

    private static String[] parseDescriptor(String descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException(
                    "Meta-file descriptor cannot be null."
            );
        }
        String[] parts = descriptor.split(":", 3);
        if (parts.length < 2 || parts[1].isBlank()) {
            throw new IllegalArgumentException(
                    "Use meta_file_classmatch:path/to/file[:method]"
            );
        }
        return parts;
    }

    private static void validateMethod(String method) {
        if (!CLASS_METHOD.equals(method)
                && !PROBABILITY_METHOD.equals(method)) {
            throw new IllegalArgumentException(
                    "Unknown meta-file class method: " + method + "."
            );
        }
    }
}
