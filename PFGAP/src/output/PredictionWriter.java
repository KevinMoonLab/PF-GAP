package output;

import util.LabelUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Writes scalar or multi-output classification and regression predictions as
 * indexed CSV records.
 *
 * <p>Scalar outputs retain the existing CSV layout. Multi-output values are
 * expanded into positional columns such as {@code prediction_0} and
 * {@code actual_0}. Classification correctness uses exact equality across the
 * complete scalar or ordered multi-output label.</p>
 */
public final class PredictionWriter {

    private PredictionWriter() {
    }

    public static Path writeClassification(
            Path path,
            List<?> predictions,
            List<?> actualLabels
    ) throws IOException {
        validateCommonArguments(path, predictions, actualLabels);
        int outputCount = validateClassificationValues(
                predictions,
                "prediction",
                false
        );
        boolean includeActual = hasActualValues(actualLabels);
        if (includeActual) {
            int actualOutputCount = validateClassificationValues(
                    actualLabels,
                    "actual label",
                    true
            );
            requireMatchingOutputCounts(outputCount, actualOutputCount);
        }

        Path outputPath = normalizePath(path);
        try (BufferedWriter writer = CsvUtils.newWriter(outputPath)) {
            CsvUtils.writeHeader(
                    writer,
                    classificationHeader(outputCount, includeActual)
            );
            for (int instanceIndex = 0;
                 instanceIndex < predictions.size();
                 instanceIndex++) {
                Object prediction = predictions.get(instanceIndex);
                Object actual = includeActual
                        ? actualLabels.get(instanceIndex)
                        : null;
                List<Object> row = new ArrayList<>();
                row.add(instanceIndex);

                if (outputCount == 1) {
                    row.add(prediction);
                    if (includeActual) {
                        row.add(actual);
                        row.add(actual == null
                                ? null
                                : Objects.equals(prediction, actual));
                    }
                } else {
                    for (int outputIndex = 0;
                         outputIndex < outputCount;
                         outputIndex++) {
                        row.add(LabelUtils.valueAt(prediction, outputIndex));
                        if (includeActual) {
                            row.add(actual == null
                                    ? null
                                    : LabelUtils.valueAt(actual, outputIndex));
                        }
                    }
                    if (includeActual) {
                        row.add(actual == null
                                ? null
                                : Objects.equals(prediction, actual));
                    }
                }
                CsvUtils.writeRow(writer, row);
            }
        }
        return outputPath;
    }

    public static Path writeClassification(
            Path path,
            List<?> predictions
    ) throws IOException {
        return writeClassification(path, predictions, null);
    }

    public static Path writeRegression(
            Path path,
            List<?> predictions,
            List<?> actualTargets
    ) throws IOException {
        validateCommonArguments(path, predictions, actualTargets);
        int outputCount = validateRegressionValues(
                predictions,
                "prediction",
                false
        );
        boolean includeActual = hasActualValues(actualTargets);
        if (includeActual) {
            int actualOutputCount = validateRegressionValues(
                    actualTargets,
                    "actual target",
                    true
            );
            requireMatchingOutputCounts(outputCount, actualOutputCount);
        }

        Path outputPath = normalizePath(path);
        try (BufferedWriter writer = CsvUtils.newWriter(outputPath)) {
            CsvUtils.writeHeader(
                    writer,
                    regressionHeader(outputCount, includeActual)
            );
            for (int instanceIndex = 0;
                 instanceIndex < predictions.size();
                 instanceIndex++) {
                Object prediction = predictions.get(instanceIndex);
                Object actual = includeActual
                        ? actualTargets.get(instanceIndex)
                        : null;
                List<Object> row = new ArrayList<>();
                row.add(instanceIndex);

                for (int outputIndex = 0;
                     outputIndex < outputCount;
                     outputIndex++) {
                    double predictedValue = numericValueAt(
                            prediction,
                            outputIndex
                    );
                    row.add(CsvUtils.numericField(predictedValue));
                    if (!includeActual) {
                        continue;
                    }
                    if (actual == null) {
                        row.add(null);
                        row.add(null);
                        row.add(null);
                        continue;
                    }
                    double actualValue = numericValueAt(actual, outputIndex);
                    double residual = predictedValue - actualValue;
                    row.add(CsvUtils.numericField(actualValue));
                    row.add(CsvUtils.numericField(residual));
                    row.add(CsvUtils.numericField(Math.abs(residual)));
                }
                CsvUtils.writeRow(writer, row);
            }
        }
        return outputPath;
    }

    public static Path writeRegression(
            Path path,
            List<?> predictions
    ) throws IOException {
        return writeRegression(path, predictions, null);
    }

    public static Path write(
            Path path,
            List<?> predictions,
            List<?> actualValues,
            boolean regression
    ) throws IOException {
        return regression
                ? writeRegression(path, predictions, actualValues)
                : writeClassification(path, predictions, actualValues);
    }

    public static Path write(
            Path path,
            List<?> predictions,
            boolean regression
    ) throws IOException {
        return write(path, predictions, null, regression);
    }

    private static List<String> classificationHeader(
            int outputCount,
            boolean includeActual
    ) {
        List<String> columns = new ArrayList<>();
        columns.add("instance_index");
        if (outputCount == 1) {
            columns.add("prediction");
            if (includeActual) {
                columns.add("actual");
                columns.add("correct");
            }
            return List.copyOf(columns);
        }
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            columns.add("prediction_" + outputIndex);
            if (includeActual) {
                columns.add("actual_" + outputIndex);
            }
        }
        if (includeActual) {
            columns.add("correct");
        }
        return List.copyOf(columns);
    }

    private static List<String> regressionHeader(
            int outputCount,
            boolean includeActual
    ) {
        List<String> columns = new ArrayList<>();
        columns.add("instance_index");
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            String suffix = outputCount == 1
                    ? ""
                    : "_" + outputIndex;
            columns.add("prediction" + suffix);
            if (includeActual) {
                columns.add("actual" + suffix);
                columns.add("residual" + suffix);
                columns.add("absolute_error" + suffix);
            }
        }
        return List.copyOf(columns);
    }

    private static void validateCommonArguments(
            Path path,
            List<?> predictions,
            List<?> actualValues
    ) {
        Objects.requireNonNull(path, "Output path cannot be null.");
        Objects.requireNonNull(predictions, "Predictions cannot be null.");
        if (predictions.isEmpty()) {
            throw new IllegalArgumentException("Predictions cannot be empty.");
        }
        if (actualValues != null
                && !actualValues.isEmpty()
                && actualValues.size() != predictions.size()) {
            throw new IllegalArgumentException(
                    "Actual-value count must match prediction count."
            );
        }
    }

    private static int validateClassificationValues(
            List<?> values,
            String role,
            boolean allowNull
    ) {
        int expectedOutputCount = -1;
        Boolean expectedMultiOutput = null;
        for (int index = 0; index < values.size(); index++) {
            Object value = values.get(index);
            if (value == null) {
                if (allowNull) {
                    continue;
                }
                throw invalidValue(role, index, value);
            }
            try {
                LabelUtils.validateClassificationLabel(value);
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(
                        "Invalid classification " + role + " at instance "
                                + index + ": " + exception.getMessage(),
                        exception
                );
            }
            int outputCount = LabelUtils.outputCount(value);
            boolean multiOutput = LabelUtils.isMultiOutput(value);
            if (expectedOutputCount < 0) {
                expectedOutputCount = outputCount;
                expectedMultiOutput = multiOutput;
            } else if (outputCount != expectedOutputCount
                    || multiOutput != expectedMultiOutput) {
                throw inconsistentShape(role, index);
            }
        }
        if (expectedOutputCount < 0) {
            throw new IllegalArgumentException(
                    "At least one non-null " + role + " is required."
            );
        }
        return expectedOutputCount;
    }

    private static int validateRegressionValues(
            List<?> values,
            String role,
            boolean allowNull
    ) {
        int expectedOutputCount = -1;
        Boolean expectedMultiOutput = null;
        for (int index = 0; index < values.size(); index++) {
            Object value = values.get(index);
            if (value == null) {
                if (allowNull) {
                    continue;
                }
                throw invalidValue(role, index, value);
            }
            try {
                LabelUtils.validateRegressionLabel(value);
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(
                        "Invalid regression " + role + " at instance "
                                + index + ": " + exception.getMessage(),
                        exception
                );
            }
            int outputCount = LabelUtils.outputCount(value);
            boolean multiOutput = LabelUtils.isMultiOutput(value);
            if (expectedOutputCount < 0) {
                expectedOutputCount = outputCount;
                expectedMultiOutput = multiOutput;
            } else if (outputCount != expectedOutputCount
                    || multiOutput != expectedMultiOutput) {
                throw inconsistentShape(role, index);
            }
        }
        if (expectedOutputCount < 0) {
            throw new IllegalArgumentException(
                    "At least one non-null " + role + " is required."
            );
        }
        return expectedOutputCount;
    }

    private static boolean hasActualValues(List<?> actualValues) {
        if (actualValues == null || actualValues.isEmpty()) {
            return false;
        }
        for (Object actualValue : actualValues) {
            if (actualValue != null) {
                return true;
            }
        }
        return false;
    }

    private static double numericValueAt(Object value, int outputIndex) {
        return ((Number) LabelUtils.valueAt(value, outputIndex)).doubleValue();
    }

    private static void requireMatchingOutputCounts(
            int predictionOutputCount,
            int actualOutputCount
    ) {
        if (predictionOutputCount != actualOutputCount) {
            throw new IllegalArgumentException(
                    "Actual values have " + actualOutputCount
                            + " outputs, but predictions have "
                            + predictionOutputCount + "."
            );
        }
    }

    private static IllegalArgumentException invalidValue(
            String role,
            int index,
            Object value
    ) {
        return new IllegalArgumentException(
                role + " at instance " + index
                        + " is invalid: " + describeValue(value) + "."
        );
    }

    private static IllegalArgumentException inconsistentShape(
            String role,
            int index
    ) {
        return new IllegalArgumentException(
                role + " at instance " + index
                        + " has an inconsistent output shape."
        );
    }

    private static Path normalizePath(Path path) {
        return path.toAbsolutePath().normalize();
    }

    private static String describeValue(Object value) {
        if (value == null) {
            return "null";
        }
        return "'" + value + "' of type "
                + value.getClass().getName();
    }
}