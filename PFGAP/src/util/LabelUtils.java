package util;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Utilities for working with scalar and multi-output labels.
 *
 * <p>A scalar label is stored directly. A multi-output label is stored as a
 * {@link List} whose position defines the output dimension. Classification
 * components must be {@link Integer} or {@link String}; regression components
 * may be any finite {@link Number}.</p>
 *
 * <p>This class does not transform, encode, or copy labels. It provides a
 * uniform view over the existing scalar-or-list representation so dataset,
 * splitter, tree, and forest code can remain minimally invasive.</p>
 */
public final class LabelUtils {

    private LabelUtils() {
        throw new AssertionError("LabelUtils cannot be instantiated.");
    }

    /** Returns whether the label represents multiple ordered outputs. */
    public static boolean isMultiOutput(Object label) {
        return label instanceof List<?>;
    }

    /**
     * Returns the number of outputs represented by a non-null label.
     * Scalar labels have one output.
     */
    public static int outputCount(Object label) {
        Objects.requireNonNull(label, "Label cannot be null.");
        if (label instanceof List<?> outputs) {
            if (outputs.isEmpty()) {
                throw new IllegalArgumentException(
                        "A multi-output label cannot be empty."
                );
            }
            return outputs.size();
        }
        return 1;
    }

    /**
     * Returns one output component from a scalar or multi-output label.
     * For a scalar label, only output index zero is valid.
     */
    public static Object valueAt(Object label, int outputIndex) {
        Objects.requireNonNull(label, "Label cannot be null.");
        if (label instanceof List<?> outputs) {
            if (outputs.isEmpty()) {
                throw new IllegalArgumentException(
                        "A multi-output label cannot be empty."
                );
            }
            checkOutputIndex(outputIndex, outputs.size());
            return outputs.get(outputIndex);
        }
        checkOutputIndex(outputIndex, 1);
        return label;
    }

    /**
     * Returns a read-only output view. Scalar labels are exposed as a
     * single-element list; existing multi-output lists are copied defensively.
     */
    public static List<Object> asOutputList(Object label) {
        int count = outputCount(label);
        ArrayList<Object> outputs = new ArrayList<>(count);
        for (int outputIndex = 0; outputIndex < count; outputIndex++) {
            outputs.add(valueAt(label, outputIndex));
        }
        return List.copyOf(outputs);
    }

    /**
     * Validates that every non-null label has the same scalar or multi-output
     * shape and returns the shared output count.
     *
     * <p>Null labels are ignored so unlabeled prediction data can still use the
     * same dataset representation. At least one non-null label is required.</p>
     */
    public static int validateConsistentShape(List<?> labels) {
        Objects.requireNonNull(labels, "Labels cannot be null.");
        int expectedCount = -1;
        boolean expectedMultiOutput = false;

        for (int index = 0; index < labels.size(); index++) {
            Object label = labels.get(index);
            if (label == null) {
                continue;
            }

            int count = outputCount(label);
            boolean multiOutput = isMultiOutput(label);
            if (expectedCount < 0) {
                expectedCount = count;
                expectedMultiOutput = multiOutput;
                continue;
            }

            if (count != expectedCount || multiOutput != expectedMultiOutput) {
                throw new IllegalArgumentException(
                        "Inconsistent label shape at index " + index
                                + ": expected "
                                + shapeDescription(expectedMultiOutput, expectedCount)
                                + " but found "
                                + shapeDescription(multiOutput, count) + "."
                );
            }
        }

        if (expectedCount < 0) {
            throw new IllegalArgumentException(
                    "At least one non-null label is required."
            );
        }
        return expectedCount;
    }

    /** Validates one classification label and all of its output components. */
    public static void validateClassificationLabel(Object label) {
        int count = outputCount(label);
        for (int outputIndex = 0; outputIndex < count; outputIndex++) {
            Object value = valueAt(label, outputIndex);
            if (!(value instanceof Integer) && !(value instanceof String)) {
                throw new IllegalArgumentException(
                        "Classification label output " + outputIndex
                                + " must be an Integer or String, but was "
                                + typeName(value) + "."
                );
            }
        }
    }

    /**
     * Validates consistent classification-label shape and component types.
     * Null labels are ignored, but at least one non-null label is required.
     */
    public static int validateClassificationLabels(List<?> labels) {
        int outputCount = validateConsistentShape(labels);
        for (int index = 0; index < labels.size(); index++) {
            Object label = labels.get(index);
            if (label == null) {
                continue;
            }
            try {
                validateClassificationLabel(label);
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(
                        "Invalid classification label at index " + index + ": "
                                + exception.getMessage(),
                        exception
                );
            }
        }
        return outputCount;
    }

    /** Validates one regression label and all of its output components. */
    public static void validateRegressionLabel(Object label) {
        int count = outputCount(label);
        for (int outputIndex = 0; outputIndex < count; outputIndex++) {
            Object value = valueAt(label, outputIndex);
            if (!(value instanceof Number number)) {
                throw new IllegalArgumentException(
                        "Regression target output " + outputIndex
                                + " must be numeric, but was "
                                + typeName(value) + "."
                );
            }
            double numericValue = number.doubleValue();
            if (!Double.isFinite(numericValue)) {
                throw new IllegalArgumentException(
                        "Regression target output " + outputIndex
                                + " must be finite, but was "
                                + numericValue + "."
                );
            }
        }
    }

    /**
     * Validates consistent regression-label shape and component types.
     * Null labels are ignored, but at least one non-null label is required.
     */
    public static int validateRegressionLabels(List<?> labels) {
        int outputCount = validateConsistentShape(labels);
        for (int index = 0; index < labels.size(); index++) {
            Object label = labels.get(index);
            if (label == null) {
                continue;
            }
            try {
                validateRegressionLabel(label);
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(
                        "Invalid regression label at index " + index + ": "
                                + exception.getMessage(),
                        exception
                );
            }
        }
        return outputCount;
    }

    /**
     * Returns classification output dimensions having at least two distinct
     * values in the supplied labeled observations.
     */
    public static List<Integer> splittableClassificationOutputs(
            List<?> labels
    ) {
        int outputCount = validateClassificationLabels(labels);
        ArrayList<Set<Object>> uniqueValues = new ArrayList<>(outputCount);
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            uniqueValues.add(new LinkedHashSet<>());
        }

        for (Object label : labels) {
            if (label == null) {
                continue;
            }
            for (int outputIndex = 0;
                 outputIndex < outputCount;
                 outputIndex++) {
                uniqueValues.get(outputIndex).add(
                        valueAt(label, outputIndex)
                );
            }
        }

        ArrayList<Integer> splittable = new ArrayList<>();
        for (int outputIndex = 0;
             outputIndex < outputCount;
             outputIndex++) {
            if (uniqueValues.get(outputIndex).size() > 1) {
                splittable.add(outputIndex);
            }
        }
        return List.copyOf(splittable);
    }

    /**
     * Returns true when every classification output is constant across all
     * non-null labels.
     */
    public static boolean isClassificationPure(List<?> labels) {
        return splittableClassificationOutputs(labels).isEmpty();
    }

    private static void checkOutputIndex(int outputIndex, int outputCount) {
        if (outputIndex < 0 || outputIndex >= outputCount) {
            throw new IndexOutOfBoundsException(
                    "Output index " + outputIndex
                            + " is outside [0, " + outputCount + ")."
            );
        }
    }

    private static String shapeDescription(
            boolean multiOutput,
            int outputCount
    ) {
        return multiOutput
                ? "a multi-output label with " + outputCount + " outputs"
                : "a scalar label";
    }

    private static String typeName(Object value) {
        return value == null
                ? "null"
                : value.getClass().getName();
    }
}
