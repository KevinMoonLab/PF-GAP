package imputation.initial;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;

import java.util.List;
import java.util.Objects;

/**
 * Replaces each originally missing numeric value with the mean of the observed
 * values in the same row or dimension. Supports primitive double and float
 * observations and preserves their storage type.
 */
public class MeanImpute extends Imputer {

    @Override
    public void Impute(ListObjectDataset dataset) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        MissingIndices missing = Objects.requireNonNull(
                dataset.getMissingIndices(),
                "MissingIndices must be attached before mean imputation."
        );
        requireInstanceCount(data.size(), missing);
        if (missing.is2D()) {
            impute2D(data, missing);
        } else {
            impute1D(data, missing);
        }
    }

    private static void impute1D(
            List<Object> data,
            MissingIndices missing
    ) {
        for (int instance = 0; instance < data.size(); instance++) {
            Object series = requireSeries(data.get(instance), instance);
            int start = missing.start1D(instance);
            int end = missing.end1D(instance);
            if (series instanceof double[] row) {
                imputeDoubleRow(row, missing, start, end, instance, -1);
            } else if (series instanceof float[] row) {
                imputeFloatRow(row, missing, start, end, instance, -1);
            } else {
                throw unsupportedType(series, instance, false);
            }
        }
    }

    private static void impute2D(
            List<Object> data,
            MissingIndices missing
    ) {
        for (int instance = 0; instance < data.size(); instance++) {
            Object series = requireSeries(data.get(instance), instance);
            int dimensions = missing.dimensionCount(instance);
            if (series instanceof double[][] matrix) {
                requireDimensionCount(matrix.length, dimensions, instance);
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    imputeDoubleRow(
                            Objects.requireNonNull(
                                    matrix[dimension],
                                    nullRowMessage(instance, dimension)
                            ),
                            missing,
                            missing.start2D(instance, dimension),
                            missing.end2D(instance, dimension),
                            instance,
                            dimension
                    );
                }
            } else if (series instanceof float[][] matrix) {
                requireDimensionCount(matrix.length, dimensions, instance);
                for (int dimension = 0; dimension < dimensions; dimension++) {
                    imputeFloatRow(
                            Objects.requireNonNull(
                                    matrix[dimension],
                                    nullRowMessage(instance, dimension)
                            ),
                            missing,
                            missing.start2D(instance, dimension),
                            missing.end2D(instance, dimension),
                            instance,
                            dimension
                    );
                }
            } else {
                throw unsupportedType(series, instance, true);
            }
        }
    }

    private static void imputeDoubleRow(
            double[] row,
            MissingIndices missing,
            int start,
            int end,
            int instance,
            int dimension
    ) {
        double sum = 0.0;
        int count = 0;
        for (double value : row) {
            if (!Double.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        double mean = count == 0 ? 0.0 : sum / count;
        for (int offset = start; offset < end; offset++) {
            int index = missing.positionAt(offset);
            requirePosition(row.length, index, instance, dimension);
            if (!Double.isNaN(row[index])) {
                throw staleIndex(index, instance, dimension);
            }
            row[index] = mean;
        }
    }

    private static void imputeFloatRow(
            float[] row,
            MissingIndices missing,
            int start,
            int end,
            int instance,
            int dimension
    ) {
        double sum = 0.0;
        int count = 0;
        for (float value : row) {
            if (!Float.isNaN(value)) {
                sum += value;
                count++;
            }
        }
        float mean = count == 0 ? 0.0f : (float) (sum / count);
        for (int offset = start; offset < end; offset++) {
            int index = missing.positionAt(offset);
            requirePosition(row.length, index, instance, dimension);
            if (!Float.isNaN(row[index])) {
                throw staleIndex(index, instance, dimension);
            }
            row[index] = mean;
        }
    }

    private static void requireInstanceCount(int dataSize, MissingIndices missing) {
        if (missing.instanceCount() != dataSize) {
            throw new IllegalArgumentException(
                    "Missing metadata contains " + missing.instanceCount()
                            + " instances, but the dataset contains "
                            + dataSize + "."
            );
        }
    }

    private static Object requireSeries(Object series, int instance) {
        if (series == null) {
            throw new IllegalArgumentException(
                    "Numeric series cannot be null at instance " + instance + "."
            );
        }
        return series;
    }

    private static void requireDimensionCount(int actual, int expected, int instance) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "Numeric matrix contains " + actual
                            + " dimensions, but missing metadata contains "
                            + expected + " at instance " + instance + "."
            );
        }
    }

    private static void requirePosition(
            int length,
            int index,
            int instance,
            int dimension
    ) {
        if (index < 0 || index >= length) {
            throw new IllegalArgumentException(
                    "Missing position " + index + " is outside row length "
                            + length + location(instance, dimension) + "."
            );
        }
    }

    private static IllegalStateException staleIndex(
            int index,
            int instance,
            int dimension
    ) {
        return new IllegalStateException(
                "Missing metadata identifies index " + index
                        + location(instance, dimension)
                        + ", but its current value is not NaN."
        );
    }

    private static IllegalArgumentException unsupportedType(
            Object series,
            int instance,
            boolean twoDimensional
    ) {
        return new IllegalArgumentException(
                "Unsupported numeric series type at instance " + instance
                        + ": " + series.getClass().getTypeName()
                        + ". Expected "
                        + (twoDimensional
                                ? "double[][] or float[][]."
                                : "double[] or float[].")
        );
    }

    private static String nullRowMessage(int instance, int dimension) {
        return "Numeric row cannot be null at instance " + instance
                + ", dimension " + dimension + ".";
    }

    private static String location(int instance, int dimension) {
        return " at instance " + instance
                + (dimension < 0 ? "" : ", dimension " + dimension);
    }
}
