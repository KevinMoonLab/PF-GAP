package imputation.results;

import datasets.ListObjectDataset;
import imputation.util.MissingIndices;
import output.SparseTensorWriter;
import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Writes only originally missing multivariate cells as sparse tensor
 * coordinates.
 *
 * <p>This class is intentionally limited to imputed-only output. It does not
 * write or reconstruct a complete dataset. Full-dataset output remains the
 * responsibility of the dataset writers package.</p>
 *
 * <p>Each output entry maps directly from the source data to a logical
 * {@code (instance, dimension, time, value)} record. No dimension/time
 * flattening or intermediate CSR representation is used. The resulting
 * {@code .tns} file is a sparse patch to be interpreted against the original
 * dataset, which permits unequal instance and dimension lengths.</p>
 */
public final class ImputedValuesTensorWriter {

    private ImputedValuesTensorWriter() {
        // Utility class.
    }

    /**
     * Writes originally missing multivariate cells in sparse tensor coordinate
     * form.
     *
     * @param dataset final imputed multivariate dataset
     * @param missingIndices original multivariate missing-position metadata
     * @param reusableStatistics optional reusable global or per-dimension
     *                           standardization statistics
     * @param perSeriesStates optional per-instance standardization states
     * @param path destination {@code .tns} path
     * @param description optional output comment
     * @return normalized absolute output path
     * @throws IOException if writing fails
     */
    public static Path write(
            ListObjectDataset dataset,
            MissingIndices missingIndices,
            StandardizationStats reusableStatistics,
            List<PerSeriesStandardizationState> perSeriesStates,
            Path path,
            String description
    ) throws IOException {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(
                missingIndices,
                "MissingIndices cannot be null."
        );
        Objects.requireNonNull(path, "Sparse tensor path cannot be null.");

        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        List<PerSeriesStandardizationState> states =
                perSeriesStates == null ? List.of() : perSeriesStates;

        validateSources(
                data,
                missingIndices,
                reusableStatistics,
                states
        );

        Shape shape = inspectShape(data);
        PreparedReusable reusable = PreparedReusable.from(
                reusableStatistics,
                shape.maximumDimensionCount()
        );

        return SparseTensorWriter.writeCoordinate(
                path,
                data.size(),
                shape.maximumDimensionCount(),
                shape.maximumTimeLength(),
                missingIndices.missingValueCount(),
                description,
                sink -> writeEntries(
                        data,
                        missingIndices,
                        reusable,
                        states,
                        sink
                )
        );
    }

    private static void writeEntries(
            List<Object> data,
            MissingIndices missing,
            PreparedReusable reusable,
            List<PerSeriesStandardizationState> states,
            SparseTensorWriter.CoordinateSink sink
    ) throws IOException {
        for (int instance = 0; instance < data.size(); instance++) {
            Object item = data.get(instance);
            PerSeriesStandardizationState state = states.isEmpty()
                    ? null
                    : states.get(instance);

            int dimensions = missing.dimensionCount(instance);
            for (int dimension = 0; dimension < dimensions; dimension++) {
                int start = missing.start2D(instance, dimension);
                int end = missing.end2D(instance, dimension);

                for (int offset = start; offset < end; offset++) {
                    int time = missing.positionAt(offset);
                    double value = value2D(
                            item,
                            dimension,
                            time,
                            instance
                    );
                    double outputValue = inverse(
                            value,
                            dimension,
                            reusable,
                            state
                    );
                    if (!Double.isFinite(outputValue)) {
                        throw new IllegalStateException(
                                "Imputed-only tensor output contains a "
                                        + "non-finite value at instance "
                                        + instance
                                        + ", dimension "
                                        + dimension
                                        + ", time "
                                        + time
                                        + ". Imputation may be incomplete."
                        );
                    }

                    sink.write(
                            instance,
                            dimension,
                            time,
                            outputValue
                    );
                }
            }
        }
    }

    private static double inverse(
            double value,
            int dimension,
            PreparedReusable reusable,
            PerSeriesStandardizationState state
    ) {
        if (state != null) {
            int group = state.getParameterGroupCount() == 1
                    ? 0
                    : dimension;
            return value * state.getScale(group)
                    + state.getCenter(group);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0
                    : dimension;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static Shape inspectShape(List<Object> data) {
        int maximumDimensions = 0;
        int maximumTime = 0;

        for (int instance = 0; instance < data.size(); instance++) {
            Object item = data.get(instance);
            int dimensions = dimensionCount(item, instance);
            maximumDimensions = Math.max(maximumDimensions, dimensions);

            for (int dimension = 0;
                    dimension < dimensions;
                    dimension++) {
                maximumTime = Math.max(
                        maximumTime,
                        dimensionLength(item, dimension, instance)
                );
            }
        }

        return new Shape(maximumDimensions, maximumTime);
    }

    private static void validateSources(
            List<Object> data,
            MissingIndices missing,
            StandardizationStats reusable,
            List<PerSeriesStandardizationState> states
    ) {
        if (!missing.is2D()) {
            throw new IllegalArgumentException(
                    "Imputed sparse tensor output requires 2D "
                            + "MissingIndices metadata."
            );
        }
        if (data.size() != missing.instanceCount()) {
            throw new IllegalArgumentException(
                    "Dataset size and MissingIndices instance count differ."
            );
        }
        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot write imputed tensor output from an empty dataset."
            );
        }
        if (reusable != null && !states.isEmpty()) {
            throw new IllegalArgumentException(
                    "Supply reusable statistics or per-series states, not both."
            );
        }
        if (!states.isEmpty() && states.size() != data.size()) {
            throw new IllegalArgumentException(
                    "Per-series state count does not match dataset size."
            );
        }

        Class<?> type = null;
        for (int instance = 0; instance < data.size(); instance++) {
            Object item = Objects.requireNonNull(
                    data.get(instance),
                    "Dataset instance cannot be null at index "
                            + instance
                            + "."
            );
            if (!(item instanceof double[][])
                    && !(item instanceof float[][])
                    && !(item instanceof Object[][])) {
                throw new IllegalArgumentException(
                        "Sparse tensor imputed-only output requires a "
                                + "multivariate array at instance "
                                + instance
                                + ", but received "
                                + item.getClass().getTypeName()
                                + "."
                );
            }
            if (type == null) {
                type = item.getClass();
            } else if (item.getClass() != type) {
                throw new IllegalArgumentException(
                        "Imputed tensor output requires homogeneous "
                                + "instance types."
                );
            }

            int actualDimensions = dimensionCount(item, instance);
            int indexedDimensions = missing.dimensionCount(instance);
            if (actualDimensions != indexedDimensions) {
                throw new IllegalArgumentException(
                        "Dataset and MissingIndices dimension counts differ "
                                + "at instance "
                                + instance
                                + ": data="
                                + actualDimensions
                                + ", missing metadata="
                                + indexedDimensions
                                + "."
                );
            }

            for (int dimension = 0;
                    dimension < indexedDimensions;
                    dimension++) {
                int length = dimensionLength(item, dimension, instance);
                int start = missing.start2D(instance, dimension);
                int end = missing.end2D(instance, dimension);
                if (start < end) {
                    int lastMissingTime = missing.positionAt(end - 1);
                    if (lastMissingTime >= length) {
                        throw new IllegalArgumentException(
                                "Missing time index "
                                        + lastMissingTime
                                        + " is outside dimension length "
                                        + length
                                        + " at instance "
                                        + instance
                                        + ", dimension "
                                        + dimension
                                        + "."
                        );
                    }
                }
            }
        }
    }

    private static double value2D(
            Object item,
            int dimension,
            int time,
            int instance
    ) {
        if (item instanceof double[][] values) {
            return values[dimension][time];
        }
        if (item instanceof float[][] values) {
            return values[dimension][time];
        }
        if (item instanceof Object[][] values) {
            Object value = values[dimension][time];
            if (value instanceof Number number) {
                return number.doubleValue();
            }
            throw new IllegalArgumentException(
                    "Imputed cell is not numeric at instance "
                            + instance
                            + ", dimension "
                            + dimension
                            + ", time "
                            + time
                            + ": "
                            + (value == null
                            ? "null"
                            : value.getClass().getTypeName())
                            + ". Sparse tensor imputed-only output supports "
                            + "numeric imputations only."
            );
        }
        throw new IllegalArgumentException(
                "Unsupported multivariate dataset representation at instance "
                        + instance
                        + "."
        );
    }

    private static int dimensionCount(Object item, int instance) {
        if (item instanceof double[][] values) {
            return values.length;
        }
        if (item instanceof float[][] values) {
            return values.length;
        }
        if (item instanceof Object[][] values) {
            return values.length;
        }
        throw new IllegalArgumentException(
                "Unsupported multivariate dataset representation at instance "
                        + instance
                        + "."
        );
    }

    private static int dimensionLength(
            Object item,
            int dimension,
            int instance
    ) {
        if (item instanceof double[][] values) {
            return Objects.requireNonNull(
                    values[dimension],
                    "Dimension cannot be null at instance "
                            + instance
                            + ", dimension "
                            + dimension
                            + "."
            ).length;
        }
        if (item instanceof float[][] values) {
            return Objects.requireNonNull(
                    values[dimension],
                    "Dimension cannot be null at instance "
                            + instance
                            + ", dimension "
                            + dimension
                            + "."
            ).length;
        }
        if (item instanceof Object[][] values) {
            return Objects.requireNonNull(
                    values[dimension],
                    "Dimension cannot be null at instance "
                            + instance
                            + ", dimension "
                            + dimension
                            + "."
            ).length;
        }
        throw new IllegalArgumentException(
                "Unsupported multivariate dataset representation at instance "
                        + instance
                        + "."
        );
    }

    private record Shape(
            int maximumDimensionCount,
            int maximumTimeLength
    ) {
    }

    private record PreparedReusable(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        private static PreparedReusable from(
                StandardizationStats statistics,
                int dimensionCount
        ) {
            if (statistics == null) {
                return null;
            }

            StandardizationScope scope = statistics.getScope();
            if (!scope.usesTrainingStatistics()) {
                throw new IllegalArgumentException(
                        "Reusable statistics must use GLOBAL or "
                                + "PER_DIMENSION."
                );
            }

            double[] centers = statistics.getCenters();
            double[] scales = statistics.getScales();
            int expected = scope == StandardizationScope.GLOBAL
                    ? 1
                    : dimensionCount;
            if (centers.length != expected || scales.length != expected) {
                throw new IllegalArgumentException(
                        "Reusable parameter count does not match tensor "
                                + "dimensions. Expected "
                                + expected
                                + " but received centers="
                                + centers.length
                                + " and scales="
                                + scales.length
                                + "."
                );
            }

            return new PreparedReusable(scope, centers, scales);
        }
    }
}
