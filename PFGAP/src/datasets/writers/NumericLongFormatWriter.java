package datasets.writers;

import datasets.ListObjectDataset;
import de.siegmar.fastcsv.writer.CsvWriter;
import preprocessing.standardization.PerSeriesStandardizationState;
import preprocessing.standardization.StandardizationScope;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Objects;

/**
 * FastCSV-backed mirror writer for {@code NumericLongFormatReader}.
 *
 * <p>One output record represents one time point:</p>
 *
 * <pre>
 * id,time,feature_0,...,feature_n,label
 * </pre>
 *
 * <p>Materialized input may be univariate ({@code double[]} or
 * {@code float[]}) or feature-major multivariate ({@code double[][]} or
 * {@code float[][]}). Boxed numeric arrays are intentionally unsupported.
 * Separate instances may have
 * different time lengths. All dimensions within one multivariate instance must
 * have the same time length because each long-form record contains one value
 * from every feature dimension.</p>
 *
 * <p>The reader groups records by ID and optionally sorts by time, but the
 * materialized dataset does not currently retain the original ID and time
 * values. This writer therefore emits deterministic synthetic IDs equal to the
 * zero-based instance index and synthetic time values equal to the zero-based
 * position within the series. This is lossless for the modeled feature and
 * label data, but it does not reproduce source ID/time tokens. Preserving those
 * tokens requires future dataset metadata support.</p>
 *
 * <p>Inverse standardization is fused with record construction. No complete
 * inverse-transformed dataset is allocated and the source arrays are not
 * mutated.</p>
 */
public final class NumericLongFormatWriter implements DatasetWriter {

    public static final String DEFAULT_ID_COLUMN = "id";
    public static final String DEFAULT_TIME_COLUMN = "time";

    private final String idColumn;
    private final String timeColumn;
    private final boolean includeTimeColumn;

    /** Creates a writer using {@code id} and {@code time} metadata columns. */
    public NumericLongFormatWriter() {
        this(DEFAULT_ID_COLUMN, DEFAULT_TIME_COLUMN, true);
    }

    /**
     * Creates a writer with explicit long-form metadata-column names.
     *
     * @param idColumn required group identifier column name
     * @param timeColumn time column name; ignored when includeTimeColumn=false
     * @param includeTimeColumn whether to emit synthetic time positions
     */
    public NumericLongFormatWriter(
            String idColumn,
            String timeColumn,
            boolean includeTimeColumn
    ) {
        this.idColumn = requireColumnName(idColumn, "idColumn");
        this.includeTimeColumn = includeTimeColumn;
        this.timeColumn = includeTimeColumn
                ? requireColumnName(timeColumn, "timeColumn")
                : null;
        if (includeTimeColumn && this.idColumn.equals(this.timeColumn)) {
            throw new IllegalArgumentException(
                    "ID and time columns must have different names."
            );
        }
    }

    @Override
    public Path write(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) throws IOException {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(options, "DatasetWriteOptions cannot be null.");

        List<Object> data = Objects.requireNonNull(
                dataset.getData(),
                "Dataset data cannot be null."
        );
        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot write an empty numeric long-format dataset."
            );
        }
        if (!supports(dataset, options)) {
            Object first = data.get(0);
            throw new IllegalArgumentException(
                    "NUMERIC_LONG_FORMAT output requires homogeneous "
                            + "double[], float[], double[][], or float[][] "
                            + "instances, but "
                            + "received "
                            + (first == null
                            ? "null"
                            : first.getClass().getTypeName())
                            + "."
            );
        }

        validateOptions(options);
        boolean multivariate = isMatrix(data.get(0));
        int featureCount = multivariate
                ? Array.getLength(data.get(0))
                : 1;
        validateFeatureNames(options, featureCount);
        validateLabels(options, data.size());
        validateStates(options.getPerSeriesStates(), data.size());

        PreparedReusableParameters reusable =
                PreparedReusableParameters.from(
                        options.getReusableStatistics(),
                        featureCount
                );
        validateDataset(data, data.get(0).getClass(), featureCount);
        createParentDirectories(options.getOutputPath());

        int metadataFields = 1 + (includeTimeColumn ? 1 : 0);
        int labelFields = options.shouldEmbedLabels() ? 1 : 0;
        String[] record = new String[
                metadataFields + featureCount + labelFields
        ];

        try (CsvWriter writer = CsvWriter.builder()
                .fieldSeparator(options.getEntrySeparatorCharacter())
                .build(
                        options.getOutputPath(),
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                )) {
            // NumericLongFormatReader requires a header, so the mirror writer
            // always emits one regardless of the generic includeHeader flag.
            fillHeader(record, options, featureCount);
            writer.writeRecord(record);

            for (int instanceIndex = 0;
                 instanceIndex < data.size();
                 instanceIndex++) {
                Object instance = data.get(instanceIndex);
                int timeLength = timeLength(instance);
                PerSeriesStandardizationState state =
                        options.getPerSeriesStates().isEmpty()
                                ? null
                                : options.getPerSeriesStates()
                                        .get(instanceIndex);
                validateInverseCompatibility(
                        featureCount,
                        reusable,
                        state,
                        instanceIndex
                );

                for (int timeIndex = 0;
                     timeIndex < timeLength;
                     timeIndex++) {
                    fillRecord(
                            record,
                            includeTimeColumn,
                            instance,
                            instanceIndex,
                            timeIndex,
                            featureCount,
                            options,
                            reusable,
                            state
                    );
                    writer.writeRecord(record);
                }
            }
        }

        return options.getOutputPath();
    }

    @Override
    public boolean supports(
            ListObjectDataset dataset,
            DatasetWriteOptions options
    ) {
        if (dataset == null || options == null || dataset.getData() == null
                || dataset.getData().isEmpty()) {
            return false;
        }
        if (options.getDataLayout()
                != DatasetWriteOptions.DataLayout.LONG_FORM
                && options.getDataLayout()
                != DatasetWriteOptions.DataLayout.AUTO) {
            return false;
        }
        Object first = dataset.getData().get(0);
        if (!isSupportedInstance(first)) {
            return false;
        }
        Class<?> expectedType = first.getClass();
        for (Object instance : dataset.getData()) {
            if (instance == null || instance.getClass() != expectedType) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String formatName() {
        return "NUMERIC_LONG_FORMAT";
    }

    private void fillHeader(
            String[] record,
            DatasetWriteOptions options,
            int featureCount
    ) {
        int index = 0;
        record[index++] = idColumn;
        if (includeTimeColumn) {
            record[index++] = timeColumn;
        }
        for (int feature = 0; feature < featureCount; feature++) {
            record[index++] = options.getFeatureNames().get(feature);
        }
        if (options.shouldEmbedLabels()) {
            record[index] = options.getTargetName();
        }
    }

    private static void fillRecord(
            String[] record,
            boolean includeTimeColumn,
            Object instance,
            int instanceIndex,
            int timeIndex,
            int featureCount,
            DatasetWriteOptions options,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        int index = 0;
        record[index++] = Integer.toString(instanceIndex);
        if (includeTimeColumn) {
            record[index++] = Integer.toString(timeIndex);
        }

        boolean floatBacked = isFloatBacked(instance);
        for (int feature = 0; feature < featureCount; feature++) {
            double value = requireWritableNumeric(
                    doubleValueAt(instance, feature, timeIndex)
            );
            value = inverseIfConfigured(
                    value,
                    feature,
                    reusable,
                    state
            );
            record[index++] = floatBacked
                    ? Float.toString((float) value)
                    : Double.toString(value);
        }

        if (options.shouldEmbedLabels()) {
            Object label = options.getLabels().get(instanceIndex);
            if (label == null) {
                throw new IllegalArgumentException(
                        "Long-format label cannot be null for instance "
                                + instanceIndex
                                + "."
                );
            }
            record[index] = label.toString();
        }
    }

    private static double doubleValueAt(
            Object instance,
            int feature,
            int timeIndex
    ) {
        if (instance instanceof double[] values) {
            return values[timeIndex];
        }
        if (instance instanceof double[][] values) {
            return values[feature][timeIndex];
        }
        if (instance instanceof float[] values) {
            return values[timeIndex];
        }
        if (instance instanceof float[][] values) {
            return values[feature][timeIndex];
        }
        throw new IllegalArgumentException(
                "Unsupported numeric long-format instance type: "
                        + instance.getClass().getTypeName()
        );
    }

    private static boolean isFloatBacked(
            Object instance
    ) {
        return instance instanceof float[] || instance instanceof float[][];
    }

    private static double inverseIfConfigured(
            double value,
            int feature,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state
    ) {
        if (Double.isNaN(value)) {
            return value;
        }
        if (state != null) {
            int group = state.getParameterGroupCount() == 1
                    ? 0
                    : feature;
            return value * state.getScale(group) + state.getCenter(group);
        }
        if (reusable != null) {
            int group = reusable.scope() == StandardizationScope.GLOBAL
                    ? 0
                    : feature;
            return value * reusable.scales()[group]
                    + reusable.centers()[group];
        }
        return value;
    }

    private static void validateOptions(
            DatasetWriteOptions options
    ) {
        if (options.getDataLayout()
                != DatasetWriteOptions.DataLayout.LONG_FORM
                && options.getDataLayout()
                != DatasetWriteOptions.DataLayout.AUTO) {
            throw new IllegalArgumentException(
                    "NumericLongFormatWriter requires LONG_FORM or AUTO "
                            + "layout."
            );
        }
        if (!options.hasFeatureNames()) {
            throw new IllegalArgumentException(
                    "Numeric long-format output requires ordered feature "
                            + "column names."
            );
        }
        if (options.shouldEmbedLabels()
                && options.getTargetPlacement()
                != DatasetWriteOptions.TargetPlacement.LAST) {
            throw new IllegalArgumentException(
                    "NumericLongFormatReader expects selected label columns "
                            + "after long-form metadata and features. Use "
                            + "TargetPlacement.LAST."
            );
        }
    }

    private static void validateFeatureNames(
            DatasetWriteOptions options,
            int featureCount
    ) {
        if (options.getFeatureNames().size() != featureCount) {
            throw new IllegalArgumentException(
                    "Numeric long-format output requires one feature-column "
                            + "name per dimension. Names="
                            + options.getFeatureNames().size()
                            + ", dimensions="
                            + featureCount
                            + "."
            );
        }
    }

    private static void validateLabels(
            DatasetWriteOptions options,
            int instanceCount
    ) {
        if (options.shouldEmbedLabels()
                && options.getLabels().size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Long-format output requires one label per instance. "
                            + "Labels="
                            + options.getLabels().size()
                            + ", instances="
                            + instanceCount
                            + "."
            );
        }
    }

    private static void validateStates(
            List<PerSeriesStandardizationState> states,
            int instanceCount
    ) {
        if (!states.isEmpty() && states.size() != instanceCount) {
            throw new IllegalArgumentException(
                    "Per-series state count "
                            + states.size()
                            + " does not match instance count "
                            + instanceCount
                            + "."
            );
        }
    }

    private static void validateDataset(
            List<Object> data,
            Class<?> expectedType,
            int expectedFeatureCount
    ) {
        for (int instanceIndex = 0;
             instanceIndex < data.size();
             instanceIndex++) {
            Object instance = Objects.requireNonNull(
                    data.get(instanceIndex),
                    "Numeric long-format instance cannot be null at index "
                            + instanceIndex
                            + "."
            );
            if (instance.getClass() != expectedType) {
                throw new IllegalArgumentException(
                        "Numeric long-format output cannot mix instance types. "
                                + "Expected "
                                + expectedType.getTypeName()
                                + " but instance "
                                + instanceIndex
                                + " is "
                                + instance.getClass().getTypeName()
                                + "."
                );
            }
            if (isMatrix(instance)) {
                int features = Array.getLength(instance);
                if (features != expectedFeatureCount) {
                    throw new IllegalArgumentException(
                            "Instance "
                                    + instanceIndex
                                    + " contains "
                                    + features
                                    + " feature dimensions; expected "
                                    + expectedFeatureCount
                                    + "."
                    );
                }
                validateMatrixLengths(instance, instanceIndex);
            } else if (Array.getLength(instance) == 0) {
                throw new IllegalArgumentException(
                        "Univariate instance "
                                + instanceIndex
                                + " is empty."
                );
            }
        }
    }

    private static void validateMatrixLengths(
            Object matrix,
            int instanceIndex
    ) {
        int dimensions = Array.getLength(matrix);
        if (dimensions == 0) {
            throw new IllegalArgumentException(
                    "Multivariate instance "
                            + instanceIndex
                            + " has no feature dimensions."
            );
        }
        int expectedLength = -1;
        for (int feature = 0; feature < dimensions; feature++) {
            Object values = Objects.requireNonNull(
                    Array.get(matrix, feature),
                    "Null feature dimension "
                            + feature
                            + " in instance "
                            + instanceIndex
                            + "."
            );
            int length = Array.getLength(values);
            if (length == 0) {
                throw new IllegalArgumentException(
                        "Empty feature dimension "
                                + feature
                                + " in instance "
                                + instanceIndex
                                + "."
                );
            }
            if (expectedLength < 0) {
                expectedLength = length;
            } else if (length != expectedLength) {
                throw new IllegalArgumentException(
                        "Long-format instance "
                                + instanceIndex
                                + " has inconsistent feature lengths."
                );
            }
        }
    }

    private static void validateInverseCompatibility(
            int featureCount,
            PreparedReusableParameters reusable,
            PerSeriesStandardizationState state,
            int instanceIndex
    ) {
        if (reusable != null
                && reusable.scope() == StandardizationScope.PER_DIMENSION
                && reusable.centers().length != featureCount) {
            throw new IllegalArgumentException(
                    "PER_DIMENSION statistics contain "
                            + reusable.centers().length
                            + " groups, but long-format instances contain "
                            + featureCount
                            + " feature dimensions."
            );
        }
        if (state != null) {
            int groups = state.getParameterGroupCount();
            if (groups != 1 && groups != featureCount) {
                throw new IllegalArgumentException(
                        "Per-series state "
                                + instanceIndex
                                + " contains "
                                + groups
                                + " groups; expected 1 or "
                                + featureCount
                                + "."
                );
            }
        }
    }

    private static int timeLength(
            Object instance
    ) {
        if (instance instanceof double[] values) {
            return values.length;
        }
        if (instance instanceof float[] values) {
            return values.length;
        }
        if (instance instanceof double[][] values) {
            return values[0].length;
        }
        if (instance instanceof float[][] values) {
            return values[0].length;
        }
        throw new IllegalArgumentException(
                "Unsupported numeric long-format instance type: "
                        + instance.getClass().getTypeName()
        );
    }

    private static boolean isSupportedInstance(
            Object instance
    ) {
        return instance instanceof double[]
                || instance instanceof float[]
                || instance instanceof double[][]
                || instance instanceof float[][];
    }

    private static boolean isMatrix(
            Object instance
    ) {
        return instance instanceof double[][]
                || instance instanceof float[][];
    }

    private static double requireWritableNumeric(
            double value
    ) {
        if (Double.isNaN(value)) {
            return value;
        }
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Cannot write a nonfinite long-format value other than "
                            + "NaN: "
                            + value
            );
        }
        return value;
    }

    private static void createParentDirectories(
            Path outputPath
    ) throws IOException {
        Path parent = outputPath.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static String requireColumnName(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    argumentName + " cannot be null or blank."
            );
        }
        return value.trim();
    }

    private record PreparedReusableParameters(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        private static PreparedReusableParameters from(
                StandardizationStats stats,
                int featureCount
        ) {
            if (stats == null) {
                return null;
            }
            StandardizationScope scope = stats.getScope();
            if (!scope.usesTrainingStatistics()) {
                throw new IllegalArgumentException(
                        "Reusable statistics must use GLOBAL or "
                                + "PER_DIMENSION scope."
                );
            }
            double[] centers = stats.getCenters();
            double[] scales = stats.getScales();
            if (centers.length == 0 || centers.length != scales.length) {
                throw new IllegalArgumentException(
                        "Reusable centers and scales must have equal, nonzero "
                                + "lengths."
                );
            }
            if (scope == StandardizationScope.GLOBAL
                    && centers.length != 1) {
                throw new IllegalArgumentException(
                        "GLOBAL inverse transformation requires one group."
                );
            }
            if (scope == StandardizationScope.PER_DIMENSION
                    && centers.length != featureCount) {
                throw new IllegalArgumentException(
                        "PER_DIMENSION statistics contain "
                                + centers.length
                                + " groups, but long-format data contain "
                                + featureCount
                                + " features."
                );
            }
            return new PreparedReusableParameters(
                    scope,
                    centers,
                    scales
            );
        }
    }
}
