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
 * FastCSV-backed mirror writer for generic long-format delimited data.
 *
 * <p>The writer supports generic {@code Object[]} and {@code Object[][]}
 * instances as well as primitive {@code double[]}, {@code float[]},
 * {@code double[][]}, and {@code float[][]} instances. Primitive numeric arrays
 * remain the required representation for datasets read with a numeric flag;
 * object arrays are reserved for genuinely generic or heterogeneous data.</p>
 *
 * <p>One output record represents one time position. Grouped output uses a
 * synthetic zero-based instance ID and synthetic zero-based time position,
 * because the current materialized dataset does not retain original source ID
 * and time tokens.</p>
 */
public final class LongFormatWriter implements DatasetWriter {

    private final String idColumn;
    private final String timeColumn;
    private final boolean includeTimeColumn;

    public LongFormatWriter() {
        this("id", "time", true);
    }

    public LongFormatWriter(
            String idColumn,
            String timeColumn,
            boolean includeTimeColumn
    ) {
        this.idColumn = requireName(idColumn, "idColumn");
        this.includeTimeColumn = includeTimeColumn;
        this.timeColumn = includeTimeColumn
                ? requireName(timeColumn, "timeColumn")
                : null;
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
                    "Cannot write an empty long-format dataset."
            );
        }
        if (!supports(dataset, options)) {
            throw new IllegalArgumentException(
                    "LONG_FORMAT output requires homogeneous supported vector "
                            + "or matrix instances and LONG_FORM or AUTO layout."
            );
        }
        if (!options.hasFeatureNames()) {
            throw new IllegalArgumentException(
                    "Long-format output requires ordered feature names."
            );
        }

        Object first = data.get(0);
        boolean matrix = isMatrix(first);
        int featureCount = matrix ? Array.getLength(first) : 1;
        if (options.getFeatureNames().size() != featureCount) {
            throw new IllegalArgumentException(
                    "Feature-name count "
                            + options.getFeatureNames().size()
                            + " does not match feature count "
                            + featureCount
                            + "."
            );
        }
        validateLabels(options, data.size());
        validateStates(options.getPerSeriesStates(), data.size());
        PreparedReusable reusable = PreparedReusable.from(
                options.getReusableStatistics(),
                featureCount
        );
        validateDataset(data, first.getClass(), featureCount);
        createParentDirectories(options.getOutputPath());

        int metadataCount = 1 + (includeTimeColumn ? 1 : 0);
        String[] record = new String[
                metadataCount
                        + featureCount
                        + (options.shouldEmbedLabels() ? 1 : 0)
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
            fillHeader(record, options, featureCount);
            writer.writeRecord(record);

            for (int instanceIndex = 0;
                 instanceIndex < data.size();
                 instanceIndex++) {
                Object instance = data.get(instanceIndex);
                PerSeriesStandardizationState state =
                        options.getPerSeriesStates().isEmpty()
                                ? null
                                : options.getPerSeriesStates().get(instanceIndex);
                validateState(state, featureCount, instanceIndex);
                int length = timeLength(instance);
                for (int timeIndex = 0; timeIndex < length; timeIndex++) {
                    fillRecord(
                            record,
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
        if (options.getDataLayout() != DatasetWriteOptions.DataLayout.AUTO
                && options.getDataLayout()
                != DatasetWriteOptions.DataLayout.LONG_FORM) {
            return false;
        }
        Object first = dataset.getData().get(0);
        if (!supported(first)) {
            return false;
        }
        Class<?> type = first.getClass();
        for (Object instance : dataset.getData()) {
            if (instance == null || instance.getClass() != type) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String formatName() {
        return "LONG_FORMAT";
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

    private void fillRecord(
            String[] record,
            Object instance,
            int instanceIndex,
            int timeIndex,
            int featureCount,
            DatasetWriteOptions options,
            PreparedReusable reusable,
            PerSeriesStandardizationState state
    ) {
        int index = 0;
        record[index++] = Integer.toString(instanceIndex);
        if (includeTimeColumn) {
            record[index++] = Integer.toString(timeIndex);
        }
        for (int feature = 0; feature < featureCount; feature++) {
            Object value = valueAt(instance, feature, timeIndex);
            record[index++] = formatValue(value, feature, reusable, state);
        }
        if (options.shouldEmbedLabels()) {
            Object label = options.getLabels().get(instanceIndex);
            record[index] = label == null ? "" : label.toString();
        }
    }

    private static String formatValue(
            Object value,
            int feature,
            PreparedReusable reusable,
            PerSeriesStandardizationState state
    ) {
        if (value == null) {
            return "";
        }
        if (!(value instanceof Number number)) {
            return value.toString();
        }
        double numeric = number.doubleValue();
        if (Double.isInfinite(numeric)) {
            throw new IllegalArgumentException(
                    "Cannot write infinite long-format numeric value."
            );
        }
        if (!Double.isNaN(numeric)) {
            if (state != null) {
                int group = state.getParameterGroupCount() == 1 ? 0 : feature;
                numeric = numeric * state.getScale(group)
                        + state.getCenter(group);
            } else if (reusable != null) {
                int group = reusable.scope() == StandardizationScope.GLOBAL
                        ? 0
                        : feature;
                numeric = numeric * reusable.scales()[group]
                        + reusable.centers()[group];
            }
        }
        return value instanceof Float
                ? Float.toString((float) numeric)
                : Double.toString(numeric);
    }

    private static Object valueAt(
            Object instance,
            int feature,
            int timeIndex
    ) {
        if (isMatrix(instance)) {
            return Array.get(Array.get(instance, feature), timeIndex);
        }
        return Array.get(instance, timeIndex);
    }

    private static int timeLength(Object instance) {
        return isMatrix(instance)
                ? Array.getLength(Array.get(instance, 0))
                : Array.getLength(instance);
    }

    private static void validateDataset(
            List<Object> data,
            Class<?> type,
            int featureCount
    ) {
        for (int instanceIndex = 0;
             instanceIndex < data.size();
             instanceIndex++) {
            Object instance = Objects.requireNonNull(data.get(instanceIndex));
            if (instance.getClass() != type) {
                throw new IllegalArgumentException(
                        "Long-format output cannot mix instance types."
                );
            }
            if (isMatrix(instance)) {
                if (Array.getLength(instance) != featureCount) {
                    throw new IllegalArgumentException(
                            "Feature count changed at instance "
                                    + instanceIndex
                                    + "."
                    );
                }
                int length = -1;
                for (int feature = 0; feature < featureCount; feature++) {
                    Object dimension = Objects.requireNonNull(
                            Array.get(instance, feature),
                            "Null feature dimension."
                    );
                    int actual = Array.getLength(dimension);
                    if (actual == 0) {
                        throw new IllegalArgumentException(
                                "Long-format dimensions cannot be empty."
                        );
                    }
                    if (length < 0) {
                        length = actual;
                    } else if (length != actual) {
                        throw new IllegalArgumentException(
                                "Feature lengths differ within instance "
                                        + instanceIndex
                                        + "."
                        );
                    }
                }
            } else if (Array.getLength(instance) == 0) {
                throw new IllegalArgumentException(
                        "Long-format vectors cannot be empty."
                );
            }
        }
    }

    private static void validateState(
            PerSeriesStandardizationState state,
            int featureCount,
            int instanceIndex
    ) {
        if (state == null) {
            return;
        }
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

    private static void validateStates(
            List<PerSeriesStandardizationState> states,
            int size
    ) {
        if (!states.isEmpty() && states.size() != size) {
            throw new IllegalArgumentException(
                    "Per-series state count does not match dataset size."
            );
        }
    }

    private static void validateLabels(
            DatasetWriteOptions options,
            int size
    ) {
        if (options.shouldEmbedLabels()
                && options.getLabels().size() != size) {
            throw new IllegalArgumentException(
                    "Embedded-label count does not match dataset size."
            );
        }
    }

    private static boolean supported(Object instance) {
        return instance instanceof Object[]
                || instance instanceof Object[][]
                || instance instanceof double[]
                || instance instanceof double[][]
                || instance instanceof float[]
                || instance instanceof float[][];
    }

    private static boolean isMatrix(Object instance) {
        return instance instanceof Object[][]
                || instance instanceof double[][]
                || instance instanceof float[][];
    }

    private static void createParentDirectories(Path path) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static String requireName(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " cannot be blank.");
        }
        return value.trim();
    }

    private record PreparedReusable(
            StandardizationScope scope,
            double[] centers,
            double[] scales
    ) {
        private static PreparedReusable from(
                StandardizationStats stats,
                int featureCount
        ) {
            if (stats == null) {
                return null;
            }
            StandardizationScope scope = stats.getScope();
            if (!scope.usesTrainingStatistics()) {
                throw new IllegalArgumentException(
                        "Reusable output statistics must use GLOBAL or "
                                + "PER_DIMENSION scope."
                );
            }
            double[] centers = stats.getCenters();
            double[] scales = stats.getScales();
            int expected = scope == StandardizationScope.GLOBAL
                    ? 1
                    : featureCount;
            if (centers.length != expected || scales.length != expected) {
                throw new IllegalArgumentException(
                        "Reusable-statistic group count does not match output "
                                + "feature count."
                );
            }
            return new PreparedReusable(scope, centers, scales);
        }
    }
}
