package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Eager coordinator for optimized numeric per-file Parquet multivariate
 * time-series datasets.
 *
 * <p>One Parquet file is one observation, each projected feature column is one
 * dimension, and each record is one time position. Observations are always
 * represented as {@code float[dimension][time]} or
 * {@code double[dimension][time]}; the one-feature case is never collapsed.</p>
 *
 * <p>The coordinator chooses an explicit FLOAT32 or FLOAT64 loop once before
 * materializing files. AUTO is resolved deterministically to FLOAT64 here so
 * one eager dataset cannot contain mixed primitive matrix types when separate
 * files have different physical schemas. The series reader still supports
 * physical DOUBLE, FLOAT, INT64, and INT32 columns.</p>
 *
 * <p>Eager materialization is raw. Standardization statistics are validated
 * here but are not passed into the series reader, because the completed eager
 * dataset pipeline owns transformation.</p>
 */
public final class NumericPerFileParquetReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final String dataPath;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean hasMissingValues;
    private final String filePattern;
    private final boolean isTest;
    private final boolean isRegression;
    private final int initialTimeCapacity;
    private final NumericPerFileParquetSeriesReader.TimeOrderPolicy
            timeOrderPolicy;
    private final NumericStorageType storageType;

    public NumericPerFileParquetReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getTimeColumn(), options.getFeatureColumns(),
                options.hasMissingValues(), options.getFilePattern(),
                options.isTest(), options.isRegression(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER,
                options.getNumericStorageType());
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires isNumeric=true.");
        }
        if (options.getLabelColumns() != null
                && !options.getLabelColumns().isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader does not interpret projected "
                            + "columns as per-observation labels.");
        }
    }

    public NumericPerFileParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats
    ) {
        this(dataPath, timeColumn, featureColumns, hasMissingValues,
                filePattern, isTest, isRegression, standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER,
                NumericStorageType.AUTO);
    }

    public NumericPerFileParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(dataPath, timeColumn, featureColumns, hasMissingValues,
                filePattern, isTest, isRegression, standardizationStats,
                initialTimeCapacity,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER,
                NumericStorageType.AUTO);
    }

    public NumericPerFileParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericPerFileParquetSeriesReader.TimeOrderPolicy timeOrderPolicy
    ) {
        this(dataPath, timeColumn, featureColumns, hasMissingValues,
                filePattern, isTest, isRegression, standardizationStats,
                initialTimeCapacity, timeOrderPolicy,
                NumericStorageType.AUTO);
    }

    public NumericPerFileParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericPerFileParquetSeriesReader.TimeOrderPolicy timeOrderPolicy,
            NumericStorageType numericStorageType
    ) {
        this.dataPath = normalizeNullableString(dataPath);
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyAndValidateFeatureColumns(featureColumns);
        this.hasMissingValues = hasMissingValues;
        this.filePattern = normalizeNullableString(filePattern);
        this.isTest = isTest;
        this.isRegression = isRegression;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        this.timeOrderPolicy = timeOrderPolicy == null
                ? NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER
                : timeOrderPolicy;
        NumericStorageType requested = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        this.storageType = requested == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64 : requested;

        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(
                    this.featureColumns);
        }
        validateConstructionOptions();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        List<Path> files = discoverFiles();
        NumericPerFileParquetSeriesReader seriesReader =
                new NumericPerFileParquetSeriesReader(
                        timeColumn, featureColumns, hasMissingValues,
                        null, initialTimeCapacity, timeOrderPolicy,
                        storageType);

        return storageType == NumericStorageType.FLOAT32
                ? readFloatDataset(files, seriesReader)
                : readDoubleDataset(files, seriesReader);
    }

    private ListObjectDataset readFloatDataset(
            List<Path> files,
            NumericPerFileParquetSeriesReader seriesReader
    ) throws IOException {
        ListObjectDataset dataset = new ListObjectDataset(files.size());
        int commonLength = -1;
        boolean unequalLengths = false;

        for (int instance = 0; instance < files.size(); instance++) {
            Path file = files.get(instance);
            final float[][] series;
            try {
                series = seriesReader.readFloatFile(file);
            } catch (IOException e) {
                throw materializationIOException(instance, file, e);
            } catch (RuntimeException e) {
                throw materializationRuntimeException(instance, file, e);
            }
            int length = validate(series, file, instance);
            dataset.add(inferLabel(file, instance), series, instance);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
            DelimitedFileReader.ProgressLogger.logProgress(instance);
        }
        return finish(dataset, commonLength, unequalLengths);
    }

    private ListObjectDataset readDoubleDataset(
            List<Path> files,
            NumericPerFileParquetSeriesReader seriesReader
    ) throws IOException {
        ListObjectDataset dataset = new ListObjectDataset(files.size());
        int commonLength = -1;
        boolean unequalLengths = false;

        for (int instance = 0; instance < files.size(); instance++) {
            Path file = files.get(instance);
            final double[][] series;
            try {
                series = seriesReader.readDoubleFile(file);
            } catch (IOException e) {
                throw materializationIOException(instance, file, e);
            } catch (RuntimeException e) {
                throw materializationRuntimeException(instance, file, e);
            }
            int length = validate(series, file, instance);
            dataset.add(inferLabel(file, instance), series, instance);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
            DelimitedFileReader.ProgressLogger.logProgress(instance);
        }
        return finish(dataset, commonLength, unequalLengths);
    }

    private static ListObjectDataset finish(
            ListObjectDataset dataset,
            int commonLength,
            boolean unequalLengths
    ) {
        int length = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(length);
        AppContext.length = length;
        return dataset;
    }

    private static int validate(
            float[][] series, Path file, int instance
    ) {
        if (series == null || series.length == 0) {
            throw invalidSeries(file, instance, "no feature dimensions");
        }
        if (series[0] == null || series[0].length == 0) {
            throw invalidSeries(file, instance, "no time positions");
        }
        int length = series[0].length;
        for (int dimension = 1;
             dimension < series.length;
             dimension++) {
            if (series[dimension] == null
                    || series[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "inconsistent length at dimension " + dimension);
            }
        }
        return length;
    }

    private static int validate(
            double[][] series, Path file, int instance
    ) {
        if (series == null || series.length == 0) {
            throw invalidSeries(file, instance, "no feature dimensions");
        }
        if (series[0] == null || series[0].length == 0) {
            throw invalidSeries(file, instance, "no time positions");
        }
        int length = series[0].length;
        for (int dimension = 1;
             dimension < series.length;
             dimension++) {
            if (series[dimension] == null
                    || series[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "inconsistent length at dimension " + dimension);
            }
        }
        return length;
    }

    private static IllegalStateException invalidSeries(
            Path file, int instance, String detail
    ) {
        return new IllegalStateException(
                "Invalid numeric Parquet per-file observation " + instance
                        + " from " + file + ": " + detail + ".");
    }

    private static IOException materializationIOException(
            int instance, Path file, IOException cause
    ) {
        return new IOException(
                "Failed to eagerly materialize numeric Parquet per-file "
                        + "instance " + instance + " from " + file + ".",
                cause);
    }

    private static IllegalArgumentException materializationRuntimeException(
            int instance, Path file, RuntimeException cause
    ) {
        return new IllegalArgumentException(
                "Failed to eagerly materialize numeric Parquet per-file "
                        + "instance " + instance + " from " + file + ".",
                cause);
    }

    private void validateConstructionOptions() {
        if (timeOrderPolicy
                == NumericPerFileParquetSeriesReader.TimeOrderPolicy
                .SORT_DOUBLE_TIME
                && timeColumn == null) {
            throw new IllegalArgumentException(
                    "SORT_DOUBLE_TIME requires a time column.");
        }
        if (timeColumn != null && featureColumns.contains(timeColumn)) {
            throw new IllegalArgumentException(
                    "The time column cannot also be a feature column: "
                            + timeColumn);
        }
    }

    private List<Path> discoverFiles() throws IOException {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires dataPath.");
        }
        Path path = Paths.get(dataPath);
        if (!Files.exists(path)) {
            throw new IOException(
                    "Numeric per-file Parquet path does not exist: "
                            + dataPath);
        }
        if (Files.isRegularFile(path)) {
            return List.of(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Numeric per-file Parquet path must be a directory or "
                            + "regular file: " + dataPath);
        }
        if (filePattern == null || filePattern.isBlank()) {
            throw new IllegalArgumentException(
                    "filePattern is required for directory input and must "
                            + "contain exactly one numeric placeholder.");
        }
        return discoverFromPattern(path, filePattern);
    }

    private List<Path> discoverFromPattern(
            Path directory, String patternText
    ) throws IOException {
        NumericPattern pattern = NumericPattern.from(patternText);
        List<IndexedPath> indexed = new ArrayList<>();
        try (Stream<Path> stream = Files.list(directory)) {
            stream.filter(Files::isRegularFile).forEach(path -> {
                String fileName = path.getFileName().toString();
                Long sequence = pattern.tryExtractNumber(fileName);
                if (sequence != null) {
                    indexed.add(new IndexedPath(path, fileName, sequence));
                }
            });
        }
        if (indexed.isEmpty()) {
            throw new IOException(
                    "No Parquet files in " + directory
                            + " matched pattern: " + patternText);
        }
        indexed.sort(Comparator
                .comparingLong(IndexedPath::sequenceNumber)
                .thenComparing(IndexedPath::fileName));
        List<Path> files = new ArrayList<>(indexed.size());
        for (IndexedPath value : indexed) {
            files.add(value.path());
        }
        return List.copyOf(files);
    }

    /** Placeholder for future filename or external metadata labels. */
    private Object inferLabel(Path file, int instanceIndex) {
        return null;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires ReaderOptions.");
        }
        return options;
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> columns
    ) {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "At least one feature column is required.");
        }
        List<String> result = new ArrayList<>(columns.size());
        Set<String> seen = new HashSet<>();
        for (String column : columns) {
            if (column == null || column.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature columns cannot be blank.");
            }
            String normalized = column.trim();
            if (!seen.add(normalized)) {
                throw new IllegalArgumentException(
                        "Duplicate feature column: " + normalized);
            }
            result.add(normalized);
        }
        return List.copyOf(result);
    }

    private static String normalizeNullableString(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() || trimmed.equalsIgnoreCase("None")
                ? null : trimmed;
    }

    private record IndexedPath(
            Path path, String fileName, long sequenceNumber
    ) {
    }

    private static final class NumericPattern {
        private static final Pattern PLACEHOLDER = Pattern.compile(
                "\\{([A-Za-z_][A-Za-z0-9_]*)(?::0?(\\d+)d)?}");

        private final Pattern regex;
        private final String numericFieldName;

        private NumericPattern(Pattern regex, String numericFieldName) {
            this.regex = regex;
            this.numericFieldName = numericFieldName;
        }

        private static NumericPattern from(String pattern) {
            Matcher matcher = PLACEHOLDER.matcher(pattern);
            if (!matcher.find()) {
                throw new IllegalArgumentException(
                        "Pattern must contain one numeric placeholder such as "
                                + "{num}, {num:04d}, {run}, or {run:03d}: "
                                + pattern);
            }
            String fieldName = matcher.group(1);
            String widthText = matcher.group(2);
            int start = matcher.start();
            int end = matcher.end();
            if (matcher.find()) {
                throw new IllegalArgumentException(
                        "Pattern supports exactly one numeric placeholder: "
                                + pattern);
            }
            String numeric = widthText == null
                    ? "(\\d+)"
                    : "(\\d{" + Integer.parseInt(widthText) + "})";
            String regex = "^"
                    + globFragment(pattern.substring(0, start))
                    + numeric
                    + globFragment(pattern.substring(end)) + "$";
            return new NumericPattern(Pattern.compile(regex), fieldName);
        }

        private Long tryExtractNumber(String fileName) {
            Matcher matcher = regex.matcher(fileName);
            if (!matcher.matches()) {
                return null;
            }
            try {
                return Long.parseLong(matcher.group(1));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Numeric field '" + numericFieldName
                                + "' exceeds long range in filename: "
                                + fileName, e);
            }
        }

        private static String globFragment(String fragment) {
            StringBuilder regex = new StringBuilder();
            StringBuilder literal = new StringBuilder();
            for (int index = 0; index < fragment.length(); index++) {
                char current = fragment.charAt(index);
                if (current == '*' || current == '?') {
                    appendLiteral(regex, literal);
                    regex.append(current == '*' ? ".*" : ".");
                } else {
                    literal.append(current);
                }
            }
            appendLiteral(regex, literal);
            return regex.toString();
        }

        private static void appendLiteral(
                StringBuilder regex, StringBuilder literal
        ) {
            if (literal.length() > 0) {
                regex.append(Pattern.quote(literal.toString()));
                literal.setLength(0);
            }
        }
    }
}
