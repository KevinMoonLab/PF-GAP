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
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Eager coordinator for general per-file Parquet multivariate time-series
 * datasets.
 *
 * <p>Each Parquet file is one observation, each selected feature column is one
 * dimension, and each record is one time position. Materialized observations
 * must be {@code float[][]}, {@code double[][]}, or {@code Object[][]}. The
 * one-feature case remains two-dimensional.</p>
 *
 * <p>The eager coordinator calls {@link PerFileParquetSeriesReader#readFile}
 * directly and therefore creates no temporary lazy references. It passes null
 * standardization statistics because the completed eager dataset pipeline owns
 * transformation.</p>
 */
public final class PerFileParquetReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final String dataPath;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final String filePattern;
    private final int initialTimeCapacity;
    private final NumericStorageType numericStorageType;

    public PerFileParquetReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(), options.isNumeric(),
                options.hasMissingValues(), options.isRegression(),
                options.getTimeColumn(), options.getFeatureColumns(),
                options.getLabelColumns(), options.getFilePattern(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY,
                options.getNumericStorageType());
    }

    public PerFileParquetReader(
            String dataPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern,
            StandardizationStats standardizationStats
    ) {
        this(dataPath, isNumeric, hasMissingValues, isRegression, timeColumn,
                featureColumns, labelColumns, filePattern,
                standardizationStats, DEFAULT_INITIAL_TIME_CAPACITY,
                NumericStorageType.AUTO);
    }

    public PerFileParquetReader(
            String dataPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericStorageType numericStorageType
    ) {
        this.dataPath = normalizeNullableString(dataPath);
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyList(featureColumns);
        this.labelColumns = copyList(labelColumns);
        this.filePattern = normalizeNullableString(filePattern);
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        NumericStorageType requested = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        // Eager datasets need one stable primitive representation across files.
        this.numericStorageType = requested == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64 : requested;

        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Standardization statistics require numeric data.");
            }
            if (!this.featureColumns.isEmpty()) {
                standardizationStats.validateFeatureCompatibility(
                        this.featureColumns);
            }
        }
        validateOptions();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        List<Path> files = discoverFiles();
        PerFileParquetSeriesReader seriesReader =
                new PerFileParquetSeriesReader(
                        timeColumn, featureColumns, isNumeric,
                        hasMissingValues, null, initialTimeCapacity,
                        numericStorageType);

        ListObjectDataset dataset = new ListObjectDataset(files.size());
        int commonLength = -1;
        boolean unequalLengths = false;

        for (int instance = 0; instance < files.size(); instance++) {
            Path file = files.get(instance);
            final Object series;
            try {
                series = seriesReader.readFile(file);
            } catch (IOException e) {
                throw new IOException(
                        "Failed to eagerly materialize Parquet per-file "
                                + "instance " + instance + " from " + file + ".",
                        e);
            } catch (RuntimeException e) {
                throw new IllegalArgumentException(
                        "Failed to eagerly materialize Parquet per-file "
                                + "instance " + instance + " from " + file + ".",
                        e);
            }

            int length = validateSeries(series, file, instance);
            dataset.add(inferLabel(file, instance), series, instance);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
            DelimitedFileReader.ProgressLogger.logProgress(instance);
        }

        int datasetLength = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(datasetLength);
        AppContext.length = datasetLength;
        return dataset;
    }

    private static int validateSeries(
            Object series, Path file, int instance
    ) {
        if (series instanceof float[][] values) {
            return validate(values, file, instance);
        }
        if (series instanceof double[][] values) {
            return validate(values, file, instance);
        }
        if (series instanceof Object[][] values) {
            return validate(values, file, instance);
        }
        throw invalidSeries(file, instance,
                "unsupported representation "
                        + (series == null ? "null"
                        : series.getClass().getName()));
    }

    private static int validate(float[][] values, Path file, int instance) {
        if (values.length == 0) {
            throw invalidSeries(file, instance, "no dimensions");
        }
        if (values[0] == null || values[0].length == 0) {
            throw invalidSeries(file, instance, "no time positions");
        }
        int length = values[0].length;
        for (int dimension = 1;
             dimension < values.length;
             dimension++) {
            if (values[dimension] == null
                    || values[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "inconsistent length at dimension " + dimension);
            }
        }
        return length;
    }

    private static int validate(double[][] values, Path file, int instance) {
        if (values.length == 0) {
            throw invalidSeries(file, instance, "no dimensions");
        }
        if (values[0] == null || values[0].length == 0) {
            throw invalidSeries(file, instance, "no time positions");
        }
        int length = values[0].length;
        for (int dimension = 1;
             dimension < values.length;
             dimension++) {
            if (values[dimension] == null
                    || values[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "inconsistent length at dimension " + dimension);
            }
        }
        return length;
    }

    private static int validate(Object[][] values, Path file, int instance) {
        if (values.length == 0) {
            throw invalidSeries(file, instance, "no dimensions");
        }
        if (values[0] == null || values[0].length == 0) {
            throw invalidSeries(file, instance, "no time positions");
        }
        int length = values[0].length;
        for (int dimension = 1;
             dimension < values.length;
             dimension++) {
            if (values[dimension] == null
                    || values[dimension].length != length) {
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
                "Invalid Parquet per-file observation " + instance
                        + " from " + file + ": " + detail + ".");
    }

    private void validateOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "PerFileParquetReader requires dataPath.");
        }
        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "At least one feature column is required.");
        }
        for (String feature : featureColumns) {
            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature columns cannot contain blank values.");
            }
        }
        if (timeColumn != null && featureColumns.contains(timeColumn)) {
            throw new IllegalArgumentException(
                    "The time column cannot also be a feature column: "
                            + timeColumn);
        }
        if (!labelColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileParquetReader does not interpret columns as "
                            + "per-observation labels. Use external metadata "
                            + "or another reader for labels.");
        }
    }

    private List<Path> discoverFiles() throws IOException {
        Path path = Paths.get(dataPath);
        if (!Files.exists(path)) {
            throw new IOException(
                    "Per-file Parquet data path does not exist: " + dataPath);
        }
        if (Files.isRegularFile(path)) {
            if (!isParquetFile(path)) {
                throw new IOException(
                        "Expected a .parquet file: " + dataPath);
            }
            return List.of(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Per-file Parquet data path must be a directory or a "
                            + "regular .parquet file: " + dataPath);
        }
        if (filePattern == null || filePattern.isBlank()) {
            throw new IllegalArgumentException(
                    "filePattern is required for directory input and must "
                            + "contain exactly one numeric placeholder.");
        }
        return discoverFromPattern(path, filePattern);
    }

    private static boolean isParquetFile(Path path) {
        return path.getFileName().toString()
                .toLowerCase(java.util.Locale.ROOT)
                .endsWith(".parquet");
    }

    private List<Path> discoverFromPattern(
            Path directory, String patternText
    ) throws IOException {
        NumericPattern pattern = NumericPattern.from(patternText);
        List<IndexedPath> indexed = new ArrayList<>();
        try (Stream<Path> stream = Files.list(directory)) {
            stream.filter(Files::isRegularFile)
                    .filter(PerFileParquetReader::isParquetFile)
                    .forEach(path -> {
                        String fileName = path.getFileName().toString();
                        Long sequence = pattern.tryExtractNumber(fileName);
                        if (sequence != null) {
                            indexed.add(new IndexedPath(
                                    path, fileName, sequence));
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
        List<Path> result = new ArrayList<>(indexed.size());
        for (IndexedPath value : indexed) {
            result.add(value.path());
        }
        return List.copyOf(result);
    }

    /** Placeholder for future filename or external metadata labels. */
    private Object inferLabel(Path file, int instanceIndex) {
        return null;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "PerFileParquetReader requires ReaderOptions.");
        }
        return options;
    }

    private static List<String> copyList(List<String> values) {
        return values == null || values.isEmpty()
                ? List.of() : List.copyOf(values);
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
