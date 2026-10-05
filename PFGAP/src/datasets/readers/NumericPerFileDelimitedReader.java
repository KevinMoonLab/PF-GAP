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
 * Eager coordinator for high-throughput numeric per-file delimited
 * multivariate time-series datasets.
 *
 * <p>One file is one observation, one selected column is one dimension, and
 * one record is one time position. Per-file observations are always
 * two-dimensional, including the single-dimension case.</p>
 *
 * <p>The eager reader returns raw primitive matrices and leaves
 * standardization to the eager dataset pipeline. It selects the FLOAT32 or
 * FLOAT64 path once before the file loop and invokes the paired series
 * reader's typed method directly.</p>
 */
public final class NumericPerFileDelimitedReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final String dataPath;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final String filePattern;
    private final boolean isTest;
    private final boolean isRegression;
    private final int initialTimeCapacity;
    private final NumericStorageType storageType;

    public NumericPerFileDelimitedReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getEntrySeparator(), options.hasHeader(),
                options.getTimeColumn(), options.getFeatureColumns(),
                options.getFilePattern(), options.isTest(),
                options.isRegression(), options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY,
                options.getNumericStorageType());
        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires isNumeric=true.");
        }
        if (options.hasMissingValues()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader does not support missing "
                            + "values. Use PerFileDelimitedReader instead.");
        }
    }

    public NumericPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats
    ) {
        this(dataPath, entrySeparator, hasHeader, timeColumn, featureColumns,
                filePattern, isTest, isRegression, standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY, NumericStorageType.AUTO);
    }

    public NumericPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(dataPath, entrySeparator, hasHeader, timeColumn, featureColumns,
                filePattern, isTest, isRegression, standardizationStats,
                initialTimeCapacity, NumericStorageType.AUTO);
    }

    public NumericPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericStorageType numericStorageType
    ) {
        this.dataPath = normalizeNullableString(dataPath);
        this.entrySeparator = normalizeSeparator(entrySeparator);
        this.hasHeader = hasHeader;
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = featureColumns == null
                ? List.of() : List.copyOf(featureColumns);
        this.filePattern = normalizeNullableString(filePattern);
        this.isTest = isTest;
        this.isRegression = isRegression;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1. Received: "
                            + initialTimeCapacity + ".");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        NumericStorageType requested = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        this.storageType = requested == NumericStorageType.AUTO
                ? NumericStorageType.FLOAT64 : requested;

        if (standardizationStats != null && !this.featureColumns.isEmpty()) {
            standardizationStats.validateFeatureCompatibility(
                    this.featureColumns);
        }
        validateConstructionOptions();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateReadOptions();
        List<Path> files = discoverFiles();

        NumericPerFileDelimitedSeriesReader seriesReader =
                new NumericPerFileDelimitedSeriesReader(
                        entrySeparator, hasHeader, timeColumn, featureColumns,
                        null, initialTimeCapacity, storageType);

        return storageType == NumericStorageType.FLOAT32
                ? readFloatDataset(files, seriesReader)
                : readDoubleDataset(files, seriesReader);
    }

    private ListObjectDataset readFloatDataset(
            List<Path> files,
            NumericPerFileDelimitedSeriesReader seriesReader
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
            int length = validateSeries(series, file, instance);
            dataset.add(inferLabel(file, instance), series, instance);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
            DelimitedFileReader.ProgressLogger.logProgress(instance);
        }
        return finalizeDataset(dataset, commonLength, unequalLengths);
    }

    private ListObjectDataset readDoubleDataset(
            List<Path> files,
            NumericPerFileDelimitedSeriesReader seriesReader
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
            int length = validateSeries(series, file, instance);
            dataset.add(inferLabel(file, instance), series, instance);
            if (commonLength < 0) {
                commonLength = length;
            } else if (length != commonLength) {
                unequalLengths = true;
            }
            DelimitedFileReader.ProgressLogger.logProgress(instance);
        }
        return finalizeDataset(dataset, commonLength, unequalLengths);
    }

    private static ListObjectDataset finalizeDataset(
            ListObjectDataset dataset,
            int commonLength,
            boolean unequalLengths
    ) {
        int length = unequalLengths ? 0 : Math.max(commonLength, 0);
        dataset.setLength(length);
        AppContext.length = length;
        return dataset;
    }

    private static int validateSeries(
            float[][] series,
            Path file,
            int instance
    ) {
        if (series == null || series.length == 0) {
            throw invalidSeries(file, instance, "no dimensions");
        }
        if (series[0] == null || series[0].length == 0) {
            throw invalidSeries(file, instance, "an empty first dimension");
        }
        int length = series[0].length;
        for (int dimension = 1;
             dimension < series.length;
             dimension++) {
            if (series[dimension] == null) {
                throw invalidSeries(file, instance,
                        "a null dimension at index " + dimension);
            }
            if (series[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "dimension " + dimension + " has length "
                                + series[dimension].length
                                + " instead of " + length);
            }
        }
        return length;
    }

    private static int validateSeries(
            double[][] series,
            Path file,
            int instance
    ) {
        if (series == null || series.length == 0) {
            throw invalidSeries(file, instance, "no dimensions");
        }
        if (series[0] == null || series[0].length == 0) {
            throw invalidSeries(file, instance, "an empty first dimension");
        }
        int length = series[0].length;
        for (int dimension = 1;
             dimension < series.length;
             dimension++) {
            if (series[dimension] == null) {
                throw invalidSeries(file, instance,
                        "a null dimension at index " + dimension);
            }
            if (series[dimension].length != length) {
                throw invalidSeries(file, instance,
                        "dimension " + dimension + " has length "
                                + series[dimension].length
                                + " instead of " + length);
            }
        }
        return length;
    }

    private static IllegalStateException invalidSeries(
            Path file,
            int instance,
            String detail
    ) {
        return new IllegalStateException(
                "Numeric series reader returned " + detail
                        + " for instance " + instance
                        + " from file " + file + ".");
    }

    private static IOException materializationIOException(
            int instance,
            Path file,
            IOException cause
    ) {
        return new IOException(
                "Failed to eagerly materialize numeric per-file delimited "
                        + "instance " + instance + " from file " + file + ".",
                cause);
    }

    private static IllegalArgumentException materializationRuntimeException(
            int instance,
            Path file,
            RuntimeException cause
    ) {
        return new IllegalArgumentException(
                "Failed to eagerly materialize numeric per-file delimited "
                        + "instance " + instance + " from file " + file + ".",
                cause);
    }

    private void validateConstructionOptions() {
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "A non-empty entry separator is required.");
        }
        if (entrySeparator.length() != 1) {
            throw new IllegalArgumentException(
                    "A single-character entry separator is required: '"
                            + entrySeparator + "'.");
        }
        char separator = entrySeparator.charAt(0);
        if (separator == '\n' || separator == '\r') {
            throw new IllegalArgumentException(
                    "A line separator cannot be the entry separator.");
        }
        for (String feature : featureColumns) {
            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature names or indices cannot be blank.");
            }
        }
    }

    private void validateReadOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires dataPath.");
        }
    }

    private List<Path> discoverFiles() throws IOException {
        Path path = Paths.get(dataPath);
        if (!Files.exists(path)) {
            throw new IOException(
                    "Numeric per-file delimited data path does not exist: "
                            + dataPath);
        }
        if (Files.isRegularFile(path)) {
            return List.of(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Numeric per-file delimited data path must be a directory "
                            + "or regular file: " + dataPath);
        }
        if (filePattern == null || filePattern.isBlank()) {
            throw new IllegalArgumentException(
                    "filePattern is required for a directory and must contain "
                            + "one numeric placeholder.");
        }
        return discoverFromPattern(path, filePattern);
    }

    private List<Path> discoverFromPattern(
            Path directory,
            String patternText
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
                    "No numeric delimited files in directory " + directory
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

    /** Placeholder for future per-file label conventions. */
    private Object inferLabel(Path file, int instanceIndex) {
        return null;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires ReaderOptions.");
        }
        return options;
    }

    private static String normalizeSeparator(String value) {
        if (value == null) {
            return null;
        }
        return switch (value) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> value;
        };
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
            Path path,
            String fileName,
            long sequenceNumber
    ) {
    }

    /**
     * Filename matcher with one arbitrary named numeric placeholder and glob
     * support outside the placeholder.
     */
    private static final class NumericPattern {
        private static final Pattern PLACEHOLDER_PATTERN = Pattern.compile(
                "\\{([A-Za-z_][A-Za-z0-9_]*)(?::0?(\\d+)d)?}");

        private final Pattern regex;
        private final String numericFieldName;

        private NumericPattern(Pattern regex, String numericFieldName) {
            this.regex = regex;
            this.numericFieldName = numericFieldName;
        }

        private static NumericPattern from(String filePattern) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "Per-file pattern cannot be blank.");
            }
            Matcher matcher = PLACEHOLDER_PATTERN.matcher(filePattern);
            if (!matcher.find()) {
                throw new IllegalArgumentException(
                        "Per-file pattern must contain exactly one numeric "
                                + "placeholder such as {num}, {num:04d}, "
                                + "{run}, or {run:03d}: " + filePattern);
            }
            String fieldName = matcher.group(1);
            String widthText = matcher.group(2);
            int start = matcher.start();
            int end = matcher.end();
            if (matcher.find()) {
                throw new IllegalArgumentException(
                        "Per-file pattern supports exactly one numeric "
                                + "placeholder: " + filePattern);
            }
            String numericRegex = widthText == null
                    ? "(\\d+)"
                    : "(\\d{" + Integer.parseInt(widthText) + "})";
            String regexText = "^"
                    + globFragmentToRegex(filePattern.substring(0, start))
                    + numericRegex
                    + globFragmentToRegex(filePattern.substring(end))
                    + "$";
            return new NumericPattern(
                    Pattern.compile(regexText), fieldName);
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

        private static String globFragmentToRegex(String fragment) {
            StringBuilder regex = new StringBuilder();
            StringBuilder literal = new StringBuilder();
            for (int index = 0; index < fragment.length(); index++) {
                char current = fragment.charAt(index);
                if (current == '*') {
                    appendQuotedLiteral(regex, literal);
                    regex.append(".*");
                } else if (current == '?') {
                    appendQuotedLiteral(regex, literal);
                    regex.append('.');
                } else {
                    literal.append(current);
                }
            }
            appendQuotedLiteral(regex, literal);
            return regex.toString();
        }

        private static void appendQuotedLiteral(
                StringBuilder regex,
                StringBuilder literal
        ) {
            if (literal.length() == 0) {
                return;
            }
            regex.append(Pattern.quote(literal.toString()));
            literal.setLength(0);
        }
    }
}