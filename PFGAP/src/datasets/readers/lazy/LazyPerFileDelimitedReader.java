package datasets.readers.lazy;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.NumericStorageType;
import datasets.readers.DatasetReader;
import datasets.readers.NumericPerFileParquetSeriesReader;
import datasets.readers.ReaderOptions;
import datasets.readers.ReaderType;
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
 * Lazy coordinator for general per-file delimited time-series datasets.
 *
 * <p>One file is one observation, one record is one time position, and one
 * selected column is one dimension. Dataset construction discovers files and
 * stores a {@link LazySeriesRef}; file contents are materialized later by the
 * registered {@code PerFileDelimitedSeriesReader}.</p>
 *
 * <p>Materialized observations are always two-dimensional:</p>
 * <ul>
 *     <li>{@code float[dimension][time]} for FLOAT32 numeric data</li>
 *     <li>{@code double[dimension][time]} for FLOAT64 numeric data</li>
 *     <li>{@code Object[dimension][time]} for generic data</li>
 * </ul>
 *
 * <p>Numeric missing values use primitive NaN. Generic missing values use
 * null. A single selected feature remains {@code [1][time]}.</p>
 */
public final class LazyPerFileDelimitedReader implements DatasetReader {
    private static final int DEFAULT_INITIAL_TIME_CAPACITY = 256;

    private final String dataPath;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final String filePattern;
    private final String readerKey;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final NumericStorageType numericStorageType;

    public LazyPerFileDelimitedReader(ReaderOptions options) {
        this(requireOptions(options).getDataPath(),
                options.getEntrySeparator(), options.hasHeader(),
                options.isNumeric(), options.hasMissingValues(),
                options.isRegression(), options.getTimeColumn(),
                options.getFeatureColumns(), options.getLabelColumns(),
                options.getFilePattern(),
                options.isTest() ? "test" : "train",
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY,
                options.getNumericStorageType());
    }

    public LazyPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats
    ) {
        this(dataPath, entrySeparator, hasHeader, isNumeric,
                hasMissingValues, isRegression, timeColumn, featureColumns,
                labelColumns, filePattern, readerKey, standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY, NumericStorageType.AUTO);
    }

    public LazyPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this(dataPath, entrySeparator, hasHeader, isNumeric,
                hasMissingValues, isRegression, timeColumn, featureColumns,
                labelColumns, filePattern, readerKey, standardizationStats,
                initialTimeCapacity, NumericStorageType.AUTO);
    }

    public LazyPerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats,
            int initialTimeCapacity,
            NumericStorageType numericStorageType
    ) {
        this.dataPath = normalizeNullableString(dataPath);
        this.entrySeparator = normalizeSeparator(entrySeparator);
        this.hasHeader = hasHeader;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.timeColumn = normalizeNullableString(timeColumn);
        this.featureColumns = copyList(featureColumns);
        this.labelColumns = copyList(labelColumns);
        this.filePattern = normalizeNullableString(filePattern);
        this.readerKey = requireNonblank(readerKey, "readerKey");
        this.standardizationStats = standardizationStats;
        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "initialTimeCapacity must be at least 1.");
        }
        this.initialTimeCapacity = initialTimeCapacity;
        this.numericStorageType = Objects.requireNonNull(
                numericStorageType, "numericStorageType cannot be null.");
        validateOptions();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        List<Path> files = discoverFiles();

        LazySeriesReaderSpec spec = new LazySeriesReaderSpec(
                readerKey,
                ReaderType.LAZY_PER_FILE_DELIMITED,
                timeColumn,
                featureColumns,
                isNumeric,
                hasMissingValues,
                entrySeparator,
                hasHeader,
                standardizationStats,
                initialTimeCapacity,
                numericStorageType,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER
        );
        AppContext.registerLazySeriesReader(spec);

        ListObjectDataset dataset = new ListObjectDataset(files.size());
        for (int instance = 0; instance < files.size(); instance++) {
            Path file = files.get(instance);
            dataset.add(
                    inferLabel(file, instance),
                    new LazySeriesRef(readerKey, instance, file),
                    instance
            );
        }

        dataset.setLength(0);
        AppContext.length = 0;
        return dataset;
    }

    private void validateOptions() {
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "A non-empty entry separator is required.");
        }
        if (entrySeparator.length() != 1
                || entrySeparator.charAt(0) == '\n'
                || entrySeparator.charAt(0) == '\r') {
            throw new IllegalArgumentException(
                    "A non-line, single-character entry separator is required.");
        }
        if (!labelColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "LazyPerFileDelimitedReader does not interpret per-file "
                            + "columns as observation labels.");
        }
        Set<String> seen = new HashSet<>();
        for (String feature : featureColumns) {
            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature columns cannot contain blank values.");
            }
            if (!seen.add(feature)) {
                throw new IllegalArgumentException(
                        "Duplicate feature column: " + feature);
            }
        }
        if (standardizationStats != null) {
            if (!isNumeric) {
                throw new IllegalArgumentException(
                        "Standardization statistics require numeric data.");
            }
            if (!featureColumns.isEmpty()) {
                standardizationStats.validateFeatureCompatibility(
                        featureColumns);
            }
        }
    }

    private List<Path> discoverFiles() throws IOException {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyPerFileDelimitedReader requires dataPath.");
        }
        Path path = Paths.get(dataPath);
        if (!Files.exists(path) || !Files.isReadable(path)) {
            throw new IOException("Data path is not readable: " + dataPath);
        }
        if (Files.isRegularFile(path)) {
            return List.of(path);
        }
        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Data path must be a regular file or directory: "
                            + dataPath);
        }
        if (filePattern == null || filePattern.isBlank()) {
            throw new IllegalArgumentException(
                    "filePattern is required for directory input and must "
                            + "contain exactly one numeric placeholder.");
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
                    "No delimited files in " + directory
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

    private Object inferLabel(Path file, int instanceIndex) {
        return null;
    }

    private static ReaderOptions requireOptions(ReaderOptions options) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "LazyPerFileDelimitedReader requires ReaderOptions.");
        }
        return options;
    }

    private static List<String> copyList(List<String> values) {
        if (values == null || values.isEmpty()) {
            return List.of();
        }
        List<String> result = new ArrayList<>(values.size());
        for (String value : values) {
            result.add(value == null ? null : value.trim());
        }
        return List.copyOf(result);
    }

    private static String requireNonblank(String value, String role) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(role + " cannot be blank.");
        }
        return value.trim();
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
                StringBuilder regex,
                StringBuilder literal
        ) {
            if (literal.length() > 0) {
                regex.append(Pattern.quote(literal.toString()));
                literal.setLength(0);
            }
        }
    }
}
