package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import preprocessing.standardization.StandardizationStats;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Eager DatasetReader for numeric per-file Parquet time-series datasets.
 *
 * <p>Storage assumption:</p>
 *
 * <pre>
 * one Parquet file = one dataset instance / time series
 * one Parquet record = one time point
 * one projected feature column = one time-series dimension
 * </pre>
 *
 * <p>Every discovered file is materialized during {@link #read()} through
 * {@link NumericPerFileParquetSeriesReader}, which uses Hardwood's
 * batch-oriented column-reading API.</p>
 *
 * <p>Returned instance representations:</p>
 *
 * <pre>
 * one feature, no missing values:
 *     double[time]
 *
 * multiple features, no missing values:
 *     double[dimension][time]
 *
 * one feature, missing values:
 *     Double[time]
 *
 * multiple features, missing values:
 *     Double[dimension][time]
 * </pre>
 *
 * <p>This reader is specialized for flat numeric Parquet feature columns.
 * The current {@link NumericPerFileParquetSeriesReader} implementation
 * expects projected feature columns to have Parquet physical type DOUBLE.</p>
 *
 * <p>Use {@link PerFileParquetReader} for generic, nonnumeric, nested, or
 * otherwise unsupported Parquet schemas.</p>
 *
 * <p>The data path may identify:</p>
 *
 * <ol>
 *     <li>
 *         A directory containing one Parquet file per instance. In this
 *         case, {@code filePattern} is required and must contain exactly one
 *         numeric placeholder.
 *     </li>
 *     <li>
 *         A single regular Parquet file. In this case, {@code filePattern}
 *         is not required and the resulting dataset contains one instance.
 *     </li>
 * </ol>
 *
 * <p>Example directory configuration:</p>
 *
 * <pre>
 * dataPath:
 *     /path/to/training
 *
 * filePattern:
 *     series_{num:04d}.parquet
 * </pre>
 *
 * <p>Other valid patterns include:</p>
 *
 * <pre>
 * trial_*.{run:03d}_freqN.parquet
 * series_{instance}.parquet
 * sample_?_{num:05d}.pq
 * </pre>
 *
 * <p>Supported numeric placeholders include:</p>
 *
 * <pre>
 * {num}
 * {num:04d}
 * {run}
 * {run:03d}
 * {instance}
 * {instance:05d}
 * </pre>
 *
 * <p>The placeholder name is arbitrary but must be a valid identifier.
 * Glob wildcards {@code *} and {@code ?} may occur outside the numeric
 * placeholder.</p>
 *
 * <p>Labels are currently assigned as {@code null}, matching the existing
 * per-file reader behavior. External label files, filename-based labels,
 * and metadata-table labels can be added separately.</p>
 *
 * <p>This implementation reads files sequentially. Hardwood still performs
 * parallel page decoding internally. File-level parallelism or coordinated
 * multi-file reading can be added separately after the column-reader
 * implementation has been benchmarked.</p>
 */
public class NumericPerFileParquetReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            1024;

    private final String dataPath;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean hasMissingValues;
    private final String filePattern;
    private final boolean isTest;
    private final boolean isRegression;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;
    private final NumericPerFileParquetSeriesReader.TimeOrderPolicy
            timeOrderPolicy;

    /**
     * Constructs the eager reader from ordinary PFGAP reader options.
     *
     * <p>The specialized reader requires {@code isNumeric=true}. Missing
     * values are supported and are controlled by
     * {@code options.hasMissingValues()}.</p>
     *
     * <p>This constructor defaults to physical Parquet file order. The time
     * column is not projected merely because it is configured. Use the full
     * constructor with {@code SORT_DOUBLE_TIME} when physical records must be
     * sorted by a DOUBLE time column.</p>
     *
     * @param options PFGAP reader options
     */
    public NumericPerFileParquetReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.hasMissingValues(),
                options.getFilePattern(),
                options.isTest(),
                options.isRegression(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER
        );

        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires "
                            + "ReaderOptions.isNumeric=true."
            );
        }
    }

    /**
     * Constructs the eager reader using physical Parquet record order and the
     * default initial time capacity.
     */
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
        this(
                dataPath,
                timeColumn,
                featureColumns,
                hasMissingValues,
                filePattern,
                isTest,
                isRegression,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                NumericPerFileParquetSeriesReader.TimeOrderPolicy.FILE_ORDER
        );
    }

    /**
     * Constructs the eager reader with an explicit time-order policy.
     */
    public NumericPerFileParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            boolean isTest,
            boolean isRegression,
            StandardizationStats standardizationStats,
            NumericPerFileParquetSeriesReader.TimeOrderPolicy timeOrderPolicy
    ) {
        this(
                dataPath,
                timeColumn,
                featureColumns,
                hasMissingValues,
                filePattern,
                isTest,
                isRegression,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY,
                timeOrderPolicy
        );
    }

    /**
     * Full constructor.
     *
     * @param dataPath             directory or single Parquet file
     * @param timeColumn           optional time column
     * @param featureColumns       projected DOUBLE feature columns
     * @param hasMissingValues     whether null feature values are permitted
     * @param filePattern          required for directory input
     * @param isTest               whether this is a test reader
     * @param isRegression         regression-mode indicator
     * @param standardizationStats optional standardization statistics
     * @param initialTimeCapacity  allocation hint per feature dimension
     * @param timeOrderPolicy      file-order or DOUBLE-time sorting policy
     */
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
        this.dataPath =
                normalizeNullableString(
                        dataPath
                );

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyAndValidateFeatureColumns(
                        featureColumns
                );

        this.hasMissingValues =
                hasMissingValues;

        this.filePattern =
                normalizeNullableString(
                        filePattern
                );

        this.isTest =
                isTest;

        this.isRegression =
                isRegression;

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader initialTimeCapacity "
                            + "must be at least 1. Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        this.timeOrderPolicy =
                timeOrderPolicy == null
                        ? NumericPerFileParquetSeriesReader
                        .TimeOrderPolicy.FILE_ORDER
                        : timeOrderPolicy;

        validateConstructionOptions();
    }

    /**
     * Discovers and eagerly materializes every configured numeric
     * per-instance Parquet file.
     *
     * @return eager dataset containing numeric time-series instances
     * @throws IOException if discovery or materialization fails
     */
    @Override
    public ListObjectDataset read()
            throws IOException {

        validateReadOptions();

        List<Path> files =
                discoverFiles();

        NumericPerFileParquetSeriesReader seriesReader =
                new NumericPerFileParquetSeriesReader(
                        timeColumn,
                        featureColumns,
                        hasMissingValues,
                        standardizationStats,
                        initialTimeCapacity,
                        timeOrderPolicy
                );

        ListObjectDataset dataset =
                new ListObjectDataset(
                        files.size()
                );

        int commonTimeLength =
                -1;

        boolean unequalTimeLengths =
                false;

        for (int instanceIndex = 0;
             instanceIndex < files.size();
             instanceIndex++) {

            Path file =
                    files.get(
                            instanceIndex
                    );

            Object series;

            try {
                series =
                        seriesReader.readFile(
                                file
                        );
            } catch (IOException e) {
                throw new IOException(
                        "Failed to eagerly materialize numeric "
                                + "per-file Parquet instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ".",
                        e
                );
            } catch (RuntimeException e) {
                throw new IllegalArgumentException(
                        "Failed to eagerly materialize numeric "
                                + "per-file Parquet instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ".",
                        e
                );
            }

            int timeLength =
                    validateAndGetTimeLength(
                            series,
                            file,
                            instanceIndex
                    );

            Object label =
                    inferLabel(
                            file,
                            instanceIndex
                    );

            dataset.add(
                    label,
                    series,
                    instanceIndex
            );

            if (commonTimeLength < 0) {
                commonTimeLength =
                        timeLength;
            } else if (timeLength != commonTimeLength) {
                unequalTimeLengths =
                        true;
            }

            /*
             * Retain the existing eager-reader progress convention.
             * The logger reports file/instance progress rather than Parquet
             * row progress.
             */
            DelimitedFileReader.ProgressLogger.logProgress(
                    instanceIndex
            );
        }

        /*
         * A positive global length is meaningful only if every instance has
         * the same number of time points. Otherwise, zero communicates that
         * no common dataset-level length applies.
         */
        int datasetLength =
                unequalTimeLengths
                        ? 0
                        : Math.max(
                        commonTimeLength,
                        0
                );

        dataset.setLength(
                datasetLength
        );

        AppContext.length =
                datasetLength;

        return dataset;
    }

    private void validateConstructionOptions() {
        if (timeOrderPolicy
                == NumericPerFileParquetSeriesReader
                .TimeOrderPolicy.SORT_DOUBLE_TIME
                && timeColumn == null) {

            throw new IllegalArgumentException(
                    "SORT_DOUBLE_TIME requires a nonempty timeColumn."
            );
        }

        if (timeColumn != null
                && featureColumns.contains(timeColumn)) {

            throw new IllegalArgumentException(
                    "The configured time column cannot also be selected "
                            + "as a numeric feature column: "
                            + timeColumn
            );
        }

        if (standardizationStats != null) {
            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private void validateReadOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires dataPath."
            );
        }
    }

    private List<Path> discoverFiles()
            throws IOException {

        Path path =
                Paths.get(
                        dataPath
                );

        if (!Files.exists(path)) {
            throw new IOException(
                    "Numeric per-file Parquet data path does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "NumericPerFileParquetReader requires filePattern "
                                + "when dataPath is a directory. The pattern "
                                + "must contain exactly one numeric "
                                + "placeholder, such as "
                                + "series_{num:04d}.parquet or "
                                + "trial_*.{run:03d}_freqN.parquet."
                );
            }

            return discoverFromPattern(
                    path,
                    filePattern
            );
        }

        if (Files.isRegularFile(path)) {
            return List.of(
                    path
            );
        }

        throw new IOException(
                "Numeric per-file Parquet data path must be a directory "
                        + "or a regular file: "
                        + dataPath
        );
    }

    /**
     * Discovers matching files and resolves each numerical sorting key once.
     *
     * <p>This avoids repeating regex matching and numerical extraction from
     * inside the sorting comparator.</p>
     */
    private List<Path> discoverFromPattern(
            Path directory,
            String patternText
    ) throws IOException {

        NumericPattern numericPattern =
                NumericPattern.from(
                        patternText
                );

        List<IndexedPath> indexedPaths =
                new ArrayList<>();

        try (Stream<Path> stream =
                     Files.list(
                             directory
                     )) {

            stream.filter(Files::isRegularFile)
                    .forEach(path -> {
                        String fileName =
                                path.getFileName()
                                        .toString();

                        Long sequenceNumber =
                                numericPattern.tryExtractNumber(
                                        fileName
                                );

                        if (sequenceNumber != null) {
                            indexedPaths.add(
                                    new IndexedPath(
                                            path,
                                            fileName,
                                            sequenceNumber
                                    )
                            );
                        }
                    });
        }

        if (indexedPaths.isEmpty()) {
            throw new IOException(
                    "No numeric Parquet files in directory "
                            + directory
                            + " matched pattern: "
                            + patternText
            );
        }

        indexedPaths.sort(
                (first, second) -> {
                    int numericComparison =
                            Long.compare(
                                    first.sequenceNumber,
                                    second.sequenceNumber
                            );

                    if (numericComparison != 0) {
                        return numericComparison;
                    }

                    return first.fileName.compareTo(
                            second.fileName
                    );
                }
        );

        List<Path> files =
                new ArrayList<>(
                        indexedPaths.size()
                );

        for (IndexedPath indexedPath : indexedPaths) {
            files.add(
                    indexedPath.path
            );
        }

        return files;
    }

    /**
     * Validates a series returned by the specialized materializer and returns
     * its time length.
     */
    private int validateAndGetTimeLength(
            Object series,
            Path file,
            int instanceIndex
    ) {
        if (series == null) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned null for "
                            + "instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (series instanceof double[] data) {
            validateNonemptyLength(
                    data.length,
                    file,
                    instanceIndex
            );

            return data.length;
        }

        if (series instanceof Double[] data) {
            validateNonemptyLength(
                    data.length,
                    file,
                    instanceIndex
            );

            return data.length;
        }

        if (series instanceof double[][] data) {
            return validatePrimitiveMatrix(
                    data,
                    file,
                    instanceIndex
            );
        }

        if (series instanceof Double[][] data) {
            return validateBoxedMatrix(
                    data,
                    file,
                    instanceIndex
            );
        }

        throw new IllegalStateException(
                "Numeric Parquet series reader returned unsupported type "
                        + series.getClass()
                        .getTypeName()
                        + " for instance "
                        + instanceIndex
                        + " from file "
                        + file
                        + "."
        );
    }

    private int validatePrimitiveMatrix(
            double[][] data,
            Path file,
            int instanceIndex
    ) {
        if (data.length == 0) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned no dimensions "
                            + "for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (data[0] == null) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned a null first "
                            + "dimension for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        int expectedLength =
                data[0].length;

        validateNonemptyLength(
                expectedLength,
                file,
                instanceIndex
        );

        for (int dimension = 1;
             dimension < data.length;
             dimension++) {

            if (data[dimension] == null) {
                throw new IllegalStateException(
                        "Numeric Parquet series reader returned a null "
                                + "dimension "
                                + dimension
                                + " for instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + "."
                );
            }

            if (data[dimension].length != expectedLength) {
                throw new IllegalStateException(
                        "Numeric Parquet series reader returned "
                                + "inconsistent dimension lengths for "
                                + "instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ". Expected "
                                + expectedLength
                                + " time points but dimension "
                                + dimension
                                + " contains "
                                + data[dimension].length
                                + "."
                );
            }
        }

        return expectedLength;
    }

    private int validateBoxedMatrix(
            Double[][] data,
            Path file,
            int instanceIndex
    ) {
        if (data.length == 0) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned no dimensions "
                            + "for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (data[0] == null) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned a null first "
                            + "dimension for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        int expectedLength =
                data[0].length;

        validateNonemptyLength(
                expectedLength,
                file,
                instanceIndex
        );

        for (int dimension = 1;
             dimension < data.length;
             dimension++) {

            if (data[dimension] == null) {
                throw new IllegalStateException(
                        "Numeric Parquet series reader returned a null "
                                + "dimension "
                                + dimension
                                + " for instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + "."
                );
            }

            if (data[dimension].length != expectedLength) {
                throw new IllegalStateException(
                        "Numeric Parquet series reader returned "
                                + "inconsistent dimension lengths for "
                                + "instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ". Expected "
                                + expectedLength
                                + " time points but dimension "
                                + dimension
                                + " contains "
                                + data[dimension].length
                                + "."
                );
            }
        }

        return expectedLength;
    }

    private void validateNonemptyLength(
            int length,
            Path file,
            int instanceIndex
    ) {
        if (length == 0) {
            throw new IllegalStateException(
                    "Numeric Parquet series reader returned an empty "
                            + "time series for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }
    }

    /**
     * Placeholder for future per-file label conventions.
     *
     * <p>Potential sources include an external label file, a filename-based
     * mapping, a label encoded in the filename, or an external metadata
     * table.</p>
     */
    private Object inferLabel(
            Path file,
            int instanceIndex
    ) {
        return null;
    }

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires non-null "
                            + "ReaderOptions."
            );
        }

        return options;
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericPerFileParquetReader requires at least one "
                            + "feature column."
            );
        }

        List<String> copy =
                new ArrayList<>(
                        featureColumns.size()
                );

        Set<String> used =
                new HashSet<>();

        for (String featureColumn : featureColumns) {
            if (featureColumn == null
                    || featureColumn.isBlank()) {

                throw new IllegalArgumentException(
                        "Numeric Parquet feature-column names cannot be "
                                + "null or blank."
                );
            }

            String normalized =
                    featureColumn.trim();

            if (!used.add(normalized)) {
                throw new IllegalArgumentException(
                        "Numeric Parquet feature column was selected more "
                                + "than once: "
                                + normalized
                );
            }

            copy.add(
                    normalized
            );
        }

        return List.copyOf(
                copy
        );
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    private static final class IndexedPath {

        private final Path path;
        private final String fileName;
        private final long sequenceNumber;

        private IndexedPath(
                Path path,
                String fileName,
                long sequenceNumber
        ) {
            this.path =
                    path;

            this.fileName =
                    fileName;

            this.sequenceNumber =
                    sequenceNumber;
        }
    }

    /**
     * Converts a filename pattern containing exactly one numeric placeholder
     * into a regular expression.
     */
    private static final class NumericPattern {

        private static final Pattern PLACEHOLDER_PATTERN =
                Pattern.compile(
                        "\\{([A-Za-z_][A-Za-z0-9_]*)(?::0?(\\d+)d)?}"
                );

        private final Pattern regex;
        private final String numericFieldName;

        private NumericPattern(
                Pattern regex,
                String numericFieldName
        ) {
            this.regex =
                    regex;

            this.numericFieldName =
                    numericFieldName;
        }

        private static NumericPattern from(
                String filePattern
        ) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "Per-file pattern cannot be null or blank."
                );
            }

            Matcher placeholderMatcher =
                    PLACEHOLDER_PATTERN.matcher(
                            filePattern
                    );

            if (!placeholderMatcher.find()) {
                throw new IllegalArgumentException(
                        "Per-file pattern must contain exactly one "
                                + "numeric placeholder, such as {num}, "
                                + "{num:04d}, {run}, or {run:03d}: "
                                + filePattern
                );
            }

            String fieldName =
                    placeholderMatcher.group(
                            1
                    );

            String widthText =
                    placeholderMatcher.group(
                            2
                    );

            int placeholderStart =
                    placeholderMatcher.start();

            int placeholderEnd =
                    placeholderMatcher.end();

            if (placeholderMatcher.find()) {
                throw new IllegalArgumentException(
                        "Per-file pattern currently supports exactly one "
                                + "numeric placeholder: "
                                + filePattern
                );
            }

            String prefix =
                    filePattern.substring(
                            0,
                            placeholderStart
                    );

            String suffix =
                    filePattern.substring(
                            placeholderEnd
                    );

            String numericRegex =
                    widthText == null
                            ? "(\\d+)"
                            : "(\\d{"
                            + Integer.parseInt(
                            widthText
                    )
                            + "})";

            String regexText =
                    "^"
                            + globFragmentToRegex(prefix)
                            + numericRegex
                            + globFragmentToRegex(suffix)
                            + "$";

            return new NumericPattern(
                    Pattern.compile(
                            regexText
                    ),
                    fieldName
            );
        }

        private Long tryExtractNumber(
                String fileName
        ) {
            Matcher matcher =
                    regex.matcher(
                            fileName
                    );

            if (!matcher.matches()) {
                return null;
            }

            try {
                return Long.parseLong(
                        matcher.group(
                                1
                        )
                );
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Numeric field '"
                                + numericFieldName
                                + "' is too large in filename: "
                                + fileName,
                        e
                );
            }
        }

        private static String globFragmentToRegex(
                String fragment
        ) {
            StringBuilder regex =
                    new StringBuilder();

            StringBuilder literal =
                    new StringBuilder();

            for (int index = 0;
                 index < fragment.length();
                 index++) {

                char current =
                        fragment.charAt(
                                index
                        );

                if (current == '*') {
                    appendQuotedLiteral(
                            regex,
                            literal
                    );

                    regex.append(
                            ".*"
                    );
                } else if (current == '?') {
                    appendQuotedLiteral(
                            regex,
                            literal
                    );

                    regex.append(
                            "."
                    );
                } else {
                    literal.append(
                            current
                    );
                }
            }

            appendQuotedLiteral(
                    regex,
                    literal
            );

            return regex.toString();
        }

        private static void appendQuotedLiteral(
                StringBuilder regex,
                StringBuilder literal
        ) {
            if (literal.length() == 0) {
                return;
            }

            regex.append(
                    Pattern.quote(
                            literal.toString()
                    )
            );

            literal.setLength(
                    0
            );
        }
    }
}