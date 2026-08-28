package datasets.readers.lazy;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.DatasetReader;
import datasets.readers.ReaderOptions;
import datasets.readers.ReaderType;
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
 * Lazy DatasetReader for high-throughput numeric per-file Parquet
 * time-series datasets.
 *
 * <p>Storage assumption:</p>
 *
 * <pre>
 * one Parquet file = one dataset instance / time series
 * one Parquet record = one time point
 * one projected feature column = one time-series dimension
 * </pre>
 *
 * <p>This reader does not decode Parquet contents while constructing the
 * {@link ListObjectDataset}. Each dataset instance is represented by a
 * {@link LazySeriesRef}.</p>
 *
 * <p>When a distance calculation requests an instance,
 * {@code NumericPerFileParquetSeriesReader} materializes the referenced file
 * using Hardwood's batch-oriented column-reading API.</p>
 *
 * <p>Returned representations:</p>
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
 * <p>This reader assumes that Parquet records are already stored in the
 * required time order. The optional time column is therefore not projected
 * or decoded merely for sorting. Physical Parquet record order is preserved.</p>
 *
 * <p>The reader supports null feature values when
 * {@code hasMissingValues=true}. Hardwood validity information is retained
 * separately from primitive numerical values during materialization.</p>
 *
 * <p>The current specialized series reader expects projected feature columns
 * to have Parquet physical type DOUBLE.</p>
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
 * <p>Labels are currently assigned as {@code null}, matching the existing
 * per-file reader family.</p>
 */
public class LazyPerFileNumericParquetReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            1024;

    private final String dataPath;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean hasMissingValues;
    private final String filePattern;
    private final String readerKey;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;

    /**
     * Constructs the lazy reader from ordinary PFGAP reader options.
     *
     * <p>The specialized reader requires
     * {@code ReaderOptions.isNumeric=true}. Missing values remain supported
     * and are controlled by {@code options.hasMissingValues()}.</p>
     *
     * @param options PFGAP reader options
     */
    public LazyPerFileNumericParquetReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.hasMissingValues(),
                options.getFilePattern(),
                options.isTest()
                        ? "test"
                        : "train",
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY
        );

        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericParquetReader requires "
                            + "ReaderOptions.isNumeric=true."
            );
        }
    }

    /**
     * Constructs the lazy reader using the default initial time capacity.
     */
    public LazyPerFileNumericParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats
    ) {
        this(
                dataPath,
                timeColumn,
                featureColumns,
                hasMissingValues,
                filePattern,
                readerKey,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Full constructor.
     *
     * @param dataPath             directory or single Parquet file
     * @param timeColumn           optional descriptive time column
     * @param featureColumns       projected DOUBLE feature columns
     * @param hasMissingValues     whether null feature values are permitted
     * @param filePattern          required for directory input
     * @param readerKey            runtime lazy-reader registry key
     * @param standardizationStats optional standardization statistics
     * @param initialTimeCapacity  allocation hint per feature dimension
     */
    public LazyPerFileNumericParquetReader(
            String dataPath,
            String timeColumn,
            List<String> featureColumns,
            boolean hasMissingValues,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
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

        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericParquetReader requires "
                            + "a non-empty readerKey."
            );
        }

        this.readerKey =
                readerKey.trim();

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericParquetReader "
                            + "initialTimeCapacity must be at least 1. "
                            + "Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        validateConstructionOptions();
    }

    /**
     * Discovers the configured Parquet files, registers the specialized
     * runtime series reader specification, and creates one lazy reference per
     * discovered file.
     *
     * @return lazy dataset containing {@link LazySeriesRef} instances
     * @throws IOException if file discovery fails
     */
    @Override
    public ListObjectDataset read()
            throws IOException {

        validateReadOptions();

        List<Path> files =
                discoverFiles();

        /*
         * Register the serializable reconstruction specification.
         *
         * The runtime factory maps this reader type to
         * NumericPerFileParquetSeriesReader. The series reader defaults to
         * physical Parquet file order, so no separate time-order field is
         * required in LazySeriesReaderSpec.
         *
         * entrySeparator and hasHeader are not applicable to Parquet and are
         * stored as null and false.
         */
        LazySeriesReaderSpec readerSpec =
                new LazySeriesReaderSpec(
                        readerKey,
                        ReaderType.LAZY_PER_FILE_NUMERIC_PARQUET,
                        timeColumn,
                        featureColumns,
                        true,
                        hasMissingValues,
                        null,
                        false,
                        standardizationStats,
                        initialTimeCapacity
                );

        AppContext.registerLazySeriesReader(
                readerSpec
        );

        ListObjectDataset dataset =
                new ListObjectDataset(
                        files.size()
                );

        for (int instanceIndex = 0;
             instanceIndex < files.size();
             instanceIndex++) {

            Path file =
                    files.get(
                            instanceIndex
                    );

            Object label =
                    inferLabel(
                            file,
                            instanceIndex
                    );

            LazySeriesRef reference =
                    new LazySeriesRef(
                            readerKey,
                            instanceIndex,
                            file
                    );

            dataset.add(
                    label,
                    reference,
                    instanceIndex
            );
        }

        /*
         * No Parquet contents are decoded during lazy dataset construction.
         * The per-file time lengths are consequently unknown and may differ.
         */
        dataset.setLength(
                0
        );

        return dataset;
    }

    private void validateConstructionOptions() {
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
                    "LazyPerFileNumericParquetReader requires dataPath."
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
                    "Lazy numeric per-file Parquet data path does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "LazyPerFileNumericParquetReader requires "
                                + "filePattern when dataPath is a directory. "
                                + "The pattern must contain exactly one "
                                + "numeric placeholder, such as "
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
                "Lazy numeric per-file Parquet data path must be "
                        + "a directory or a regular file: "
                        + dataPath
        );
    }

    /**
     * Discovers matching files and extracts the numerical sorting key once
     * per file.
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
     * Placeholder for future per-file label conventions.
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
                    "LazyPerFileNumericParquetReader requires "
                            + "non-null ReaderOptions."
            );
        }

        return options;
    }

    private static List<String> copyAndValidateFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericParquetReader requires at least "
                            + "one feature column."
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
                            + globFragmentToRegex(
                            prefix
                    )
                            + numericRegex
                            + globFragmentToRegex(
                            suffix
                    )
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