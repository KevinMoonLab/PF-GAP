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
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Lazy DatasetReader for high-throughput numeric per-file delimited
 * time-series datasets.
 *
 * <p>Storage assumption:</p>
 *
 * <pre>
 * one delimited file = one dataset instance / time series
 * one CSV record = one time point
 * one selected column = one time-series dimension
 * </pre>
 *
 * <p>This reader does not parse instance-file contents while constructing
 * the {@link ListObjectDataset}. Each dataset item is represented by a
 * {@link LazySeriesRef}.</p>
 *
 * <p>{@code NumericPerFileDelimitedSeriesReader} materializes an individual
 * file when a distance calculation requests the corresponding series.</p>
 *
 * <p>Materialized representation:</p>
 *
 * <pre>
 * double[dimension][time]
 * </pre>
 *
 * <p>This reader is deliberately specialized for numeric files without
 * missing values. It uses the custom FastCSV callback series reader rather
 * than the general-purpose per-file delimited series reader.</p>
 *
 * <p>The data path may identify:</p>
 *
 * <ol>
 *     <li>
 *         A directory containing one delimited file per instance. In this
 *         case, {@code filePattern} is required and must contain exactly one
 *         numeric placeholder.
 *     </li>
 *     <li>
 *         A single regular file. In this case, {@code filePattern} is not
 *         required and the resulting dataset contains one instance.
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
 *     series_{num:04d}.csv
 * </pre>
 *
 * <p>Other valid patterns include:</p>
 *
 * <pre>
 * trial_*.{run:03d}_freqN.tsv
 * series_{instance}.csv
 * sample_?_{num:05d}.txt
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
 * lazy per-file reader behavior.</p>
 */
public class LazyPerFileNumericDelimitedReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private final String dataPath;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final String filePattern;
    private final String readerKey;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;

    /**
     * Constructs the reader from ordinary PFGAP reader options.
     *
     * <p>The specialized reader requires:</p>
     *
     * <ul>
     *     <li>{@code isNumeric=true}</li>
     *     <li>{@code hasMissingValues=false}</li>
     * </ul>
     *
     * @param options PFGAP reader options
     */
    public LazyPerFileNumericDelimitedReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getFilePattern(),
                options.isTest()
                        ? "test"
                        : "train",
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY
        );

        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader requires "
                            + "ReaderOptions.isNumeric=true."
            );
        }

        if (options.hasMissingValues()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader does not support "
                            + "missing values. Use "
                            + "LazyPerFileDelimitedReader when "
                            + "hasMissingValues=true."
            );
        }
    }

    /**
     * Constructs the reader using the default initial time capacity.
     */
    public LazyPerFileNumericDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats
    ) {
        this(
                dataPath,
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                filePattern,
                readerKey,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Constructs the reader with an explicit initial time capacity.
     *
     * <p>The capacity is passed to the reconstructed
     * {@code NumericPerFileDelimitedSeriesReader}. It controls only the first
     * allocation for each selected dimension. It does not constrain the final
     * series length.</p>
     */
    public LazyPerFileNumericDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            String filePattern,
            String readerKey,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        this.dataPath =
                normalizeNullableString(
                        dataPath
                );

        this.entrySeparator =
                normalizeSeparator(
                        entrySeparator
                );

        this.hasHeader =
                hasHeader;

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                featureColumns == null
                        ? List.of()
                        : List.copyOf(
                        featureColumns
                );

        this.filePattern =
                normalizeNullableString(
                        filePattern
                );

        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader requires "
                            + "a non-empty readerKey."
            );
        }

        this.readerKey =
                readerKey.trim();

        this.standardizationStats =
                standardizationStats;

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader "
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
     * Discovers the configured files, registers the specialized runtime
     * series reader, and creates one lazy reference per file.
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
         * Register both:
         *
         * 1. The runtime NumericPerFileDelimitedSeriesReader.
         * 2. Its serializable reconstruction specification.
         *
         * The saved-model snapshot retains the training specification so
         * training LazySeriesRef instances can still be materialized in a
         * fresh JVM.
         */
        LazySeriesReaderSpec readerSpec =
                new LazySeriesReaderSpec(
                        readerKey,
                        ReaderType.LAZY_PER_FILE_NUMERIC_DELIMITED,
                        timeColumn,
                        featureColumns,
                        true,
                        false,
                        entrySeparator,
                        hasHeader,
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
                    files.get(instanceIndex);

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
         * The files may contain unequal numbers of time points. No instance
         * file is opened during lazy dataset construction, so a common time
         * length is unavailable and may not exist.
         */
        dataset.setLength(
                0
        );

        return dataset;
    }

    private void validateConstructionOptions() {
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader requires "
                            + "a non-empty entry separator."
            );
        }

        if (entrySeparator.length() != 1) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader requires a "
                            + "single-character entry separator. Received: '"
                            + entrySeparator
                            + "'."
            );
        }

        char separator =
                entrySeparator.charAt(0);

        if (separator == '\n' || separator == '\r') {
            throw new IllegalArgumentException(
                    "A line-separator character cannot be used as "
                            + "the entry separator."
            );
        }

        for (String featureColumn : featureColumns) {
            if (featureColumn == null
                    || featureColumn.isBlank()) {

                throw new IllegalArgumentException(
                        "Numeric per-file feature-column names or indices "
                                + "cannot be null or blank."
                );
            }
        }

        if (standardizationStats != null
                && !featureColumns.isEmpty()) {

            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private void validateReadOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyPerFileNumericDelimitedReader requires dataPath."
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
                    "Lazy numeric per-file delimited data path "
                            + "does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "LazyPerFileNumericDelimitedReader requires "
                                + "filePattern when dataPath is a directory. "
                                + "The pattern must contain exactly one "
                                + "numeric placeholder, such as "
                                + "series_{num:04d}.csv or "
                                + "trial_*.{run:03d}_freqN.tsv."
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
                "Lazy numeric per-file delimited data path must be "
                        + "a directory or a regular file: "
                        + dataPath
        );
    }

    /**
     * Discovers matching files and extracts every numeric sort key exactly
     * once before sorting.
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
                    "No numeric delimited files in directory "
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
     *
     * <p>Potential label sources include an external label file, a
     * filename-to-label mapping, a label encoded in the filename, or an
     * external metadata table.</p>
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
                    "LazyPerFileNumericDelimitedReader requires "
                            + "non-null ReaderOptions."
            );
        }

        return options;
    }

    private static String normalizeSeparator(
            String separator
    ) {
        if (separator == null) {
            return null;
        }

        return switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
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
                    placeholderMatcher.group(1);

            String widthText =
                    placeholderMatcher.group(2);

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
                        matcher.group(1)
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