package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
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
 * Eager DatasetReader for high-throughput numeric per-file delimited
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
 * <p>Every discovered file is materialized during {@link #read()} by
 * {@link NumericPerFileDelimitedSeriesReader}.</p>
 *
 * <p>This eager dataset reader always returns raw, unstandardized numeric
 * arrays. Standardization statistics may be present in {@link ReaderOptions}
 * because the same option object is also used by lazy readers, but eager
 * transformation is coordinated after reading. This prevents reader-time
 * transformation from being applied a second time by the eager workflow.</p>
 *
 * <p>Returned instance representation:</p>
 *
 * <pre>
 * double[dimension][time]
 * </pre>
 *
 * <p>This reader is deliberately specialized for numeric files without
 * missing values. Use {@link PerFileDelimitedReader} when numeric missing
 * values, nonnumeric values, or mixed data must be supported.</p>
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
 * per-file reader behavior. External label files, filename-based labels,
 * and metadata-table labels can be added separately without changing the
 * numeric series materializer.</p>
 *
 * <p>This initial implementation reads files sequentially. That gives us a
 * clean benchmark of the specialized FastCSV callback path without
 * introducing file-level parallelism as a confounding factor.</p>
 */
public class NumericPerFileDelimitedReader
        implements DatasetReader {

    private static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private final String dataPath;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final String filePattern;
    private final boolean isTest;
    private final boolean isRegression;
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
    public NumericPerFileDelimitedReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getFilePattern(),
                options.isTest(),
                options.isRegression(),
                options.getStandardizationStats(),
                DEFAULT_INITIAL_TIME_CAPACITY
        );

        if (!options.isNumeric()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires "
                            + "ReaderOptions.isNumeric=true."
            );
        }

        if (options.hasMissingValues()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader does not support "
                            + "missing values. Use PerFileDelimitedReader "
                            + "when hasMissingValues=true."
            );
        }
    }

    /**
     * Constructs the eager reader using the default initial time capacity.
     */
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
        this(
                dataPath,
                entrySeparator,
                hasHeader,
                timeColumn,
                featureColumns,
                filePattern,
                isTest,
                isRegression,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Constructs the eager reader with an explicit per-dimension initial time
     * capacity.
     *
     * <p>When the approximate number of time points per file is known,
     * supplying that number can reduce dynamic-buffer expansion and copying.
     * The value is an initial capacity rather than a required fixed length,
     * so unequal-length files remain supported.</p>
     */
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

        this.isTest =
                isTest;

        this.isRegression =
                isRegression;

        /*
         * Retain this constructor parameter for source compatibility and for
         * early feature-order validation. The eager reader deliberately does
         * not retain or apply the statistics. Lazy readers own reader-time
         * standardization; eager workflows transform the completed dataset.
         */
        if (standardizationStats != null
                && !this.featureColumns.isEmpty()) {

            standardizationStats.validateFeatureCompatibility(
                    this.featureColumns
            );
        }

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader initialTimeCapacity "
                            + "must be at least 1. Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.initialTimeCapacity =
                initialTimeCapacity;

        validateConstructionOptions();
    }

    /**
     * Discovers and eagerly materializes every configured numeric
     * per-instance file.
     *
     * @return eager dataset containing raw, unstandardized
     *         {@code double[][]} instances
     * @throws IOException if files cannot be discovered or materialized
     */
    @Override
    public ListObjectDataset read()
            throws IOException {

        validateReadOptions();

        List<Path> files =
                discoverFiles();

        NumericPerFileDelimitedSeriesReader seriesReader =
                new NumericPerFileDelimitedSeriesReader(
                        entrySeparator,
                        hasHeader,
                        timeColumn,
                        featureColumns,
                        null,
                        initialTimeCapacity
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
                    files.get(instanceIndex);

            double[][] series;

            try {
                series =
                        seriesReader.readFile(
                                file
                        );
            } catch (IOException e) {
                throw new IOException(
                        "Failed to eagerly materialize numeric "
                                + "per-file delimited instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ".",
                        e
                );
            } catch (RuntimeException e) {
                throw new IllegalArgumentException(
                        "Failed to eagerly materialize numeric "
                                + "per-file delimited instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ".",
                        e
                );
            }

            validateMaterializedSeries(
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

            int timeLength =
                    series[0].length;

            if (commonTimeLength < 0) {
                commonTimeLength =
                        timeLength;
            } else if (timeLength != commonTimeLength) {
                unequalTimeLengths =
                        true;
            }

            /*
             * ProgressLogger retains the same progress-reporting behavior as
             * the other eager delimited reader.
             */
            DelimitedFileReader.ProgressLogger.logProgress(
                    instanceIndex
            );
        }

        /*
         * ListObjectDataset.length and AppContext.length historically
         * represent one common per-instance time length. That value is only
         * meaningful when all materialized files have the same time length.
         *
         * A value of zero communicates that no single global length applies.
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
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires a non-empty "
                            + "entry separator."
            );
        }

        if (entrySeparator.length() != 1) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires a "
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

    }

    private void validateReadOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "NumericPerFileDelimitedReader requires dataPath."
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
                    "Numeric per-file delimited data path does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "NumericPerFileDelimitedReader requires "
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
                "Numeric per-file delimited data path must be a directory "
                        + "or a regular file: "
                        + dataPath
        );
    }

    /**
     * Discovers files and extracts each numerical sort key exactly once.
     *
     * <p>This avoids repeatedly applying the pattern and parsing the sequence
     * number from inside the sorting comparator.</p>
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

    private void validateMaterializedSeries(
            double[][] series,
            Path file,
            int instanceIndex
    ) {
        if (series == null) {
            throw new IllegalStateException(
                    "Numeric series reader returned null for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (series.length == 0) {
            throw new IllegalStateException(
                    "Numeric series reader returned no dimensions for "
                            + "instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (series[0] == null) {
            throw new IllegalStateException(
                    "Numeric series reader returned a null first "
                            + "dimension for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        int expectedTimeLength =
                series[0].length;

        if (expectedTimeLength == 0) {
            throw new IllegalStateException(
                    "Numeric series reader returned an empty time series "
                            + "for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        for (int dimension = 1;
             dimension < series.length;
             dimension++) {

            if (series[dimension] == null) {
                throw new IllegalStateException(
                        "Numeric series reader returned a null dimension "
                                + dimension
                                + " for instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + "."
                );
            }

            if (series[dimension].length != expectedTimeLength) {
                throw new IllegalStateException(
                        "Numeric series reader returned inconsistent "
                                + "dimension lengths for instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ". Expected "
                                + expectedTimeLength
                                + " time points but dimension "
                                + dimension
                                + " contains "
                                + series[dimension].length
                                + "."
                );
            }
        }
    }

    /**
     * Placeholder for future per-file label conventions.
     *
     * <p>Potential sources include an external label file, a filename-based
     * mapping, or an external metadata table.</p>
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
                    "NumericPerFileDelimitedReader requires "
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
                            + Integer.parseInt(widthText)
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
                        fragment.charAt(index);

                if (current == '*') {
                    appendQuotedLiteral(
                            regex,
                            literal
                    );

                    regex.append(".*");
                } else if (current == '?') {
                    appendQuotedLiteral(
                            regex,
                            literal
                    );

                    regex.append(".");
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