package datasets.readers;

import datasets.ListObjectDataset;

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
 * Eager DatasetReader for per-file delimited time-series datasets.
 *
 * Storage assumption:
 *
 *      one delimited file = one dataset instance / one time series
 *
 * Each instance file is materialized while read() constructs the
 * ListObjectDataset. The resulting dataset therefore contains realized
 * series objects rather than LazySeriesRef objects.
 *
 * Expected instance-file organization:
 *
 *      one row = one time point
 *      one selected column = one time-series dimension
 *
 * Returned instance representation:
 *
 *      numeric data:
 *          Double[dimension][time]
 *
 *      nonnumeric data:
 *          Object[dimension][time]
 *
 * The dataPath may identify:
 *
 *      1. A directory containing per-instance delimited files.
 *         In this case, filePattern is required and must contain exactly
 *         one numeric placeholder.
 *
 *      2. A single regular file.
 *         In this case, filePattern is not required.
 *
 * Example directory configuration:
 *
 *      dataPath:
 *          /path/to/training
 *
 *      filePattern:
 *          series_{num:04d}.csv
 *
 * Another valid pattern:
 *
 *      trial_*.{run:03d}_freqN.tsv
 *
 * Supported numeric placeholders include:
 *
 *      {num}
 *      {num:04d}
 *      {run}
 *      {run:03d}
 *
 * Glob wildcards '*' and '?' may appear outside the numeric placeholder.
 *
 * Labels:
 *
 *      This initial implementation assigns null labels, matching the
 *      current per-file Parquet readers. External label-file support can
 *      be added separately without changing discovery or materialization.
 */
public class PerFileDelimitedReader implements DatasetReader {

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

    public PerFileDelimitedReader(
            ReaderOptions options
    ) {
        this(
                options.getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.getFilePattern()
        );
    }

    public PerFileDelimitedReader(
            String dataPath,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern
    ) {
        this.dataPath =
                normalizeNullableString(dataPath);

        this.entrySeparator =
                normalizeSeparator(entrySeparator);

        this.hasHeader =
                hasHeader;

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.isRegression =
                isRegression;

        this.timeColumn =
                normalizeNullableString(timeColumn);

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.labelColumns =
                labelColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(labelColumns);

        this.filePattern =
                normalizeNullableString(filePattern);
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateOptions();

        List<Path> files =
                discoverFiles();

        /*
         * PerFileDelimitedSeriesReader contains the shared single-file
         * materialization behavior used by both eager and lazy dataset
         * readers.
         */
        PerFileDelimitedSeriesReader seriesReader =
                new PerFileDelimitedSeriesReader(
                        entrySeparator,
                        hasHeader,
                        timeColumn,
                        featureColumns,
                        isNumeric,
                        hasMissingValues
                );

        ListObjectDataset dataset =
                new ListObjectDataset(files.size());

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

            Object materializedSeries =
                    seriesReader.readFile(file);

            dataset.add(
                    label,
                    materializedSeries,
                    instanceIndex
            );
        }

        /*
         * Different per-file instances may contain different numbers of
         * time points. A single global series length is therefore not
         * meaningful.
         *
         * Reset this after all calls to dataset.add() in case add() updates
         * the dataset length based on the most recently added instance.
         */
        dataset.setLength(0);

        return dataset;
    }

    private void validateOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedReader requires dataPath."
            );
        }

        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedReader requires a non-empty "
                            + "entry separator."
            );
        }

        /*
         * Unlike the Parquet reader, an empty feature-column list is valid.
         *
         * PerFileDelimitedSeriesReader interprets it as:
         *
         *      use every column except the configured time column
         *
         * This is especially convenient for headerless files.
         */
    }

    private List<Path> discoverFiles() throws IOException {
        Path path =
                Paths.get(dataPath);

        if (!Files.exists(path)) {
            throw new IOException(
                    "Per-file delimited data path does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "PerFileDelimitedReader requires file_pattern "
                                + "when dataPath is a directory. The pattern "
                                + "must contain one numeric placeholder, such "
                                + "as series_{num:04d}.csv or "
                                + "trial_*.{run:03d}_freqN.tsv."
                );
            }

            return discoverFromPattern(
                    path,
                    filePattern
            );
        }

        if (Files.isRegularFile(path)) {
            return List.of(path);
        }

        throw new IOException(
                "Per-file delimited data path must be a directory "
                        + "or a regular file: "
                        + dataPath
        );
    }

    private List<Path> discoverFromPattern(
            Path directory,
            String patternText
    ) throws IOException {

        NumericPattern numericPattern =
                NumericPattern.from(
                        patternText
                );

        List<Path> files;

        try (Stream<Path> stream =
                     Files.list(directory)) {

            files = stream
                    .filter(Files::isRegularFile)
                    .filter(path ->
                            numericPattern.matches(
                                    path.getFileName()
                                            .toString()
                            )
                    )
                    .sorted((first, second) ->
                            compareMatchingFiles(
                                    first,
                                    second,
                                    numericPattern
                            )
                    )
                    .toList();
        }

        if (files.isEmpty()) {
            throw new IOException(
                    "No delimited files in directory "
                            + directory
                            + " matched pattern: "
                            + patternText
            );
        }

        return files;
    }

    private int compareMatchingFiles(
            Path first,
            Path second,
            NumericPattern numericPattern
    ) {
        String firstName =
                first.getFileName()
                        .toString();

        String secondName =
                second.getFileName()
                        .toString();

        long firstNumber =
                numericPattern.extractNumber(
                        firstName
                );

        long secondNumber =
                numericPattern.extractNumber(
                        secondName
                );

        int numericComparison =
                Long.compare(
                        firstNumber,
                        secondNumber
                );

        if (numericComparison != 0) {
            return numericComparison;
        }

        /*
         * Deterministic tie-breaker for patterns that allow multiple
         * filenames to contain the same numeric field.
         */
        return firstName.compareTo(
                secondName
        );
    }

    /**
     * Placeholder for future per-file label conventions.
     *
     * Potential label sources include:
     *
     *      an external label file
     *      a filename-to-label mapping
     *      a label encoded in the filename
     *      an external metadata table
     *
     * labelColumns are retained as reader configuration for future use,
     * but labels currently remain null.
     */
    private Object inferLabel(
            Path file,
            int instanceIndex
    ) {
        return null;
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

    /**
     * Converts a filename pattern containing exactly one numeric
     * placeholder into a regular expression and exposes the extracted
     * numeric value for deterministic ordering.
     *
     * Examples:
     *
     *      series_{num}.csv
     *      series_{num:04d}.tsv
     *      trial_*.{run:03d}_freqN.txt
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
            if (filePattern == null
                    || filePattern.isBlank()) {

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
                        "Per-file pattern currently supports exactly "
                                + "one numeric placeholder: "
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
                    Pattern.compile(regexText),
                    fieldName
            );
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
                    literal.append(current);
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

            literal.setLength(0);
        }

        private boolean matches(
                String fileName
        ) {
            return regex.matcher(
                    fileName
            ).matches();
        }

        private long extractNumber(
                String fileName
        ) {
            Matcher matcher =
                    regex.matcher(fileName);

            if (!matcher.matches()) {
                throw new IllegalArgumentException(
                        "File does not match per-file pattern: "
                                + fileName
                );
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
    }
}