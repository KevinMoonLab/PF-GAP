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
// import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * DatasetReader for lazy per-file Parquet time-series datasets.
 *
 * Storage assumption:
 *
 *      one Parquet file = one dataset instance / one time series
 *
 * The dataPath may identify:
 *
 *      1. A directory containing the per-instance Parquet files.
 *         In this case, filePattern must identify and numerically order
 *         the desired files.
 *
 *      2. A single Parquet file.
 *         In this case, filePattern is not required.
 *
 * Example directory configuration:
 *
 *      dataPath:
 *          /path/to/training
 *
 *      filePattern:
 *          trial_*.{run:03d}_freqN.parquet
 *
 * Supported numeric placeholders include:
 *
 *      {num}
 *      {num:03d}
 *      {run}
 *      {run:03d}
 *
 * Glob wildcards '*' and '?' may appear outside the numeric placeholder.
 *
 * This reader does not deserialize the time-series values while building
 * the ListObjectDataset. Each dataset item is represented by a
 * LazySeriesRef. PerFileParquetSeriesReader materializes an individual
 * series only when requested by lazy distance evaluation.
 *
 * Labels:
 *
 *      This initial implementation assigns null labels.
 *      External label-file support can be added without changing the lazy
 *      data path.
 */
public class LazyPerFileParquetReader implements DatasetReader {

    private final String dataPath;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final String readerKey;
    private final String filePattern;
    private final StandardizationStats standardizationStats;

    public LazyPerFileParquetReader(ReaderOptions options) {
        this(
                options.getDataPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.isTest() ? "test" : "train",
                options.getFilePattern(),
                options.getStandardizationStats()
        );
    }

    public LazyPerFileParquetReader(
            String dataPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String readerKey,
            String filePattern,
            StandardizationStats standardizationStats
    ) {
        this.dataPath = dataPath;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.timeColumn = timeColumn;

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.labelColumns =
                labelColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(labelColumns);

        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyPerFileParquetReader requires a non-empty readerKey."
            );
        }

        this.readerKey = readerKey;

        this.filePattern =
                filePattern == null || filePattern.isBlank()
                        ? null
                        : filePattern.trim();

        this.standardizationStats = standardizationStats;
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateOptions();

        List<Path> files =
                discoverFiles();

        /*AppContext.registerLazySeriesReader(
                readerKey,
                new PerFileParquetSeriesReader(
                        timeColumn,
                        featureColumns,
                        isNumeric,
                        hasMissingValues
                )
        );*/

        LazySeriesReaderSpec readerSpec =
                new LazySeriesReaderSpec(
                        readerKey,
                        ReaderType.LAZY_PER_FILE_PARQUET,
                        timeColumn,
                        featureColumns,
                        isNumeric,
                        hasMissingValues,
                        standardizationStats
                );

        AppContext.registerLazySeriesReader(
                readerSpec
        );

        ListObjectDataset dataset =
                new ListObjectDataset(files.size());

        for (int i = 0; i < files.size(); i++) {
            Path file =
                    files.get(i);

            Object label =
                    inferLabel(file, i);

            LazySeriesRef ref =
                    new LazySeriesRef(
                            readerKey,
                            i,
                            file
                    );

            dataset.add(
                    label,
                    ref,
                    i
            );
        }

        /*
         * Individual series may have unequal lengths. No series files are
         * opened while constructing this dataset.
         */
        dataset.setLength(0);

        return dataset;
    }

    private void validateOptions() {
        if (dataPath == null ||
                dataPath.trim().isEmpty()) {

            throw new IllegalArgumentException(
                    "LazyPerFileParquetReader requires dataPath."
            );
        }

        if (featureColumns == null ||
                featureColumns.isEmpty()) {

            throw new IllegalArgumentException(
                    "LazyPerFileParquetReader requires at least one feature column."
            );
        }
    }

    private List<Path> discoverFiles() throws IOException {
        Path path =
                Paths.get(dataPath);

        if (!Files.exists(path)) {
            throw new IOException(
                    "Per-file Parquet data path does not exist: "
                            + dataPath
            );
        }

        if (Files.isDirectory(path)) {
            if (filePattern == null || filePattern.isBlank()) {
                throw new IllegalArgumentException(
                        "LazyPerFileParquetReader requires file_pattern when "
                                + "dataPath is a directory. The pattern must "
                                + "contain one numeric placeholder, such as "
                                + "trial_*.{run:03d}_freq.parquet."
                );
            }

            return discoverFromPattern(
                    path,
                    filePattern
            );
        }

        if (Files.isRegularFile(path)) {
            if (!path.getFileName()
                    .toString()
                    .toLowerCase()
                    .endsWith(".parquet")) {

                throw new IOException(
                        "LazyPerFileParquetReader expected a .parquet file: "
                                + dataPath
                );
            }

            return List.of(path);
        }

        throw new IOException(
                "Per-file Parquet data path must be a directory "
                        + "or a regular .parquet file: "
                        + dataPath
        );
    }

    /*private List<Path> discoverFromDirectory(
            Path directory
    ) throws IOException {

        try (Stream<Path> stream = Files.list(directory)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path ->
                            path.getFileName()
                                    .toString()
                                    .toLowerCase()
                                    .endsWith(".parquet")
                    )
                    .sorted(Comparator.comparing(
                            path -> path.getFileName().toString()
                    ))
                    .toList();
        }
    }*/

    /*private List<Path> discoverFromNumericPattern(
            String patternWithNum
    ) throws IOException {

        Path patternPath =
                Paths.get(patternWithNum);

        Path parent =
                patternPath.getParent();

        if (parent == null) {
            parent = Paths.get(".");
        }

        String filePattern =
                patternPath.getFileName().toString();

        NumericPattern numericPattern =
                NumericPattern.from(filePattern);

        try (Stream<Path> stream = Files.list(parent)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(path ->
                            numericPattern.matches(
                                    path.getFileName().toString()
                            )
                    )
                    .sorted((a, b) ->
                            Integer.compare(
                                    numericPattern.extractNumber(
                                            a.getFileName().toString()
                                    ),
                                    numericPattern.extractNumber(
                                            b.getFileName().toString()
                                    )
                            )
                    )
                    .toList();
        }
    }*/

    private List<Path> discoverFromPattern(
            Path directory,
            String filePattern
    ) throws IOException {

        NumericPattern numericPattern =
                NumericPattern.from(filePattern);

        List<Path> files;

        try (Stream<Path> stream = Files.list(directory)) {
            files = stream
                    .filter(Files::isRegularFile)
                    .filter(path ->
                            numericPattern.matches(
                                    path.getFileName().toString()
                            )
                    )
                    .sorted((a, b) -> {
                        long aNumber =
                                numericPattern.extractNumber(
                                        a.getFileName().toString()
                                );

                        long bNumber =
                                numericPattern.extractNumber(
                                        b.getFileName().toString()
                                );

                        int numericComparison =
                                Long.compare(
                                        aNumber,
                                        bNumber
                                );

                        if (numericComparison != 0) {
                            return numericComparison;
                        }

                        /*
                         * Stable deterministic tie-breaker in case two filenames
                         * contain the same numeric field.
                         */
                        return a.getFileName()
                                .toString()
                                .compareTo(
                                        b.getFileName().toString()
                                );
                    })
                    .toList();
        }

        if (files.isEmpty()) {
            throw new IOException(
                    "No Parquet files in directory "
                            + directory
                            + " matched pattern: "
                            + filePattern
            );
        }

        return files;
    }

    /**
     * Placeholder for future label conventions.
     *
     * Later options:
     *
     *      filename -> label from CSV
     *      filename -> label encoded in name
     *      external metadata table
     *
     * For now, return null so unsupervised/outlier workflows still work,
     * and supervised workflows can be wired once label conventions are fixed.
     */
    private Object inferLabel(
            Path file,
            int instanceIndex
    ) {
        return null;
    }

    private static class NumericPattern {

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
            this.regex = regex;
            this.numericFieldName = numericFieldName;
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
                    PLACEHOLDER_PATTERN.matcher(filePattern);

            if (!placeholderMatcher.find()) {
                throw new IllegalArgumentException(
                        "Per-file pattern must contain one numeric placeholder, "
                                + "such as {num}, {num:03d}, {run}, or {run:03d}: "
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
                            : "(\\d{" + Integer.parseInt(widthText) + "})";

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

            for (int i = 0; i < fragment.length(); i++) {
                char current =
                        fragment.charAt(i);

                if (current == '*') {
                    appendQuotedLiteral(regex, literal);
                    regex.append(".*");
                } else if (current == '?') {
                    appendQuotedLiteral(regex, literal);
                    regex.append(".");
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

            regex.append(
                    Pattern.quote(literal.toString())
            );

            literal.setLength(0);
        }

        private boolean matches(
                String fileName
        ) {
            return regex.matcher(fileName).matches();
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