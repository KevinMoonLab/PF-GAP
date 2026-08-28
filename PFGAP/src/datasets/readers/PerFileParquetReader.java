package datasets.readers;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

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
 * Eager DatasetReader for per-file Parquet time-series datasets.
 *
 * Storage assumption:
 *
 *      one Parquet file = one dataset instance / one time series
 *
 * Unlike LazyPerFileParquetReader, this reader materializes every series
 * while constructing the ListObjectDataset. The dataset therefore contains
 * the realized series objects returned by PerFileParquetSeriesReader,
 * rather than LazySeriesRef objects.
 *
 * The dataPath may identify:
 *
 *      1. A directory containing per-instance Parquet files.
 *         In this case, filePattern is required and must contain exactly
 *         one numeric placeholder.
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
 *      {num:04d}
 *      {run}
 *      {run:03d}
 *
 * Glob wildcards '*' and '?' may appear outside the numeric placeholder.
 *
 * Labels:
 *
 *      This initial implementation assigns null labels, matching the
 *      current LazyPerFileParquetReader behavior. External label-file
 *      support can be added separately without changing the file discovery
 *      or Parquet materialization logic.
 */
public class PerFileParquetReader implements DatasetReader {

    private final String dataPath;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;
    private final String filePattern;

    public PerFileParquetReader(
            ReaderOptions options
    ) {
        this(
                options.getDataPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns(),
                options.getFilePattern()
        );
    }

    public PerFileParquetReader(
            String dataPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns,
            String filePattern
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

        this.filePattern =
                filePattern == null || filePattern.isBlank()
                        ? null
                        : filePattern.trim();
    }

    @Override
    public ListObjectDataset read() throws IOException {
        validateOptions();

        List<Path> files =
                discoverFiles();

        /*
         * Reuse the same single-file Parquet materialization logic used by
         * the lazy reader. No runtime reader registration is required because
         * every temporary reference is resolved immediately.
         */
        PerFileParquetSeriesReader seriesReader =
                new PerFileParquetSeriesReader(
                        timeColumn,
                        featureColumns,
                        isNumeric,
                        hasMissingValues
                );

        ListObjectDataset dataset =
                new ListObjectDataset(files.size());

        for (int i = 0; i < files.size(); i++) {
            Path file =
                    files.get(i);

            Object label =
                    inferLabel(file, i);

            /*
             * This reference exists only long enough to invoke the shared
             * single-file reader. It is not stored in the dataset.
             */
            LazySeriesRef temporaryReference =
                    new LazySeriesRef(
                            "eager-per-file-parquet",
                            i,
                            file
                    );

            Object materializedSeries =
                    seriesReader.read(
                            temporaryReference
                    );

            dataset.add(
                    label,
                    materializedSeries,
                    i
            );
        }

        /*
         * Per-file series may have unequal lengths. A single global series
         * length is therefore not meaningful.
         *
         * If ListObjectDataset.add() updates this value for each instance,
         * explicitly reset it after all instances have been added.
         */
        dataset.setLength(0);

        return dataset;
    }

    private void validateOptions() {
        if (dataPath == null || dataPath.isBlank()) {
            throw new IllegalArgumentException(
                    "PerFileParquetReader requires dataPath."
            );
        }

        if (featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileParquetReader requires at least one "
                            + "feature column."
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
                        "PerFileParquetReader requires file_pattern when "
                                + "dataPath is a directory. The pattern must "
                                + "contain one numeric placeholder, such as "
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
            if (!isParquetFile(path)) {
                throw new IOException(
                        "PerFileParquetReader expected a .parquet file: "
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

    private boolean isParquetFile(
            Path path
    ) {
        return path.getFileName()
                .toString()
                .toLowerCase()
                .endsWith(".parquet");
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
                    "No Parquet files in directory "
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
         * Deterministic tie-breaker when multiple filenames contain the
         * same numeric value.
         */
        return firstName.compareTo(
                secondName
        );
    }

    /**
     * Placeholder for future label conventions.
     *
     * Potential sources include:
     *
     *      filename-to-label mappings
     *      an external label file
     *      labels encoded in filenames
     *      labels stored in a metadata table
     *
     * For now, labels remain null so this reader can be used for isolation
     * and other unsupervised workflows in the same way as the lazy reader.
     */
    private Object inferLabel(
            Path file,
            int instanceIndex
    ) {
        return null;
    }

    /**
     * Converts a filename pattern containing exactly one numbered
     * placeholder into a regular expression.
     *
     * Examples:
     *
     *      series_{num}.parquet
     *      series_{num:04d}.parquet
     *      trial_*.{run:03d}_freqN.parquet
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
            this.regex = regex;
            this.numericFieldName = numericFieldName;
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

            for (int i = 0;
                 i < fragment.length();
                 i++) {

                char current =
                        fragment.charAt(i);

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