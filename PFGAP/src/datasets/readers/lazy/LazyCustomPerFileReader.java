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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Lazy per-file dataset reader backed by a user-defined Java series-reader
 * plugin.
 *
 * <p>PFGAP is responsible for:</p>
 *
 * <ul>
 *     <li>Discovering instance files</li>
 *     <li>Matching the configured filename pattern</li>
 *     <li>Sorting files deterministically</li>
 *     <li>Creating one {@link LazySeriesRef} per file</li>
 *     <li>Registering the custom reader reconstruction specification</li>
 *     <li>Reconstructing the custom plugin when a saved model is loaded</li>
 * </ul>
 *
 * <p>The user-supplied plugin is responsible only for converting one
 * {@link LazySeriesRef} into one non-null raw instance object. When prepared
 * standardization statistics are supplied for a numeric custom reader, PFGAP
 * wraps the reconstructed plugin reader and standardizes each materialized
 * instance exactly once.</p>
 *
 * <p>Example descriptor:</p>
 *
 * <pre>
 * javareader:/path/to/readers.jar:proprietary.reader.BinReader
 * </pre>
 *
 * <p>The plugin must implement:</p>
 *
 * <pre>
 * datasets.readers.api.CustomSeriesReader
 * </pre>
 *
 * <p>No instance file is opened during {@link #read()}. The custom plugin is
 * loaded by the lazy-series reader factory and invoked only when PFGAP
 * requests a referenced instance.</p>
 *
 * <p>The same plugin can be used with both:</p>
 *
 * <pre>
 * ReaderType.PER_FILE_CUSTOM
 * ReaderType.LAZY_PER_FILE_CUSTOM
 * </pre>
 *
 * <p>Labels are currently assigned as {@code null}. Separate label-file
 * support can be added to this wrapper without changing the custom plugin
 * interface.</p>
 */
public final class LazyCustomPerFileReader
        implements DatasetReader {

    private final String dataPath;
    private final String filePattern;
    private final String customReaderDescriptor;
    private final Map<String, String> customReaderParameters;
    private final List<String> featureColumns;

    private final boolean isTest;
    private final boolean isRegression;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean customReaderThreadSafe;
    private final StandardizationStats standardizationStats;

    private final String readerKey;

    /**
     * Constructs the lazy custom reader from PFGAP reader options.
     *
     * @param options reader configuration
     */
    public LazyCustomPerFileReader(
            ReaderOptions options
    ) {
        this(
                requireOptions(options).getDataPath(),
                options.getFilePattern(),
                options.getCustomReaderDescriptor(),
                options.getCustomReaderParameters(),
                options.getFeatureColumns(),
                options.isTest(),
                options.isRegression(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isCustomReaderThreadSafe(),
                options.getStandardizationStats(),
                options.isTest()
                        ? "test"
                        : "train"
        );
    }

    /**
     * Constructs a lazy custom reader using synchronized plugin invocation.
     */
    public LazyCustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            String readerKey
    ) {
        this(
                dataPath,
                filePattern,
                customReaderDescriptor,
                customReaderParameters,
                List.of(),
                isTest,
                isRegression,
                isNumeric,
                hasMissingValues,
                false,
                null,
                readerKey
        );
    }

    /**
     * Full constructor.
     *
     * @param dataPath                directory containing instance files, or
     *                                one regular instance file
     * @param filePattern             required for directory input
     * @param customReaderDescriptor  plugin JAR and implementation class
     * @param customReaderParameters  plugin-specific configuration
     * @param isTest                  whether test data is being read
     * @param isRegression            whether this is a regression task
     * @param isNumeric               numerical-output hint
     * @param hasMissingValues        missing-value hint
     * @param customReaderThreadSafe  whether concurrent plugin invocation is
     *                                permitted
     * @param readerKey               lazy-reader registry key
     */
    /**
     * Full constructor including feature metadata and optional prepared
     * standardization statistics.
     */
    public LazyCustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            List<String> featureColumns,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe,
            StandardizationStats standardizationStats,
            String readerKey
    ) {
        this.dataPath =
                requireNonblank(
                        dataPath,
                        "dataPath"
                );

        this.filePattern =
                normalizeNullableString(
                        filePattern
                );

        this.customReaderDescriptor =
                requireNonblank(
                        customReaderDescriptor,
                        "customReaderDescriptor"
                );

        this.customReaderParameters =
                copyParameters(
                        customReaderParameters
                );

        this.featureColumns =
                featureColumns == null
                        ? List.of()
                        : List.copyOf(
                        featureColumns
                );

        this.isTest =
                isTest;

        this.isRegression =
                isRegression;

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.customReaderThreadSafe =
                customReaderThreadSafe;

        if (standardizationStats != null && !isNumeric) {
            throw new IllegalArgumentException(
                    "Lazy custom standardization requires isNumeric=true."
            );
        }

        if (standardizationStats != null
                && !this.featureColumns.isEmpty()) {

            standardizationStats.validateFeatureCompatibility(
                    this.featureColumns
            );
        }

        this.standardizationStats =
                standardizationStats;

        this.readerKey =
                requireNonblank(
                        readerKey,
                        "readerKey"
                );
    }

    public LazyCustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe,
            String readerKey
    ) {
        this(
                dataPath,
                filePattern,
                customReaderDescriptor,
                customReaderParameters,
                List.of(),
                isTest,
                isRegression,
                isNumeric,
                hasMissingValues,
                customReaderThreadSafe,
                null,
                readerKey
        );
    }

    public LazyCustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe,
            StandardizationStats standardizationStats,
            String readerKey
    ) {
        this(
                dataPath,
                filePattern,
                customReaderDescriptor,
                customReaderParameters,
                List.of(),
                isTest,
                isRegression,
                isNumeric,
                hasMissingValues,
                customReaderThreadSafe,
                standardizationStats,
                readerKey
        );
    }

    /**
     * Discovers the configured files and creates one lazy reference per file.
     *
     * <p>The custom plugin is not loaded or invoked by this method.</p>
     *
     * @return lazy dataset containing {@link LazySeriesRef} instances
     * @throws IOException if file discovery fails
     */
    @Override
    public ListObjectDataset read()
            throws IOException {

        List<Path> files =
                discoverFiles();

        Path configuredDataPath =
                Path.of(
                                dataPath
                        )
                        .toAbsolutePath()
                        .normalize();

        /*
         * Runtime plugin instances and class loaders are not serialized.
         *
         * This specification contains enough information for
         * LazySeriesReaderFactory to reconstruct:
         *
         *     CustomReaderContext
         *     JavaSeriesReader
         *
         * in the current JVM or in a fresh JVM after loading a saved model.
         */
        LazySeriesReaderSpec readerSpec =
                new LazySeriesReaderSpec(
                        readerKey,
                        ReaderType.LAZY_PER_FILE_CUSTOM,
                        null,
                        featureColumns,
                        isNumeric,
                        hasMissingValues,
                        null,
                        false,
                        standardizationStats,
                        LazySeriesReaderSpec
                                .DEFAULT_INITIAL_TIME_CAPACITY,
                        customReaderDescriptor,
                        customReaderParameters,
                        customReaderThreadSafe,
                        configuredDataPath.toString(),
                        isTest,
                        isRegression
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
         * Instance files are not materialized during construction, so their
         * lengths are unknown and may differ.
         */
        dataset.setLength(
                0
        );

        return dataset;
    }

    private List<Path> discoverFiles()
            throws IOException {

        Path path =
                Path.of(
                                dataPath
                        )
                        .toAbsolutePath()
                        .normalize();

        if (!Files.exists(path)) {
            throw new IOException(
                    "Lazy custom per-file data path does not exist: "
                            + path
            );
        }

        if (Files.isRegularFile(path)) {
            return List.of(
                    path
            );
        }

        if (!Files.isDirectory(path)) {
            throw new IOException(
                    "Lazy custom per-file data path must be a regular file "
                            + "or directory: "
                            + path
            );
        }

        if (filePattern == null) {
            throw new IllegalArgumentException(
                    "LazyCustomPerFileReader requires filePattern when "
                            + "dataPath is a directory."
            );
        }

        return discoverFromPattern(
                path,
                filePattern
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

        List<IndexedPath> indexedPaths =
                new ArrayList<>();

        try (Stream<Path> stream =
                     Files.list(
                             directory
                     )) {

            stream.filter(
                            Files::isRegularFile
                    )
                    .forEach(path -> {
                        String fileName =
                                path.getFileName()
                                        .toString();

                        Long sequence =
                                numericPattern.tryExtractNumber(
                                        fileName
                                );

                        if (sequence != null) {
                            indexedPaths.add(
                                    new IndexedPath(
                                            path,
                                            fileName,
                                            sequence
                                    )
                            );
                        }
                    });
        }

        if (indexedPaths.isEmpty()) {
            throw new IOException(
                    "No files in directory "
                            + directory
                            + " matched lazy custom per-file pattern: "
                            + patternText
            );
        }

        indexedPaths.sort(
                (first, second) -> {
                    int sequenceComparison =
                            Long.compare(
                                    first.sequenceNumber,
                                    second.sequenceNumber
                            );

                    if (sequenceComparison != 0) {
                        return sequenceComparison;
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
     * Placeholder for future separate-label support.
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
                    "LazyCustomPerFileReader requires non-null "
                            + "ReaderOptions."
            );
        }

        return options;
    }

    private static String requireNonblank(
            String value,
            String argumentName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(
                    "LazyCustomPerFileReader requires "
                            + argumentName
                            + "."
            );
        }

        return value.trim();
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

    private static Map<String, String> copyParameters(
            Map<String, String> parameters
    ) {
        if (parameters == null || parameters.isEmpty()) {
            return Map.of();
        }

        Map<String, String> copy =
                new LinkedHashMap<>(
                        Math.max(
                                16,
                                parameters.size() * 2
                        )
                );

        for (Map.Entry<String, String> entry
                : parameters.entrySet()) {

            String name =
                    entry.getKey();

            if (name == null || name.isBlank()) {
                throw new IllegalArgumentException(
                        "Custom-reader parameter names cannot be null "
                                + "or blank."
                );
            }

            copy.put(
                    name.trim(),
                    entry.getValue()
            );
        }

        return Map.copyOf(
                copy
        );
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
     * Filename-pattern parser shared conceptually with the built-in per-file
     * readers.
     *
     * <p>The pattern must contain exactly one numeric placeholder. Glob
     * wildcards may appear outside that placeholder.</p>
     *
     * <pre>
     * instance_{num:05d}.bin
     * sample_*.{run:03d}.dat
     * proprietary_{instance}.bin
     * </pre>
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
                        "Lazy custom per-file pattern cannot be null "
                                + "or blank."
                );
            }

            Matcher matcher =
                    PLACEHOLDER_PATTERN.matcher(
                            filePattern
                    );

            if (!matcher.find()) {
                throw new IllegalArgumentException(
                        "Lazy custom per-file pattern must contain exactly "
                                + "one numeric placeholder, such as {num}, "
                                + "{num:05d}, {run}, or {instance}: "
                                + filePattern
                );
            }

            String fieldName =
                    matcher.group(
                            1
                    );

            String widthText =
                    matcher.group(
                            2
                    );

            int placeholderStart =
                    matcher.start();

            int placeholderEnd =
                    matcher.end();

            if (matcher.find()) {
                throw new IllegalArgumentException(
                        "Lazy custom per-file pattern supports exactly one "
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

            String numberRegex =
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
                            + numberRegex
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