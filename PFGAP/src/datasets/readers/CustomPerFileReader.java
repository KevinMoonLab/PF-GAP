package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import datasets.readers.api.CustomReaderContext;
import datasets.readers.interop.JavaSeriesReader;
import datasets.readers.lazy.LazySeriesRef;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Eager per-file dataset reader backed by a user-supplied Java reader plugin.
 *
 * <p>PFGAP remains responsible for:</p>
 *
 * <ul>
 *     <li>Discovering instance files</li>
 *     <li>Matching the configured filename pattern</li>
 *     <li>Sorting files deterministically</li>
 *     <li>Constructing instance references</li>
 *     <li>Invoking the custom reader</li>
 *     <li>Assembling the ListObjectDataset</li>
 *     <li>Closing the plugin and its class loader</li>
 * </ul>
 *
 * <p>The user-supplied plugin is responsible only for converting one
 * {@link LazySeriesRef} into one non-null instance object.</p>
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
 * <p>This eager reader uses the same custom plugin contract as the future
 * lazy per-file custom reader. Switching between eager and lazy reading
 * therefore does not require changes to the plugin implementation.</p>
 *
 * <p>Labels are currently assigned as null. A separate label source can be
 * added to the built-in wrapper later without changing the custom
 * per-instance reader API.</p>
 *
 * <p>The custom plugin may return any representation understood by the
 * configured distance functions. Common representations include:</p>
 *
 * <pre>
 * double[]
 * Double[]
 * Object[]
 * double[][]
 * Double[][]
 * Object[][]
 * </pre>
 */
public final class CustomPerFileReader
        implements DatasetReader {

    private static final String EAGER_READER_KEY =
            "custom-eager";

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

    /**
     * Constructs the eager custom reader from PFGAP reader options.
     *
     * <p>This constructor assumes ReaderOptions exposes:</p>
     *
     * <pre>
     * getCustomReaderDescriptor()
     * getCustomReaderParameters()
     * isCustomReaderThreadSafe()
     * </pre>
     *
     * <p>If those fields have not yet been added, use the explicit
     * constructor until the ReaderOptions wiring is implemented.</p>
     *
     * @param options PFGAP reader options
     */
    public CustomPerFileReader(
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
                options.isCustomReaderThreadSafe()
        );
    }

    /**
     * Constructs the eager custom reader with synchronized plugin invocation.
     *
     * <p>Synchronized invocation is the safe default for third-party
     * plugins. Since this initial eager implementation reads files
     * sequentially, it does not currently affect throughput.</p>
     */
    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues
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
                false
        );
    }

    /**
     * Full constructor including ordered feature-column metadata.
     *
     * @param dataPath directory containing instance files, or one regular file
     * @param filePattern required for directory input
     * @param customReaderDescriptor plugin JAR and implementation class
     * @param customReaderParameters plugin-specific configuration
     * @param featureColumns ordered selected source feature columns
     * @param isTest whether test data is being read
     * @param isRegression whether this is a regression task
     * @param isNumeric numerical-output hint
     * @param hasMissingValues missing-value hint
     * @param customReaderThreadSafe whether concurrent plugin invocation is
     *                               permitted
     */
    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            List<String> featureColumns,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe
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
                customReaderParameters == null
                        ? Map.of()
                        : Map.copyOf(
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
    }

    public CustomPerFileReader(
            String dataPath,
            String filePattern,
            String customReaderDescriptor,
            Map<String, String> customReaderParameters,
            boolean isTest,
            boolean isRegression,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean customReaderThreadSafe
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
                customReaderThreadSafe
        );
    }

    /**
     * Discovers all configured instance files and eagerly invokes the custom
     * plugin once per file.
     *
     * @return eagerly materialized dataset
     * @throws IOException if discovery, plugin loading, plugin invocation,
     *                     or plugin cleanup fails
     */
    @Override
    public ListObjectDataset read()
            throws IOException {

        List<Path> files =
                discoverFiles();

        Path configuredDataPath =
                Paths.get(
                                dataPath
                        )
                        .toAbsolutePath()
                        .normalize();

        CustomReaderContext context =
                new CustomReaderContext(
                        configuredDataPath,
                        isTest,
                        isRegression,
                        isNumeric,
                        hasMissingValues,
                        featureColumns,
                        customReaderParameters
                );

        JavaSeriesReader javaReader =
                loadJavaReader(
                        context
                );

        Throwable readingFailure =
                null;

        try {
            return materializeDataset(
                    files,
                    javaReader
            );
        } catch (Throwable failure) {
            readingFailure =
                    failure;

            throw failure;
        } finally {
            closeJavaReader(
                    javaReader,
                    readingFailure
            );
        }
    }

    private ListObjectDataset materializeDataset(
            List<Path> files,
            JavaSeriesReader javaReader
    ) {
        ListObjectDataset dataset =
                new ListObjectDataset(
                        files.size()
                );

        int commonLength =
                -1;

        boolean unequalLengths =
                false;

        boolean unknownLength =
                false;

        for (int instanceIndex = 0;
             instanceIndex < files.size();
             instanceIndex++) {

            Path file =
                    files.get(
                            instanceIndex
                    );

            LazySeriesRef reference =
                    new LazySeriesRef(
                            EAGER_READER_KEY,
                            instanceIndex,
                            file
                    );

            Object instance =
                    javaReader.read(
                            reference
                    );

            validateInstance(
                    instance,
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
                    instance,
                    instanceIndex
            );

            int instanceLength =
                    inferInstanceLength(
                            instance
                    );

            if (instanceLength < 0) {
                unknownLength =
                        true;
            } else if (commonLength < 0) {
                commonLength =
                        instanceLength;
            } else if (instanceLength != commonLength) {
                unequalLengths =
                        true;
            }

            if (instanceIndex > 0) {
                DelimitedFileReader.ProgressLogger.logProgress(
                        instanceIndex
                );
            }
        }

        int datasetLength =
                unknownLength
                        || unequalLengths
                        || commonLength < 0
                        ? 0
                        : commonLength;

        dataset.setLength(
                datasetLength
        );

        AppContext.length =
                datasetLength;

        return dataset;
    }

    private JavaSeriesReader loadJavaReader(
            CustomReaderContext context
    ) throws IOException {
        try {
            return new JavaSeriesReader(
                    customReaderDescriptor,
                    context,
                    customReaderThreadSafe
            );
        } catch (ReflectiveOperationException e) {
            throw new IOException(
                    "Could not load custom per-file reader from descriptor: "
                            + customReaderDescriptor,
                    e
            );
        }
    }

    /**
     * Closes the plugin while preserving an earlier read failure.
     *
     * <p>If reading already failed and cleanup also fails, the cleanup
     * failure is attached to the original exception as a suppressed
     * exception. If reading succeeded but cleanup fails, the cleanup failure
     * becomes the reported IOException.</p>
     */
    private void closeJavaReader(
            JavaSeriesReader javaReader,
            Throwable readingFailure
    ) throws IOException {
        try {
            javaReader.close();
        } catch (Exception closeFailure) {
            if (readingFailure != null) {
                readingFailure.addSuppressed(
                        closeFailure
                );

                return;
            }

            throw new IOException(
                    "Failed to close custom per-file reader "
                            + javaReader.getImplementationClassName()
                            + ".",
                    closeFailure
            );
        }
    }

    /**
     * Performs minimal validation without restricting proprietary in-memory
     * representations.
     */
    private void validateInstance(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (instance == null) {
            throw new IllegalStateException(
                    "Custom reader returned null for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        if (instance.getClass().isArray()
                && Array.getLength(instance) == 0) {

            throw new IllegalStateException(
                    "Custom reader returned an empty array for instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        validateRecognizedMatrixShape(
                instance,
                file,
                instanceIndex
        );
    }

    /**
     * Validates rectangularity for recognized two-dimensional Java arrays.
     *
     * <p>Unknown custom representations are deliberately accepted unchanged.
     * A custom reader and custom distance may agree on a proprietary object
     * representation that PFGAP itself does not inspect.</p>
     */
    private void validateRecognizedMatrixShape(
            Object instance,
            Path file,
            int instanceIndex
    ) {
        if (!(instance instanceof double[][])
                && !(instance instanceof Double[][])
                && !(instance instanceof Object[][])) {

            return;
        }

        int outerLength =
                Array.getLength(
                        instance
                );

        if (outerLength == 0) {
            return;
        }

        Object firstRow =
                Array.get(
                        instance,
                        0
                );

        if (firstRow == null) {
            throw new IllegalStateException(
                    "Custom reader returned a null first dimension for "
                            + "instance "
                            + instanceIndex
                            + " from file "
                            + file
                            + "."
            );
        }

        int expectedLength =
                Array.getLength(
                        firstRow
                );

        for (int dimension = 1;
             dimension < outerLength;
             dimension++) {

            Object row =
                    Array.get(
                            instance,
                            dimension
                    );

            if (row == null) {
                throw new IllegalStateException(
                        "Custom reader returned a null dimension "
                                + dimension
                                + " for instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + "."
                );
            }

            int actualLength =
                    Array.getLength(
                            row
                    );

            if (actualLength != expectedLength) {
                throw new IllegalStateException(
                        "Custom reader returned a nonrectangular array for "
                                + "instance "
                                + instanceIndex
                                + " from file "
                                + file
                                + ". Expected dimension length "
                                + expectedLength
                                + " but dimension "
                                + dimension
                                + " has length "
                                + actualLength
                                + "."
                );
            }
        }
    }

    /**
     * Infers the conventional series length for common PFGAP array types.
     *
     * @return nonnegative length, or -1 for an unknown custom representation
     */
    private int inferInstanceLength(
            Object instance
    ) {
        if (instance instanceof double[] values) {
            return values.length;
        }

        if (instance instanceof Double[] values) {
            return values.length;
        }

        if (instance instanceof Object[] values
                && !(instance instanceof Object[][])) {

            return values.length;
        }

        if (instance instanceof double[][] values) {
            return values.length == 0
                    ? 0
                    : values[0].length;
        }

        if (instance instanceof Double[][] values) {
            return values.length == 0
                    ? 0
                    : values[0].length;
        }

        if (instance instanceof Object[][] values) {
            return values.length == 0
                    ? 0
                    : values[0].length;
        }

        return -1;
    }

    /**
     * Placeholder for future per-file label sources.
     *
     * <p>The initial custom-reader milestone assigns null labels. Label
     * support can later be added through an external label file or metadata
     * mapping without changing CustomSeriesReader.</p>
     */
    private Object inferLabel(
            Path file,
            int instanceIndex
    ) {
        return null;
    }

    private List<Path> discoverFiles()
            throws IOException {

        Path path =
                Paths.get(
                                dataPath
                        )
                        .toAbsolutePath()
                        .normalize();

        if (!Files.exists(path)) {
            throw new IOException(
                    "Custom per-file data path does not exist: "
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
                    "Custom per-file data path must be a regular file or "
                            + "directory: "
                            + path
            );
        }

        if (filePattern == null) {
            throw new IllegalArgumentException(
                    "CustomPerFileReader requires filePattern when "
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
                            + " matched custom per-file pattern: "
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

    private static ReaderOptions requireOptions(
            ReaderOptions options
    ) {
        if (options == null) {
            throw new IllegalArgumentException(
                    "CustomPerFileReader requires non-null ReaderOptions."
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
                    "CustomPerFileReader requires "
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
     * <p>The pattern must contain exactly one numerical placeholder. Glob
     * wildcards are allowed outside the placeholder.</p>
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
                        "Custom per-file pattern cannot be null or blank."
                );
            }

            Matcher matcher =
                    PLACEHOLDER_PATTERN.matcher(
                            filePattern
                    );

            if (!matcher.find()) {
                throw new IllegalArgumentException(
                        "Custom per-file pattern must contain exactly one "
                                + "numeric placeholder, such as {num}, "
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
                        "Custom per-file pattern supports exactly one "
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
                            + globFragmentToRegex(prefix)
                            + numberRegex
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